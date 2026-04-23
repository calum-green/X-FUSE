from hr_dv2.high_res import HighResDV2
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from types import MethodType

import torch.nn.functional as F
import re

from timm import create_model

from functools import partial
from typing import List, Tuple, TypeAlias, Literal, Callable

from .utils import get_alibi_model


Interpolation: TypeAlias = Literal[
    "nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"
]
AttentionOptions: TypeAlias = Literal["q", "k", "v", "o", "none"]
FusionOptions: TypeAlias = Literal["gating", "learned_gating", "attention"]


class XRDCrossAttention(nn.Module):
    """Cross-attention: XRD attends to feature space to find relevant features."""

    def __init__(self, feat_dim: int = 384, num_heads: int = 8):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads

        # Project XRD intensity to query space
        self.xrd_to_query = nn.Linear(1, feat_dim)

        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )

    def forward(self, features: torch.Tensor, xrd_map: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention: XRD queries the feature space.

        Args:
            features: (B, C, H, W) DINO features
            xrd_map: (1, 1, H, W) or (H, W) XRD intensity map

        Returns:
            attended_features: (B, C, H, W) features modulated by XRD attention
        """
        B, C = features.shape[:2]

        # Ensure xrd_map has same spatial dims as features
        if xrd_map.dim() == 2:
            xrd_map = xrd_map.unsqueeze(0).unsqueeze(0)
        elif xrd_map.dim() == 3:
            xrd_map = xrd_map.unsqueeze(0)

        # Flatten spatial dimensions
        feat_flat = features.reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        xrd_flat = xrd_map.reshape(1, -1, 1)  # (1, H*W, 1)

        # Project XRD to query space
        queries = self.xrd_to_query(xrd_flat)  # (1, H*W, feat_dim)

        # Expand features batch dimension to match queries if needed
        if B > 1:
            # Repeat queries for each batch
            queries = queries.expand(B, -1, -1)  # (B, H*W, feat_dim)
            feat_batch = feat_flat  # (B, H*W, feat_dim)
        else:
            feat_batch = feat_flat  # (B, H*W, feat_dim)

        # Cross-attention: XRD queries attend to feature space
        attended, _ = self.attention(queries, feat_batch, feat_batch)  # (B, H*W, C)

        # Reshape back to spatial dimensions
        attended = attended.permute(0, 2, 1).reshape(features.shape)  # (B, C, H, W)

        return attended


LossOptions: TypeAlias = Literal["pearson", "bce"]


class LearnedChannelGating(nn.Module):
    """Linear probe to rank DINO channels by XRD relevance, then hard-select top-K."""

    def __init__(self, feat_dim: int = 384, loss_fn: LossOptions = "bce"):
        super().__init__()
        self.feat_dim = feat_dim
        self.loss_fn = loss_fn
        self.pred_head = nn.Linear(feat_dim, 1)
        self._channel_mask: torch.Tensor | None = None

    def compute_loss(
        self, features: torch.Tensor, xrd_map: torch.Tensor
    ) -> torch.Tensor:
        """Train the linear probe on raw (un-gated) features."""
        B, C, H, W = features.shape
        feat_flat = features.float().reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        pred = self.pred_head(feat_flat).squeeze(-1)  # (B, H*W)
        target = xrd_map.reshape(1, -1).expand(B, -1).float()

        if self.loss_fn == "bce":
            n_pos = target.sum().clamp(min=1)
            n_neg = (1 - target).sum().clamp(min=1)
            pos_weight = (n_neg / n_pos).to(pred.device)
            return F.binary_cross_entropy_with_logits(
                pred, target, pos_weight=pos_weight
            )

        pred_c = pred - pred.mean(dim=-1, keepdim=True)
        target_c = target - target.mean(dim=-1, keepdim=True)
        corr = (pred_c * target_c).sum(dim=-1) / (
            pred_c.norm(dim=-1) * target_c.norm(dim=-1) + 1e-8
        )
        return (1 - corr).mean()

    def select_top_k(self, k: int) -> None:
        """Derive a hard channel mask from probe weights after training.

        Channels with the highest absolute weight in pred_head are the ones
        the linear probe found most predictive of the XRD signal.
        """
        weights = self.pred_head.weight.data.squeeze(0).abs()  # (feat_dim,)
        top_k = weights.topk(k).indices
        mask = torch.zeros(self.feat_dim, device=weights.device)
        mask[top_k] = 1.0
        self._channel_mask = mask

    def forward(self, features: torch.Tensor, _xrd_map: torch.Tensor) -> torch.Tensor:
        """Zero out all channels not in the top-K mask."""
        if self._channel_mask is None:
            return features
        mask = self._channel_mask.to(features.device).view(1, -1, 1, 1)
        return features * mask


class XRDFusionMethod(nn.Module):
    def __init__(
        self,
        xrd_img: None,  # normalised XRD image from xrd_to_tensor
        transform: partial,
        require_grad: bool = False,
        learned_gating: nn.Module = None,
        spatial_attention: nn.Module = None,
        loss_fn: LossOptions = "pearson",
    ):
        super().__init__()

        self.require_grad = require_grad
        # load the XRD image as a normalised tensor
        self.xrd_image = xrd_img
        self.transform = transform
        self.learned_gating_module = learned_gating
        self.spatial_attention_module = spatial_attention
        self.loss_fn = loss_fn

    def get_tr(self) -> torch.Tensor:
        """Apply all transforms to XRD data and return as a batch.
        Mirrors get_transformed_input_batch.
        """
        if len(self.transform) == 0:
            return self.xrd_image.unsqueeze(0)  # (1, C, H, W)
        xrd_list = [t(self.xrd_image) for t in self.transform]
        return torch.stack(xrd_list)  # (N_transforms, C, H, W)

    def __translen__(self, transform: partial):
        return len(transform)

    def forward_xrd(
        self,
        x: torch.Tensor,  # [B,C,H,W] transformed DINO features
        idx: int,  # index of the transform applied to the input image
        xrd_fusion_method: FusionOptions,
        top_k: int | None = None,
    ):
        # return transformed XRD features ready to be fused with the DINO features
        tr_xrd = self.get_tr()[idx]

        if xrd_fusion_method == "gating":
            return self._direct_correlation_gating(x, tr_xrd, top_k=top_k)

        elif xrd_fusion_method == "learned_gating":
            return self._learned_channel_gating(x, tr_xrd)

        elif xrd_fusion_method == "attention":
            return self._spatial_attention(x, tr_xrd)

    def _direct_correlation_gating(
        self, features: torch.Tensor, xrd_map: torch.Tensor, top_k: int | None = None
    ) -> torch.Tensor:
        """
        Compute per-channel score against XRD map, use as soft gate weights.

        Pearson: weight = correlation clipped to [0, 1] (higher = better match).
        BCE: weight = 1 - normalised_loss (lower loss = better match = higher weight).

        If top_k is set, only the top_k scoring channels are kept (others zeroed).
        Defaults to C // 4.

        Args:
            features: (B, C, H, W) DINO features
            xrd_map: (1, 1, H, W)
            top_k: number of channels to keep (default: C // 4)

        Returns:
            gated_features: (B, C, H, W) with gates applied per channel
        """
        B, C, H, W = features.shape
        B_xrd, C_xrd, H_xrd, W_xrd = xrd_map.shape

        if top_k is None:
            top_k = C // 4

        # Ensure xrd_map has same spatial dims as features
        xrd_map = F.interpolate(
            xrd_map.to(features.dtype),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # Flatten spatial dimensions for correlation computation
        feat_flat = features.float().reshape(B, C, -1)  # (B, C, H*W)
        xrd_flat = xrd_map.reshape(-1).float()  # (H*W,)

        # Compute per-channel score against XRD
        gates = []
        for ch in range(C):
            # Score each channel across all batch and spatial locations
            feat_ch = feat_flat[:, ch, :].reshape(-1)  # (B*H*W,)

            if self.loss_fn == "pearson":
                if feat_ch.shape[0] > 1:
                    score = torch.corrcoef(torch.stack([feat_ch, xrd_flat]))[0, 1]
                else:
                    score = torch.tensor(0.0, device=features.device)
                score = torch.nan_to_num(score, nan=0.0).clamp(0, 1)

            else:  # bce
                # Normalise channel to [0, 1] for BCE
                ch_min, ch_max = feat_ch.min(), feat_ch.max()
                ch_norm = (feat_ch - ch_min) / (ch_max - ch_min + 1e-8)
                bce = F.binary_cross_entropy(ch_norm, xrd_flat, reduction="mean")
                # Invert so high score = good match; max BCE is log(2) ≈ 0.693
                score = (1 - bce / 0.693).clamp(0, 1)

            gates.append(score)

        gates = torch.stack(gates).to(features.dtype)  # (C,)

        # Zero all but the top_k scoring channels
        threshold = gates.topk(top_k).values[-1]
        gates = gates * (gates >= threshold)

        # Apply gates: multiply each channel by its gate weight
        gated = features * gates.view(1, C, 1, 1)  # Broadcasting

        return gated

    def _learned_channel_gating(
        self, features: torch.Tensor, xrd_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Learned channel gating: MLP learns to map XRD intensity → channel gates.

        Args:
            features: (B, C, H, W) DINO features
            xrd_map: (1, 1, H, W) or (H, W) XRD intensity map

        Returns:
            gated_features: (B, C, H, W)
        """
        if not hasattr(self, "learned_gating_module"):
            self.learned_gating_module = LearnedChannelGating(
                feat_dim=features.shape[1]
            ).to(device=features.device, dtype=features.dtype)

        _, _, H, W = features.shape
        if xrd_map.dim() == 2:
            xrd_map = xrd_map.unsqueeze(0).unsqueeze(0)
        elif xrd_map.dim() == 3:
            xrd_map = xrd_map.unsqueeze(0)
        xrd_map = F.interpolate(
            xrd_map.to(features.dtype),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        return self.learned_gating_module(features, xrd_map)

    def _spatial_attention(
        self, features: torch.Tensor, xrd_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Spatial attention: XRD-guided cross-attention over features.

        Args:
            features: (B, C, H, W) DINO features
            xrd_map: (1, 1, H, W) or (H, W) XRD intensity map

        Returns:
            attended_features: (B, C, H, W)
        """
        if not hasattr(self, "spatial_attention_module"):
            self.spatial_attention_module = XRDCrossAttention(
                feat_dim=features.shape[1], num_heads=8
            ).to(device=features.device, dtype=features.dtype)

        _, _, H, W = features.shape
        if xrd_map.dim() == 2:
            xrd_map = xrd_map.unsqueeze(0).unsqueeze(0)
        elif xrd_map.dim() == 3:
            xrd_map = xrd_map.unsqueeze(0)
        xrd_map = F.interpolate(
            xrd_map.to(features.dtype),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        return self.spatial_attention_module(features, xrd_map)


class XFuse(HighResDV2):
    def __init__(
        self,
        dino_name: str,
        xrd_fuse_method: str,
        stride: int,
        pca_dim: int,
        track_grad: bool = False,
        dtype: torch.dtype | int = torch.float16,
        loss_fn: LossOptions = "bce",
        *args,
        **kwargs,
    ):
        """
        I want this XFuse class to inherit from HighResDV2 and have the
        same functionality, however, I will add the extra modules that
        perform the XRD fusion, given an XCT input.
        """
        # pass to HighResDV2 init
        super().__init__(
            dino_name,
            stride=stride,
            pca_dim=pca_dim,
            track_grad=track_grad,
            dtype=dtype,
            *args,
            **kwargs,
        )

        self.dinov2: nn.Module
        device = kwargs.get("device", None)

        if "alibi" in dino_name:
            dino_name = dino_name
            model_path = kwargs["model_path"]
            self.dinov2 = get_alibi_model(dino_name, model_path, device=device, stride=stride)

        elif "dinov2" in dino_name:
            hub_path = "facebookresearch/dinov2"
            self.dinov2 = torch.hub.load(hub_path, dino_name)
        elif "dinov3" in dino_name:
            lib_path = kwargs["lib_path"]
            chk_path = kwargs["chk_path"]
            assert (chk_path is not None) and (
                lib_path is not None
            ), "Must supply a local checkpoint for DINOv3!"
            self.dinov2 = torch.hub.load(
                lib_path, dino_name, source="local", weights=chk_path
            )
        elif "dino" in dino_name:
            hub_path = "facebookresearch/dino:main"
            self.dinov2 = torch.hub.load(hub_path, dino_name)
        elif "deit" in dino_name:
            self.dinov2 = create_model(  # was 224
                "deit_small_patch16_224", pretrained=True
            )
        elif "384" in dino_name:
            self.dinov2 = create_model(  # was 224
                "vit_small_patch16_384", pretrained=True
            )
        else:
            self.dinov2 = create_model(  # was 224
                "vit_small_patch16_224", pretrained=True
            )

        # self.dinov2: nn.Module = torch.hub.load("facebookresearch/dinov2", dino_name)

        self.dinov2.eval()

        if "dinov2" not in dino_name:
            self.dinov2.num_heads = 6  # type: ignore
            self.dinov2.num_register_tokens = 0  # type: ignore

        # Get params of Dv2 model and store references to original settings & methods
        feat, patch, n_heads = self.get_model_params(dino_name)
        self.original_patch_size: int = patch
        self.original_stride = _pair(patch)
        # may need to deepcopy this instead of just referencing
        # self.original_pos_enc = self.dinov2.interpolate_pos_encoding
        self.feat_dim: int = feat
        self.n_heads: int = n_heads
        self.n_register_tokens = 4

        self.stride = _pair(stride)
        # we need to set the stride to the original once before set it to desired stride
        # i don't know why
        self.set_model_stride(self.dinov2, patch)
        self.set_model_stride(self.dinov2, stride)

        self.transforms: List[partial] = []
        self.inverse_transforms: List[partial] = []
        self.interpolation_mode: Interpolation = "nearest-exact"
        self.pca_dim = pca_dim
        self.do_pca = pca_dim > 3

        # define the fusion method for fusing the XRD features with the DINO features
        self.xrd_fuse_method = xrd_fuse_method
        self.loss_fn = loss_fn

        # Initialize fusion modules based on method
        if xrd_fuse_method == "learned_gating":
            self.learned_gating = LearnedChannelGating(
                feat_dim=self.feat_dim, loss_fn=loss_fn
            )
        else:
            self.learned_gating = None

        if xrd_fuse_method == "attention":
            self.spatial_attention = XRDCrossAttention(
                feat_dim=self.feat_dim, num_heads=8
            )
        else:
            self.spatial_attention = None

        # XRDFusionMethod is created in set_transforms() once transforms are known
        self.xrd_fusion_module = None

        # If we want to save memory, change to float16
        if type(dtype) is int:
            dtype = torch.float16 if dtype == 16 else torch.float32

        self.dtype = dtype
        if dtype != torch.float32:
            self.dinov2 = self.dinov2.to(dtype)
        if device is not None:
            self.dinov2 = self.dinov2.to(device)
            if self.learned_gating is not None:
                self.learned_gating = self.learned_gating.to(device)
        self.track_grad = track_grad  # off by default to save memory

        if self.learned_gating is not None:
            self.fusion_optimizer = torch.optim.Adam(
                self.learned_gating.parameters(), lr=1e-4
            )
        else:
            self.fusion_optimizer = None

        self.patch_last_block(self.dinov2, dino_name)

    def set_model_stride(self, dino_model: nn.Module, stride_l: int, verbose: bool = False) -> None:
        try:
            new_stride_pair = _pair(stride_l)
            self.stride = new_stride_pair
            dino_model.model.patch_embed.proj.stride = new_stride_pair
            dino_model.stride = stride_l
            if verbose:
                print(f"Setting stride to ({stride_l},{stride_l})")
        except AttributeError:
            super().set_model_stride(dino_model, stride_l, verbose)

    def patch_last_block(self, dino_model: nn.Module, dino_name: str) -> None:
        if "alibi" not in dino_name:
            super().patch_last_block(dino_model, dino_name)
            return

        def forward_feats_attn(self_model, x, masks=None, attn_choice="none"):
            feats = self_model.forward_features(x)  # (B, C, N_patches)
            feats = feats.permute(0, 2, 1)  # (B, N_patches, C)
            return {"x_norm_patchtokens": feats, "masks": masks}

        dino_model.forward_feats_attn = MethodType(forward_feats_attn, dino_model)

    def get_model_params(self, dino_name: str) -> Tuple[int, int, int]:
        for segment in dino_name.split("_"):
            m = re.match(r"^vit([sblg])(\d+)$", segment)
            if m:
                arch, patch_size = m.group(1), int(m.group(2))
                feat_dim_lookup = {"s": 384, "b": 768, "l": 1024, "g": 1536}
                n_heads_lookup = {"s": 6, "b": 12, "l": 16, "g": 16}
                return feat_dim_lookup[arch], patch_size, n_heads_lookup[arch]
        return super().get_model_params(dino_name)

    @torch.no_grad()
    def forward_sequential(
        self,
        x: torch.Tensor,
        attn_choice: AttentionOptions = "none",
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Perform transform -> featurise -> upscale -> inverse -> average forward pass
        sequentially, performing more calls to DINOv2 but reducing the memory overhead.

        :param x: unbatched image tensor
        :type x: torch.Tensor
        :param top_k: for 'gating' method, number of channels to keep (default: C // 4)
        :type top_k: int | None
        :return: tuple of low-res Dv2 features and our upsample high-res Dv2 features
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        x.requires_grad = self.track_grad
        if self.dtype != torch.float32:  # cast (i.e to f16)
            x = x.type(self.dtype)
        img_batch = self.get_transformed_input_batch(x, self.transforms)
        temp_stride = self.stride

        _, img_h, img_w = x.shape

        if attn_choice != "none":
            c = self.n_heads + self.feat_dim
        else:
            c = self.feat_dim

        stride_l = temp_stride[0]
        n_patch_w: int = 1 + (img_w - self.original_patch_size) // stride_l
        n_patch_h: int = 1 + (img_h - self.original_patch_size) // stride_l

        out_feature_img: torch.Tensor = torch.zeros(
            1,
            c,
            img_h,
            img_w,
            device=x.device,
            dtype=self.dtype,
            requires_grad=self.track_grad,
        )

        N_transforms = len(self.transforms)
        for i in range(N_transforms):
            transformed_img = img_batch[i].unsqueeze(0)
            out_dict = self.dinov2.forward_feats_attn(
                transformed_img, None, attn_choice
            )  # type: ignore
            if attn_choice != "none":
                feats, attn = out_dict["x_norm_patchtokens"], out_dict["x_patchattn"]
                features = torch.concat((feats, attn), dim=-1)
            else:
                features = out_dict["x_norm_patchtokens"]

            features = features.squeeze(0)
            feat_patch = features.view((n_patch_h, n_patch_w, c))
            permuted = feat_patch.permute((2, 0, 1)).unsqueeze(0)
            fused_img = self.xrd_fusion_module.forward_xrd(
                permuted, i, self.xrd_fuse_method, top_k=top_k
            )

            full_size = F.interpolate(
                fused_img,
                (img_h, img_w),
                mode=self.interpolation_mode,
            )
            inv_transform = self.inverse_transforms[i]
            inverted: torch.Tensor = inv_transform(full_size)
            out_feature_img += inverted

        mean = out_feature_img / N_transforms
        return mean

    def train_fusion_step(self, x: torch.Tensor) -> float:
        """Update LearnedChannelGating weights for one step.

        DINO backbone is frozen throughout. Only the gating MLP and
        prediction head are updated.

        :param x: unbatched XCT image tensor (C, H, W)
        :return: mean MSE loss across transforms
        """
        assert (
            self.xrd_fuse_method == "learned_gating"
        ), "train_fusion_step only applies to learned_gating"
        assert (
            self.xrd_fusion_module is not None
        ), "Call set_xrd_transforms before training"

        if self.dtype != torch.float32:
            x = x.type(self.dtype)

        _, img_h, img_w = x.shape
        stride_l = self.stride[0]
        n_patch_w = 1 + (img_w - self.original_patch_size) // stride_l
        n_patch_h = 1 + (img_h - self.original_patch_size) // stride_l

        with torch.no_grad():
            img_batch = self.get_transformed_input_batch(x, self.transforms)

        N_transforms = len(self.transforms)
        total_loss = torch.tensor(0.0, dtype=torch.float32)

        self.fusion_optimizer.zero_grad()

        for i in range(N_transforms):
            # Extract DINO features — backbone frozen
            with torch.no_grad():
                transformed_img = img_batch[i].unsqueeze(0)
                out_dict = self.dinov2.forward_feats_attn(
                    transformed_img, None, "none"
                )  # type: ignore
                features = out_dict["x_norm_patchtokens"].squeeze(0)
                feat_patch = features.view((n_patch_h, n_patch_w, self.feat_dim))
                permuted = feat_patch.permute((2, 0, 1)).unsqueeze(0)  # (1, C, H, W)

            # Interpolate XRD map to match feature spatial dims
            tr_xrd = self.xrd_fusion_module.get_tr()[i]
            _, _, H, W = permuted.shape
            if tr_xrd.dim() == 3:
                tr_xrd = tr_xrd.unsqueeze(0)
            tr_xrd = F.interpolate(
                tr_xrd.to(permuted.dtype),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

            # Train probe on raw features — no gating during training
            loss = self.learned_gating.compute_loss(permuted, tr_xrd)
            total_loss = total_loss + loss.cpu().float()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(self.learned_gating.parameters(), max_norm=1.0)
        self.fusion_optimizer.step()
        return (total_loss / N_transforms).item()

    def set_xrd_transforms(self, xrd_tensor, fwd, inv):
        super().set_transforms(fwd, inv)
        assert len(fwd) == len(
            inv
        ), "Forward and inverse transform lists must be same length!"
        self.xrd_fusion_module = XRDFusionMethod(
            xrd_img=xrd_tensor,
            transform=self.transforms,
            require_grad=self.track_grad,
            learned_gating=self.learned_gating,
            spatial_attention=self.spatial_attention,
            loss_fn=self.loss_fn,
        )
