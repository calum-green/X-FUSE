from hr_dv2.high_res import HighResDV2


class XFuse(HighResDV2):
    def __init__(self, xrd_fuse_method, *args, **kwargs):
        """
        I want this XFuse class to inherit from HighResDV2 and have the
        same functionality, however, I will add the extra modules that
        perform the XRD fusion, given an XCT input.
        """
        super().__init__(*args, **kwargs)
        self.xrd_fuse_method = xrd_fuse_method

    def get_dino_feats(self, **kwargs):
        pass

    def fuse_xrd(self, xrd_fuse_method, **kwargs):
        pass

    def get_upsampled_feats(self, **kwargs):
        pass
