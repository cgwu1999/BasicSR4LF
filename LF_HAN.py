#### This file is for the BasicLFSR, put the file under BasicLFSR/model/SR

from basicsr.archs.LF_CSwin_arch import CST_
from torch import nn
from einops import rearrange


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.ang = args.angRes_in
        self.scale = args.scale_factor
        self.module = CST_(upscale=self.scale)

    def forward(self, lr, data_info=None):
        lr = rearrange(lr, "B C (U H) (V W)-> B C U V H W", U=self.ang, V=self.ang)
        out = self.module(lr)
        return rearrange(out, "B C U V H W-> B C (U H) (V W)")
