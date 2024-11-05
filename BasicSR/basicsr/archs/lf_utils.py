import math
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, to_3tuple


def interpolate(x, scale_factor, mode="bilinear"):
    B, C, U, V, H, W = x.shape
    x = rearrange(x, "B C U V H W -> (B U V) C H W")
    x = F.interpolate(x, scale_factor=scale_factor, mode=mode)
    x = rearrange(x, "(B U V) C H W -> B C U V H W ", B=B, U=U, V=V)
    return x


class Conv2d(nn.Conv2d):
    def forward(self, x):
        B, C, U, V, H, W = x.size()
        x = rearrange(x, "B C U V H W -> (B U V) C H W")
        x = self._conv_forward(x, self.weight, self.bias)
        return rearrange(x, "(B U V) C H W -> B C U V H W", B=B, U=U, V=V)


# class PixelShuffle(nn.PixelShuffle):
#     def forward(self, x):
#         B, C, U, V, H, W = x.size()
#         x = rearrange(x, "B C U V H W -> (B U V) C H W")
#         x = F.pixel_shuffle(x, self.upscale_factor)
#         return rearrange(x, "(b u v ) c h w -> B C U V H W", B=B, U=U, V=V)


# class LayerNormFunction(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, x, weight, bias, eps):
#         ctx.eps = eps
#         N, C, U, V, H, W = x.size()
#         mu = x.mean(1, keepdim=True)
#         var = (x - mu).pow(2).mean(1, keepdim=True)
#         y = (x - mu) / (var + eps).sqrt()
#         ctx.save_for_backward(y, var, weight)
#         y = weight.view(1, C, 1, 1, 1, 1) * y + bias.view(1, C, 1, 1, 1, 1)
#         return y

#     @staticmethod
#     def backward(ctx, grad_output):
#         eps = ctx.eps

#         N, C, H, W = grad_output.size()
#         y, var, weight = ctx.saved_variables
#         g = grad_output * weight.view(1, C, 1, 1, 1, 1)
#         mean_g = g.mean(dim=1, keepdim=True)

#         mean_gy = (g * y).mean(dim=1, keepdim=True)
#         gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
#         return (
#             gx,
#             (grad_output * y).sum([0, 2, 3, 4, 5]),
#             grad_output.sum([0, 2, 3, 4, 5]),
#             None,
#         )


# class LayerNorm(nn.Module):

#     def __init__(self, channels, eps=1e-6):
#         super(LayerNorm, self).__init__()
#         self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
#         self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
#         self.eps = eps

#     def forward(self, x):
#         return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            if out.dim() == 6:
                out = out * self.weight.view(1, -1, 1, 1, 1, 1) + self.bias.view(
                    1, -1, 1, 1, 1, 1
                )
            elif out.dim() == 4:
                out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
            else:
                raise ValueError(
                    "Unsupported dimension of input tensor for layer norm."
                )
        return out


# class ConvLayer(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size=3,
#         stride=1,
#         dilation=1,
#         groups=1,
#         use_bias=False,
#         norm=LayerNorm,
#         act_func=nn.SiLU,
#     ):
#         super().__init__()
#         self.conv = Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=(kernel_size // 2) * dilation,
#             dilation=dilation,
#             groups=groups,
#             bias=use_bias,
#             # padding_mode="reflect",
#         )
#         self.norm = norm(out_channels) if norm is not None else nn.Identity()
#         self.act_func = (
#             act_func(inplace=True) if act_func is not None else nn.Identity()
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.norm(x)
#         x = self.act_func(x)
#         return x


# class MBConv(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size=3,
#         stride=1,
#         mid_channels=None,
#         expand_ratio=4,
#         use_bias=False,
#         norm=LayerNorm,
#         act_func=(nn.SiLU, nn.SiLU, None),
#     ):
#         super().__init__()

#         use_bias = to_3tuple(use_bias)
#         norm = to_3tuple(norm)
#         act_func = to_3tuple(act_func)
#         mid_channels = mid_channels or round(in_channels * expand_ratio)

#         self.inverted_conv = ConvLayer(
#             in_channels,
#             mid_channels,
#             1,
#             stride=1,
#             norm=norm[0],
#             act_func=act_func[0],
#             use_bias=use_bias[0],
#         )
#         self.depth_conv = ConvLayer(
#             mid_channels,
#             mid_channels,
#             kernel_size,
#             stride=stride,
#             groups=mid_channels,
#             norm=norm[1],
#             act_func=act_func[1],
#             use_bias=use_bias[1],
#         )
#         self.point_conv = ConvLayer(
#             mid_channels,
#             out_channels,
#             1,
#             norm=norm[2],
#             act_func=act_func[2],
#             use_bias=use_bias[2],
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.inverted_conv(x)
#         x = self.depth_conv(x)
#         x = self.point_conv(x)
#         return x

class Star(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x1,x2=torch.chunk(x,2,1)
        return x1*x2

class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=4,
        use_bias=False,
        norm=(LayerNorm,LayerNorm,None),
        act_func=(nn.SiLU,nn.SiLU,None),
        star=False,
    ):
        super().__init__()
        self.star=star
        use_bias = to_3tuple(use_bias)
        norm = to_3tuple(norm)
        act_func = to_3tuple(act_func)
        mid_channels = mid_channels or round(in_channels * expand_ratio)
        star_channels = mid_channels//2 if star else mid_channels
        net = (
            [
                nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=use_bias[0]),
                norm[0](mid_channels) if norm[0] else nn.Identity(),
                act_func[0]() if act_func[0] else nn.Identity(),
                nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=mid_channels,
                    bias=use_bias[1],
                ),
                norm[1](mid_channels) if norm[1] else nn.Identity(),
                act_func[1]() if act_func[1] else nn.Identity(),
            ]
            + ([Star()] if star else [])
            + [
                nn.Conv2d(star_channels, out_channels, 1, 1, 0, bias=use_bias[2]),
                norm[2](out_channels) if norm[2] else nn.Identity(),
                act_func[2]() if act_func[2] else nn.Identity(),
            ]
        )

        self.net = nn.Sequential(*net)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, U, V, H, W = x.size()
        x = rearrange(x, "B C U V H W -> (B U V) C H W")
        x = self.net(x)
        return rearrange(x, "(B U V) C H W -> B C U V H W", B=B, U=U, V=V)


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, seseparate=False):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                if seseparate:
                    m.append(nn.Conv2d(num_feat, 4 * num_feat, 1, 1, 0))
                    m.append(
                        nn.Conv2d(
                            4 * num_feat, 4 * num_feat, 3, 1, 1, groups=4 * num_feat
                        )
                    )
                else:
                    m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. Supported scales: 2^n and 3."
            )
        super(Upsample, self).__init__(*m)

    def forward(self, x, *args, **kwargs):
        B, C, U, V, H, W = x.size()
        x = rearrange(x, "B C U V H W -> (B U V) C H W")
        x = super(Upsample, self).forward(x, *args, **kwargs)
        return rearrange(x, "(B U V) C H W -> B C U V H W", B=B, U=U, V=V)


# class LFDataWarp:
#     def __init__(self, func):
#         self.func = func
#     def __call__(self, obj,x):
#         _ , _ , u , v , _ , _=x.shape
#         x=rearrange(x, "B C U V H W -> B C (U H) (V W)")
#         y= self.func(obj, x)
#         return rearrange(y, "B C (U H) (V W) -> B C U V H W", U=u, V=v)


def LFDataWarp(func):
    def wrapper(self, x):
        _ , _ , u , v , _ , _=x.shape
        x=rearrange(x, "B C U V H W -> B C (U H) (V W)")
        return rearrange(func(self, x), "B C (U H) (V W) -> B C U V H W", U=u, V=v)
    return wrapper
