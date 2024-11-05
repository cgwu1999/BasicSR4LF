import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast
from timm.layers import to_2tuple, to_3tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY
import math
from .lf_utils import *


# from timm.models.swin_transformer import WindowAttention


def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(
        torch.meshgrid([torch.arange(win_h), torch.arange(win_w)])
    )  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = (
        coords_flatten[:, :, None] - coords_flatten[:, None, :]
    )  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


# class CA(nn.Sequential):
#     def __init__(self, dim, aexs=[2, 3], r=4, act=nn.ReLU, weight=0.1):
#         self.axes = aexs
#         mid_dim = dim // r
#         super().__init__(
#             nn.Conv2d(dim, mid_dim, 1, 1, 0),
#             act(False) if act else nn.Identity(),
#             nn.Conv2d(mid_dim, dim, 1, 1, 0),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         dim = self.axes
#         x = x.mean(dim=dim)
#         # x = x.mean(dim=dim[-2])
#         x = super().forward(x)
#         x = x.unsqueeze(dim=dim[0])
#         x = x.unsqueeze(dim=dim[1])
#         return x.contiguous()


# class MCA_(nn.Module):

#     def __init__(self, dim):
#         super().__init__()
#         self.weights = nn.Parameter(torch.ones(4) * 0.1)
#         self.uvca = CA(dim, aexs=[2, 3])
#         self.uhca = CA(dim, aexs=[2, 4])
#         self.vwca = CA(dim, aexs=[3, 5])
#         self.hwca = CA(dim, aexs=[4, 5])

#     def forward(self, x):
#         # return 0
#         uv = self.uvca(x) * self.weights[0]
#         uh = self.uhca(x) * self.weights[1]
#         vw = self.vwca(x) * self.weights[2]
#         hw = self.hwca(x) * self.weights[3]
#         return x * uv + x * uh + x * vw + x * hw * self.weights


# def one2four(uw_vh, u, v, h, w):
#     h_w = uw_vh[..., :h, :w]
#     v_w = uw_vh[..., h:, :w]
#     u_h = uw_vh[..., :h, w:].transpose(-1, -2)
#     u_v = uw_vh[..., h:, w:].transpose(-1, -2)
#     return h_w, u_v, u_h, v_w

# class SCA(nn.Module):
#     def __init__(self,dim,mid_dim=None, act=nn.SiLU, weight=0.1, d='s'):
#         super().__init__()
#         mid_dim = dim if mid_dim is None else mid_dim
#         self.body = nn.Sequential(
#             nn.Conv2d(dim, mid_dim, 1, 1, 0),
#             act(inplace=True) if act else nn.Identity(),
#             nn.Conv2d(mid_dim, dim, 1, 1, 0),
#             nn.Conv2d(mid_dim, dim, 1, 1, 0),
#             # act(inplace=True) if act else nn.Identity(),
#             # nn.Sigmoid(),
#         )
#         self.d=d
#         nn.init.constant_(self.body[2].weight, 0)
#         nn.init.constant_(self.body[2].bias, 0)
#     def forward(self,x):
#         shortcut=x
#         if self.d=='a':
#             x = x.mean([2, 3])
#             x=self.body(x)
#             return x.unsqueeze(2).unsqueeze(3)*shortcut
#         elif self.d=='s':
#             x = x.mean([4, 5])
#             x=self.body(x)
#             return x.unsqueeze(4).unsqueeze(5)*shortcut
#         else:
#             x1 = x.mean([2, 4])
#             x2 = x.mean([3, 5])
#             x1=self.body(x1)
#             x2=self.body(x2)
#             return x1.unsqueeze(2).unsqueeze(4)*shortcut+x2.unsqueeze(3).unsqueeze(5)*shortcut

class MCA(nn.Module):
    def __init__(self, dim, mid_dim=None, act=nn.SiLU, weight=0.1):
        super().__init__()
        mid_dim = dim if mid_dim is None else mid_dim
        self.body = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0),
            act(inplace=True) if act else nn.Identity(),
            nn.Conv2d(mid_dim, dim, 1, 1, 0),
            # act(inplace=True) if act else nn.Identity(),
            # nn.Sigmoid(),
        )
        self.fuses = nn.ModuleList([nn.Conv2d(dim, dim, 1, 1, 0) for _ in range(4)])

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        B, C, U, V, H, W = x.shape

        uvhw = torch.empty((B, C, H + V, U + W), device=x.device, dtype=x.dtype)
        uvhw[..., H:, W:] = x.mean([4, 5]).transpose(-1, -2)
        uvhw[..., :H, W:] = x.mean([3, 5]).transpose(-1, -2)
        uvhw[..., :H, :W] = x.mean([2, 3])
        uvhw[..., H:, :W] = x.mean([2, 4])

        uvhw = self.body(uvhw)

        hw = uvhw[..., :H, :W]
        vw = uvhw[..., H:, :W]
        uh = uvhw[..., :H, W:].transpose(-1, -2)
        uv = uvhw[..., H:, W:].transpose(-1, -2)

        #wcg: no flash
        # hw=self.body(x.mean([2,3]))
        # vw=self.body(x.mean([2,4]))
        # uh=self.body(x.mean([3,5]))
        # uv=self.body(x.mean([4,5]))

        hw = self.fuses[0](hw)
        uv = self.fuses[1](uv)
        uh = self.fuses[2](uh)
        vw = self.fuses[3](vw)

        hw = hw.reshape(B, C, 1, 1, H, W)
        uv = uv.reshape(B, C, U, V, 1, 1)
        uh = uh.reshape(B, C, U, 1, H, 1)
        vw = vw.reshape(B, C, 1, V, 1, W)
        # print(uh, vw)
        return (
            x * hw + 
            x * uv + 
            x * uh + 
            x * vw
                )
        # return x*hw*uv*uh*vw
        return x*hw+x*uv+x*uh+x*vw


class WindowAttention(nn.Module):
    # idx=0
    def __init__(
        self,
        dim,
        num_heads,
        window="(B U V H W) (wv wh) C",
        shift_size=0,
        window_size=8,
        pos=False,
        pos_repeat=1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.window = window
        self.shift_size = shift_size
        self.window_size = to_2tuple(window_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self.pos = pos
        self.pos_repeat = pos_repeat
        if self.pos:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                    num_heads,
                )
            )
            self.register_buffer(
                "relative_position_index",
                get_relative_position_index(self.window_size[0], self.window_size[1]),
            )
            trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_area, self.window_area, -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward_feat(self, x):
        B, C, U, V, H, W = x.shape
        x = rearrange(
            x,
            f"B C U V (H wv) (W wh)-> {self.window}",
            wv=self.window_size[0],
            wh=self.window_size[1],
            U=U,
            V=V,
            H=H // self.window_size[0],
            W=W // self.window_size[1],
        )
        pos_repeat = self.pos_repeat
        qkv = rearrange(self.qkv(x), "b s (h c)-> b h s c", h=self.num_heads)
        # if self.window=="(B H wv W wh) (U V) C":
        #     torch.save(
        #         qkv.detach(), f"/opt/data/private/LFSR/T4LF/attn/qkv{WindowAttention.idx}.pt"
        #     )
        #     WindowAttention.idx = WindowAttention.idx + 1
        #     print(WindowAttention.idx)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q * self.scale
        attn = q @ k.transpose(-1, -2)
        if self.pos:
            attn += self._get_rel_pos_bias().repeat(1, 1, pos_repeat, pos_repeat)

        attn = self.softmax(attn)
        x = rearrange(attn @ v, "b h s c ->b s (h c)", h=self.num_heads)
        x = self.proj(x)
        x = rearrange(
            x,
            f"{self.window} -> B C U V (H wv) (W wh)",
            wv=self.window_size[0],
            wh=self.window_size[1],
            U=U,
            V=V,
            H=H // self.window_size[0],
            W=W // self.window_size[1],
        )
        return x

    def forward(self, x):
        x = (
            torch.roll(x, to_2tuple(self.shift_size), dims=(-1, -2))
            if self.shift_size
            else x
        )
        x = self.forward_feat(x)
        x = (
            torch.roll(x, to_2tuple(-self.shift_size), dims=(-1, -2))
            if self.shift_size
            else x
        )
        return x


# class WindowAttention(nn.Module):
#     # idx=0
#     # New version
#     def __init__(
#         self,
#         dim,
#         num_heads,
#         window="(B U V H W) (wv wh) C",
#         shift_size=0,
#         window_size=8,
#         pos=False,
#         pos_repeat=1,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim**-0.5
#         self.window = window
#         self.shift_size = shift_size
#         self.window_size = to_2tuple(window_size)
#         self.window_area = self.window_size[0] * self.window_size[1]
#         self.pos = pos
#         self.pos_repeat = pos_repeat
#         if self.pos:
#             self.relative_position_bias_table = nn.Parameter(
#                 torch.zeros(
#                     (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
#                     num_heads,
#                 )
#             )
#             self.register_buffer(
#                 "relative_position_index",
#                 get_relative_position_index(self.window_size[0], self.window_size[1]),
#             )
#             trunc_normal_(self.relative_position_bias_table, std=0.02)
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.proj = nn.Linear(dim, dim)
#         self.softmax = nn.Softmax(dim=-1)

#     def _get_rel_pos_bias(self) -> torch.Tensor:
#         relative_position_bias = self.relative_position_bias_table[
#             self.relative_position_index.view(-1)
#         ].view(
#             self.window_area, self.window_area, -1
#         )  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(
#             2, 0, 1
#         ).contiguous()  # nH, Wh*Ww, Wh*Ww
#         return relative_position_bias.unsqueeze(0)

#     def forward_feat(self, x):
#         B, C, U, V, H, W = x.shape
#         x = rearrange(
#             x,
#             f"B C U V (H wv) (W wh)-> {self.window}",
#             wv=self.window_size[0],
#             wh=self.window_size[1],
#             U=U,
#             V=V,
#             H=H // self.window_size[0],
#             W=W // self.window_size[1],
#         )
#         pos_repeat = self.pos_repeat
#         qkv = rearrange(self.qkv(x), "b s (h c)-> b h s c", h=self.num_heads)
#         # if self.window=="(B H wv W wh) (U V) C":
#         #     torch.save(
#         #         qkv.detach(), f"/opt/data/private/LFSR/T4LF/attn/qkv{WindowAttention.idx}.pt"
#         #     )
#         #     WindowAttention.idx = WindowAttention.idx + 1
#         #     print(WindowAttention.idx)
#         q, k, v = torch.chunk(qkv, 3, dim=-1)
#         # q = q * self.scale
#         # attn = q @ k.transpose(-1, -2)
#         x = nn.functional.scaled_dot_product_attention(
#                 q,
#                 k,
#                 v,
#             )
        
#         # if self.pos:
#         #     attn += self._get_rel_pos_bias().repeat(1, 1, pos_repeat, pos_repeat)

#         # attn = self.softmax(attn)
#         # x = rearrange(attn @ v, "b h s c ->b s (h c)", h=self.num_heads)
#         x = rearrange(x, "b h s c ->b s (h c)", h=self.num_heads)
#         x = self.proj(x)
#         x = rearrange(
#             x,
#             f"{self.window} -> B C U V (H wv) (W wh)",
#             wv=self.window_size[0],
#             wh=self.window_size[1],
#             U=U,
#             V=V,
#             H=H // self.window_size[0],
#             W=W // self.window_size[1],
#         )
#         return x

#     def forward(self, x):
#         x = (
#             torch.roll(x, to_2tuple(self.shift_size), dims=(-1, -2))
#             if self.shift_size
#             else x
#         )
#         x = self.forward_feat(x)
#         x = (
#             torch.roll(x, to_2tuple(-self.shift_size), dims=(-1, -2))
#             if self.shift_size
#             else x
#         )
#         return x

class SwinLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window,
        shift,
        window_size=8,
        mlp_ratio=4,
        pos=False,
        pos_repeat=1,
    ):
        super().__init__()
        self.mca = MCA(dim)
        # self.mca=SCA(dim,d='e')
        self.norm1 = LayerNorm(dim)
        self.window_attn = WindowAttention(
            dim, num_heads, window, shift, window_size, pos, pos_repeat
        )
        self.norm2 = LayerNorm(dim)
        self.mlp = MBConv(
            dim, dim, kernel_size=5, norm=None, use_bias=True, expand_ratio=mlp_ratio
        )
        # self.mlp=nn.Sequential(
        #     Conv2d(dim, 2*dim, 1, 1, 0),
        #     nn.SiLU(),
        #     Conv2d(2*dim, dim, 1, 1, 0),
        # )

    def forward(self, x):
        t = self.norm1(x)
        x = x + self.window_attn(t) + self.mca(t)
        x = self.mlp(self.norm2(x)) + x
        return x


class BasicBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        windows=[
            "(B U V H W) (wv wh) C",
            "(B H wv W wh) (U V) C",
            # "(B H wv W wh) (U V) C",
            "(B U V H W) (wv wh) C",
            # "(B H wv W wh) (U V) C",
            "(B H wv W wh) (U V) C",
        ],
        w_size=[8, 1, 8, 1],
        shift=[0, 0, 4, 0],
        pos_embed=[True, False, True, False],
        mlp_ratio=4,
    ):
        super().__init__()
        self.windows = windows
        self.shift = shift
        self.layers = nn.ModuleList(
            [
                SwinLayer(dim, num_heads, win, s, w_size, mlp_ratio, pos)
                for win, s, w_size, pos in zip(windows, shift, w_size, pos_embed)
            ]
        )
        self.conv = MBConv(dim, dim, kernel_size=3, expand_ratio=mlp_ratio)
        # self.conv = Conv2d(dim,dim,3,1,1)
        # nn.init.constant_(self.conv.weight, 0)
        # nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        shortcut = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return self.conv(x) + shortcut


@ARCH_REGISTRY.register()
class CST_(nn.Module):

    def __init__(
        self,
        num_in_ch=1,
        num_out_ch=1,
        num_feat=96,
        num_group=7,
        heads=6,
        upscale=2,
        windows=[
            "(B U V H W) (wv wh) C",
            "(B H wv W wh) (U V) C",
            # "(B H wv W wh) (U V) C",
            "(B U V H W) (wv wh) C",
            "(B H wv W wh) (U V) C",
            # "(B H wv W wh) (U V) C",
        ],
        w_size=[8, 1, 8, 1],
        shift=[0, 0, 4, 0],
        pos_embed=[True, False, True, False],
        train_tail=False,
        expand_ratio=2,
    ):
        super().__init__()
        self.upscale = upscale
        self.w_size = w_size
        self.shift = shift
        self.conv_first = nn.Sequential(
            Conv2d(num_in_ch, num_feat, 3, 1, 1),
            # MBConv(num_feat, num_feat, kernel_size=3),
        )
        self.layers = nn.ModuleList(
            [
                BasicBlock(
                    num_feat, heads, windows, w_size, shift, pos_embed, expand_ratio
                )
                for _ in range(num_group)
            ]
        )
        self.conv_after_body = Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat, True)
        self.conv_last = Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self._freeze_backbone(train_tail)

    def _freeze_backbone(self, train_tail):
        if train_tail:
            for k, v in self.named_parameters():
                if ("conv_last" in k) or ("conv_after_body" in k) or ("upsample" in k):
                    v.requires_grad = True
                else:
                    v.requires_grad = False
        else:
            for k, v in self.named_parameters():
                v.requires_grad = True

    def forward_feat(self, x):  #
        x, x_size = self._check_image_shape(x)
        for layer in self.layers:
            x = layer(x) + x
        # return x
        return x[..., : x_size[0], : x_size[1]]

    def forward(self, x):
        # bilinear = interpolate(x, scale_factor=self.upscale)
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_feat(x)) + x
        x = self.conv_last(self.upsample(x))
        return x

    def _check_image_shape(self, x):
        b, c, u, v, h, w = x.shape
        ws1, ws2 = to_2tuple(max(self.w_size))
        mod_pad_h = (ws1 - h % ws1) % ws1
        mod_pad_w = (ws2 - w % ws2) % ws2
        x = x.reshape(-1, h, w)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect").reshape(
            b, c, u, v, h + mod_pad_h, w + mod_pad_w
        )
        return x, [h, w]
