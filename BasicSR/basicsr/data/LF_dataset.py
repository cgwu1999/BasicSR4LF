from torch.utils import data as data
from basicsr.utils.registry import DATASET_REGISTRY
from glob import glob
import os
import h5py
import numpy as np
import random
import torch
from timm.layers import to_2tuple
from einops import rearrange
import os


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[..., ::-1]
        label = label[..., ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1, :]
        label = label[:, ::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(0, 2, 1)
        label = label.transpose(0, 2, 1)
    return data, label


class ToTensor:
    def __init__(self, angRes=5):
        self.angRes = to_2tuple(angRes)

    def __call__(self, data):
        data = torch.tensor(data)
        return rearrange(
            data, "c (u h) (v w)-> c u v h w", u=self.angRes[0], v=self.angRes[1]
        )


# class Shear:
#     def __init__(self, shear=0):
#         self.shear = shear

#     def __call__(self, x, scale=4):
#         c, u, v, h, w = x.shape
#         new = torch.zeros((c, u, v, h, w - self.shear * (v - 1) * scale))
#         for i in range(u):
#             for j in range(v):
#                 new[:, i, j] = x[
#                     :,
#                     i,
#                     j,
#                     :,
#                     j * self.shear * scale : (j - v + 1) * self.shear * scale + w,
#                 ]
#         return new


class Sampler:
    def __init__(self, angRes=5, patchsize=32, scale_factor=4):
        self.angRes = to_2tuple(angRes)
        self.patchsize = to_2tuple(patchsize)
        self.scale_factor = scale_factor

    def __call__(self, lq, gt):
        C, U, V, H, W = lq.shape
        u = random.randint(0, U - self.angRes[0])
        v = random.randint(0, V - self.angRes[1])
        h = random.randint(0, H - self.patchsize[0])
        w = random.randint(0, W - self.patchsize[1])
        return (
            lq[
                :,
                u : u + self.angRes[0],
                v : v + self.angRes[1],
                h : h + self.patchsize[0],
                w : w + self.patchsize[1],
            ],
            gt[
                :,
                u : u + self.angRes[0],
                v : v + self.angRes[1],
                self.scale_factor * h : self.scale_factor * (h + self.patchsize[0]),
                self.scale_factor * w : self.scale_factor * (w + self.patchsize[1]),
            ],
        )


@DATASET_REGISTRY.register()
class H5LFDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.train = opt["phase"] == "train"
        self.data_folder = opt["dataroot"]
        self.dataset_names = opt["dataset_name"]
        self.paths = []
        for self.dataset_name in self.dataset_names:
            self.paths.extend(
                glob(os.path.join(self.data_folder, self.dataset_name, "*.h5"))
            )
        # print(self.paths)
        self.totensor = ToTensor(opt.get("angRes_ori", 9))
        if self.train:
            self.sampler = Sampler(opt["angRes"], opt["patch_size"], opt["scale_factor"])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.train:
            with h5py.File(self.paths[index], "r") as hf:
                Lr_SAI_y = np.expand_dims(np.array(hf.get("Lr_SAI_y")), axis=0)
                Hr_SAI_y = np.expand_dims(np.array(hf.get("Hr_SAI_y")), axis=0)
                Lr_SAI_y, Hr_SAI_y = augmentation(Lr_SAI_y, Hr_SAI_y)
                lq = self.totensor(Lr_SAI_y.copy())
                gt = self.totensor(Hr_SAI_y.copy())
                lq, gt = self.sampler(lq, gt)
                return {
                    "lq": lq,
                    "gt": gt,
                    "lq_path": self.paths[index],
                    "gt_path": self.paths[index],
                }
        else:
            with h5py.File(self.paths[index], "r") as hf:
                Lr_SAI_y = np.expand_dims(np.array(hf.get("Lr_SAI_y")), axis=0)
                Hr_SAI_y = np.expand_dims(np.array(hf.get("Hr_SAI_y")), axis=0)
                Sr_SAI_cbcr = np.array(hf.get("Sr_SAI_cbcr"), dtype="single")
                # Lr_SAI_y, Hr_SAI_y = augmentation(Lr_SAI_y, Hr_SAI_y)
                lq = self.totensor(Lr_SAI_y.copy())
                gt = self.totensor(Hr_SAI_y.copy())
                cbcr=self.totensor(Sr_SAI_cbcr.copy())
                # lq, gt = self.sampler(lq, gt)
                return {
                    "lq": lq,
                    "gt": gt,
                    "lq_path": self.paths[index],
                    "gt_path": self.paths[index],
                    "cbcr": cbcr
                }

@DATASET_REGISTRY.register()
class NpyLFDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.train = opt["phase"] == "train"
        self.data_folder = opt["dataroot"]
        self.dataset_names = opt["dataset_name"]
        self.paths = []
        for self.dataset_name in self.dataset_names:
            self.paths.extend(
                glob(os.path.join(self.data_folder, self.dataset_name, "*"))
            )
        self.totensor = ToTensor(opt["angRes"])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        dir = self.paths[index]

        Lr_SAI_y = np.expand_dims(np.load(os.path.join(dir, "lr_y.npy")), axis=0)
        Hr_SAI_y = np.expand_dims(np.load(os.path.join(dir, "hr_y.npy")), axis=0)
        Sr_SAI_cbcr = np.load(os.path.join(dir, "hr_cbcr.npy"))
        return {
            "lq": self.totensor(Lr_SAI_y.copy()),
            "gt": self.totensor(Hr_SAI_y.copy()),
            "lq_path": self.paths[index],
            "gt_path": self.paths[index],
            "cbcr": self.totensor(Sr_SAI_cbcr.copy()),
        }


# @DATASET_REGISTRY.register()
# class ShearNpyLFDataset(data.Dataset):
#     def __init__(self, opt):
#         self.opt = opt
#         self.train = opt["phase"] == "train"
#         self.data_folder = opt["dataroot"]
#         self.dataset_names = opt["dataset_name"]
#         self.paths = []
#         for self.dataset_name in self.dataset_names:
#             self.paths.extend(
#                 glob(os.path.join(self.data_folder, self.dataset_name, "*"))
#             )
#         self.totensor = ToTensor(opt["angRes"])
#         self.shear = Shear(opt["shear"])

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, index):
#         dir = self.paths[index]

#         Lr_SAI_y = np.expand_dims(np.load(os.path.join(dir, "lr_y.npy")), axis=0)
#         Hr_SAI_y = np.expand_dims(np.load(os.path.join(dir, "hr_y.npy")), axis=0)
#         Sr_SAI_cbcr = np.load(os.path.join(dir, "hr_cbcr.npy"))
#         lq = self.totensor(Lr_SAI_y.copy())
#         gt = self.totensor(Hr_SAI_y.copy())
#         cbcr = self.totensor(Sr_SAI_cbcr.copy())
#         lq = self.shear(lq, scale=1)
#         gt = self.shear(gt, scale=4)
#         cbcr = self.shear(cbcr, scale=4)
#         return {
#             "lq": lq,
#             "gt": gt,
#             "lq_path": self.paths[index],
#             "gt_path": self.paths[index],
#             "cbcr": cbcr,
#         }
