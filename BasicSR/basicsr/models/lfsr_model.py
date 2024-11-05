from collections import OrderedDict
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from basicsr.metrics import calculate_metric
from os import path as osp
from tqdm import tqdm
import torch
import os
from einops import rearrange
import numpy as np
import torchvision
from torch.cuda.amp import autocast, GradScaler
from basicsr.utils import get_root_logger
from apex import amp
import imageio
def tensor2img(x):
    if isinstance(x, list):
        x = torch.concat(x, dim=0)
    x = rearrange(x, "b c u v h w -> (b u v) c h w")
    x = x.clamp_(0, 1).detach().cpu().numpy()
    return (x * 255).clip(16, 235)


mat = (
    np.array(
        [
            [65.481, 128.553, 24.966],
            [-37.797, -74.203, 112.0],
            [112.0, -93.786, -18.214],
        ],
    )
    / 255
)
mat_inv = np.linalg.inv(mat).transpose(1, 0)
mat_inv = torch.tensor(mat_inv).float()
bias = torch.tensor([-16.0 / 255.0, -128.0 / 255.0, -128.0 / 255.0]).view(1, 3, 1, 1)


def ycrcb2rgb(x):
    # x=x.clamp(16/255,235/255)
    x = x + bias
    return torch.einsum("bihw, ij->bjhw", x, mat_inv).permute(0, 1, 3, 2)


@MODEL_REGISTRY.register()
class LFSRModel(SRModel):
    def __init__(self, opt):
        super(LFSRModel, self).__init__(opt)
        self.split_test = opt["val"].get("split_test", None)
        self.TTA = opt["val"].get("TTA", False)
        self.cumulation = (
            opt["train"].get("cumulation", 1) if opt.get("train", False) else 1
        )
        self.log_dict = OrderedDict()
        self.metric_DF={}

    def optimize_parameters(self, current_iter):

        self.output = self.net_g(self.lq)
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict["l_style"] = l_style

        l_total.backward(retain_graph=(self.cumulation > 1))

        if current_iter % self.cumulation == 0:
            self.optimizer_g.step()
            self.optimizer_g.zero_grad()

        self.update_log(self.reduce_loss_dict(loss_dict))

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def update_log(self, loss_dict):
        for k, v in loss_dict.items():
            if k in self.log_dict:
                self.log_dict[k] = 0.99 * self.log_dict[k] + 0.01 * v
            else:
                self.log_dict[k] = v
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt["rank"] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        use_pbar = self.opt["val"].get("pbar", False)

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                self.metric_results = {
                    metric: 0 for metric in self.opt["val"]["metrics"].keys()
                }
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            metric_data["img"] = tensor2img([visuals["result"]])

            if save_img:
                # print(visuals["result"].detach().cpu().shape)
                sr_img = torch.concat(
                    [visuals["result"].detach().cpu(), val_data["cbcr"]], dim=1
                )
                b, c, u, v, h, w = sr_img.shape
                sr_img = rearrange(sr_img, "b c u v h w -> b c (u h) (v w)")

                os.makedirs(
                    osp.join(
                        self.opt["path"]["visualization"],
                        dataset_name,
                        img_name,
                    ),
                    exist_ok=True,
                )
                # np.save(save_img_path.replace("bmp", "npy"), sr_img[0, 0])
                sr_imgs = rearrange(
                    ycrcb2rgb(sr_img), "b c (u h) (v w) -> b u v h w c", u=u, v=v
                )
                for i in range(u):
                    for j in range(v):
                        imageio.imwrite(
                            osp.join(
                                self.opt["path"]["visualization"],
                                dataset_name,
                                img_name,
                                f"View_{i}_{j}.bmp",
                            ),
                            (sr_imgs[0, i, j].numpy().clip(0, 1) * 255).astype("uint8"),
                        )

            if "gt" in visuals:
                metric_data["img2"] = tensor2img([visuals["gt"]])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt["val"]["metrics"].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= idx + 1
                # update the best metric result
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter
                )
                if metric in self.metric_DF.keys():
                    self.metric_DF[metric][dataset_name] = self.metric_results[metric]
                else:
                    self.metric_DF[metric] = {dataset_name: self.metric_results[metric]}
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def test(self):
        if self.split_test:
            self.test_split()
        else:
            if self.TTA:
                self.test_TTA()
            else:
                super().test()

    def test_split(self):
        # todo: add split test
        scale = self.opt["scale"]
        output = torch.zeros_like(self.lq).repeat(1, 1, 1, 1, scale, scale)
        count = output.clone()
        H, W = self.lq.shape[-2:]
        model = self.net_g_ema if hasattr(self, "net_g_ema") else self.net_g
        model.eval()
        with torch.no_grad():
            for i in range(0, H, self.split_test):
                for j in range(0, W, self.split_test):
                    rx = min(i + self.split_test, H)
                    lx = max(0,rx-self.split_test)
                    ry = min(j + self.split_test, W)
                    ly = max(0,ry-self.split_test)
                    lq = self.lq[
                        ...,
                        lx : rx,
                        ly : ry,
                    ]
                    count[
                        ...,
                        scale * lx : scale * rx,
                        scale * ly : scale * ry,
                    ] += 1
                    output[
                        ...,
                        scale * lx : scale * rx,
                        scale * ly : scale * ry,
                    ] += model(lq)

        model.train()
        self.output = output / count

    def test_TTA(self):
        # todo: add split test
        scale = self.opt["scale"]
        output = torch.zeros_like(self.lq).repeat(1, 1, 1, 1, scale, scale)
        # count = output.clone()
        H, W = self.lq.shape[-2:]
        model = self.net_g_ema if hasattr(self, "net_g_ema") else self.net_g
        model.eval()
        with torch.no_grad():
            lq = self.lq
            output += model(lq)
            lq = self.lq.flip([2, 4])
            output += model(lq).flip([2, 4])
            lq = self.lq.flip([3, 5])
            output += model(lq).flip([3, 5])
            lq = self.lq.flip([2, 3, 4, 5])
            output += model(lq).flip([2, 3, 4, 5])

            lq = self.lq.permute(0, 1, 3, 2, 5, 4)
            output += model(lq).permute(0, 1, 3, 2, 5, 4)
            lq = self.lq.flip([2, 4]).permute(0, 1, 3, 2, 5, 4)
            output += model(lq).permute(0, 1, 3, 2, 5, 4).flip([2, 4])
            lq = self.lq.flip([3, 5]).permute(0, 1, 3, 2, 5, 4)
            output += model(lq).permute(0, 1, 3, 2, 5, 4).flip([3, 5])
            lq = self.lq.flip([2, 3, 4, 5]).permute(0, 1, 3, 2, 5, 4)
            output += model(lq).permute(0, 1, 3, 2, 5, 4).flip([2, 3, 4, 5])
        model.train()
        self.output = output / 8

    def test_overlap(self):
        scale = self.opt["scale"]

        # padding
        self.lq= rearrange(self.lq,'b c u v h w-> (b u v) c h w')
        self.lq = torch.nn.functional.pad(
            self.lq,
            (
                self.split_test // 4,
                self.split_test // 4,
                self.split_test // 4,
                self.split_test // 4,
            ),
            mode="replicate",
        )
        self.lq= rearrange(self.lq,'(b u v) c h w-> b c u v h w',u=5,v=5)

        output = torch.zeros_like(self.lq).repeat(1, 1, 1, 1, scale, scale)
        count = output.clone()
        H, W = self.lq.shape[-2:]
        model = self.net_g_ema if hasattr(self, "net_g_ema") else self.net_g
        model.eval()
        with torch.no_grad():
            for i in range(0, H-self.split_test//2, self.split_test//2):
                for j in range(0, W - self.split_test // 2, self.split_test // 2):
                    lq = self.lq[
                        ...,
                        i : min(i + self.split_test, H),
                        j : min(j + self.split_test, W),
                    ]
                    count[
                        ...,
                        scale * i : scale * min(i + self.split_test, H),
                        scale * j : scale * min(j + self.split_test, W),
                    ] += 1
                    output[
                        ...,
                        scale * i : scale * min(i + self.split_test, H),
                        scale * j : scale * min(j + self.split_test, W),
                    ] += model(lq)

        model.train()
        self.output = (output / count)[
            ...,
            scale * self.split_test // 4 : -scale * self.split_test // 4,
            scale * self.split_test // 4 : -scale * self.split_test // 4,
        ]
