import math
from collections import OrderedDict
from os import path as osp

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler # 新增这一行
from torch.nn import functional as F
from tqdm import tqdm
import time
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.loss_util import get_refined_artifact_map
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.summary_utils import (
    get_model_activation,
    get_model_complexity_info,
    get_model_flops,
)

from .base_model import BaseModel

@MODEL_REGISTRY.register()
class PCModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(PCModel, self).__init__(opt)

        # define network
        in_channels = opt["network_g"].get("inp_channels", 3)
        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model_to_device(self.net_g)
        h = opt["network_g"].get("h", 256)
        self.print_network(self.net_g, (1, in_channels, h, h), get_inference_speed=opt["val"].get("speed", False))

        self.grad_clip = opt.get("grad_clip", 0)
        self.accelerator = None
        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            param_key = self.opt["path"].get("param_key_g", "params")
            self.load_network(
                self.net_g,
                load_path,
                self.opt["path"].get("strict_load_g", True),
                param_key,
                self.opt.get("remove_norm", False),
            )

        if self.is_train:
            self.init_training_settings()
            # ================= 新增 AMP 初始化 =================
            # 从配置文件读取是否开启 AMP，默认设为 False
            self.use_amp = self.opt['train'].get('use_amp', False)
            if self.use_amp:
                # 可以根据需要调整 init_scale，如果容易 NaN 可以改成 1024 或更小
                self.scaler = GradScaler(init_scale=65536.0)
            # ===================================================
        self.val_amp = self.opt['val'].get('use_amp', False)
        if "slide" in self.opt:
            self.overlap_size = self.opt["slide"].get("overlap_size", 48)
            self.mini_bs = self.opt["slide"].get("mini_bs", None)
            self.ker_size = self.opt["slide"].get("ker_size", 256)  # [int(train_size[0] * 1.), int(train_size[0] * 1.)]
            self.up_scale = self.opt.get("scale", 1)
            self.overlap_size_up = (self.overlap_size[0] * self.up_scale, self.overlap_size[1] * self.up_scale)
            # self.kernel_size = [train_size, train_size]
            self.kernel_size_up = [self.ker_size[0] * self.up_scale, self.ker_size[1] * self.up_scale]
            self.out_channels =  opt["network_g"].get("out_channels", 3)
            self.h = 0
            self.w = 0
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            # load pretrained model
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(
                    self.net_g_ema,
                    load_path,
                    self.opt["path"].get("strict_load_g", True),
                    "params_ema",
                )
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()


        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f"Params {k} will not be optimized.")

        optim_type = train_opt["optim_g"].pop("type")
        self.optimizer_g = self.get_optimizer(
            optim_type, optim_params, **train_opt["optim_g"]
        )
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device, non_blocking=True)
        if "gt" in data:
            self.gt = data["gt"].to(self.device, non_blocking=True)

    def optimize_parameters(self, current_iter):
        self.net_g.train()
        self.optimizer_g.zero_grad()
        # ================= 1. 前向传播与 Loss 计算 =================
        # 使用 autocast 上下文管理器，如果 self.use_amp 为 False，则相当于正常执行
        # print(self.use_amp)
        with autocast(enabled=self.use_amp):
            # self.output = self.net_g(self.lq)
            preds = self.net_g(self.lq, self.gt)

            l_total = 0
            loss_dict = OrderedDict()
            l_model = preds['loss_model']
            loss_dict["l_model"] = l_model
            l_total += l_model

        # ================= 2. 反向传播与参数更新 =================
        # print(loss_dict)
        # l_total += l_model
        if self.use_amp:
            # AMP 模式下的反向传播
            self.scaler.scale(l_total).backward()

            # (可选) 如果你需要梯度裁剪，取消下面两行的注释：
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            l_total.backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip)

            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        if current_iter == 5:
            # self._get_gpu_memory()
            torch.cuda.reset_peak_memory_stats(0)
        if current_iter == 10:
            self._get_gpu_memory()
            torch.cuda.reset_peak_memory_stats(0)
        if current_iter == 100:
            self.time_start = time.time()
        if current_iter == 1100:
            self.time_end = time.time()
            self._get_train_speed()

    def _get_train_speed(self):
        logger = get_root_logger()

        time_100iter = self.time_end - self.time_start
        time_100iter = round(time_100iter, 3)
        # print(mem_mb)
        logger.info(f'Model train speed: {time_100iter} s')
    def _get_gpu_memory(self):
        logger = get_root_logger()

        mem = torch.cuda.max_memory_allocated(0)
        mem_mb = int(mem) // (1024 * 1024)
        # print(mem_mb)
        logger.info(f'Model gpu memory: {mem_mb} MB')
    def test(self):
        with autocast(enabled=self.val_amp):
            if hasattr(self, "net_g_ema"):
                self.net_g_ema.eval()
                with torch.no_grad():
                    pred = self.net_g_ema(self.lq, self.gt)

                    self.output = pred['acc']
            else:
                self.net_g.eval()
                with torch.no_grad():
                    pred = self.net_g(self.lq, self.gt)

                    self.output = pred['acc']
                self.net_g.train()

    def check_window_size(self, window_size_stats):
        window_size, stats = window_size_stats
        if not (
            isinstance(window_size, tuple)
            or isinstance(window_size, list)
            and not stats
        ):
            return [window_size, True]
        return self.check_window_size([max(window_size), False])

    def pre_test(self):
        # pad to multiplication of window_size
        _, _, h, w = self.lq.size()
        if "window_size" not in self.opt["network_g"]:
            return

        # FIXME: this is only supported when the shape of lq's H == W
        window_size, _ = self.check_window_size(
            [self.opt["network_g"].get("window_size", h), False]
        )
        self.scale = self.opt.get("scale", 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        self.lq = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect")

    def post_test(self):
        _, _, h, w = self.output.size()
        if "window_size" not in self.opt["network_g"]:
            return
        self.output = self.output[
            :,
            :,
            0 : h - self.mod_pad_h * self.scale,
            0 : w - self.mod_pad_w * self.scale,
        ]


    def dist_validation(
        self, dataloader, current_iter, tb_logger, save_img, clamp=True
    ):
        if self.opt["rank"] == 0:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, clamp
            )

    def dist_profile(self, dataloader):
        if self.opt["rank"] == 0:
            self.nondist_profile(dataloader)

    def nondist_validation(
        self, dataloader, current_iter, tb_logger, save_img, clamp=True
    ):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        use_pbar = self.opt["val"].get("pbar", False)

        # if with_metrics:
        #     if not hasattr(self, "metric_results"):  # only execute in the first run
        #         self.metric_results = {
        #             metric: 0 for metric in ['Acc']
        #         }
        #     # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        #     self._initialize_best_metric_results(dataset_name)
        self.metric_results = {
            metric: 0 for metric in ['acc']
        }
        # zero self.metric_results
        # if with_metrics:
        self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")
        max_idx = self.opt["val"].get("max_idx", 1e9) # 1200
        idx_ = 0
        for idx, val_data in enumerate(dataloader):
            if idx_ > max_idx:
                break
            self.feed_data(val_data)

            self.pre_test()

            self.test()
            # self.post_test()

            visuals = self.get_current_visuals()

            idx_ += visuals["result"].shape[0]
            # print(visuals["result"].shape[0])
            # visuals["result"] = visuals["result"].numpy()
            # visuals["gt"] = visuals["gt"].numpy()

            del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            # if with_metrics:
                # calculate metrics
            self.metric_results['acc'] += visuals["result"].sum()
            # print(dataset_name, self.metric_results['acc'])

            if use_pbar:
                pbar.update(1)
                # pbar.set_description(f"Test {img_name}")
        if use_pbar:
            pbar.close()

        # if with_metrics:

            # idx = min(idx, max_idx)
        # print(self.metric_results)
        log_str = f"Validation {dataset_name}\n"
        for metric in self.metric_results.keys():
            self.metric_results[metric] /= idx_ # idx + 1
            # print(self.metric_results)
            # update the best metric result

        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            log_str += "\n"
        logger = get_root_logger()
        logger.info(log_str)
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f"metrics/{dataset_name}/{metric}", value, current_iter
                )

    def nondist_profile(self, dataloader, flops=False):
        logger = get_root_logger()

        if flops:
            H, W = 1280 // self.scale, 720 // self.scale
            # warmup
            try:
                logger.info(
                    get_model_complexity_info(
                        self.net_g, (3, H, W), print_per_layer_stat=False
                    )
                )
                logger.info(get_model_activation(self.net_g, (3, H, W)))
                logger.info(
                    get_model_flops(self.net_g, (3, H, W), print_per_layer_stat=False)
                )
            except:
                logger.warning("OOM when testing on (1280, 720).")

        # test time
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        timings = np.zeros((len(dataloader), 1))
        memorys = np.zeros((len(dataloader), 1))
        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)

            self.pre_test()
            torch.cuda.reset_peak_memory_stats()
            starter.record()
            if "tile" in self.opt:
                self.test_tile()
            elif "ensemble" in self.opt and self.opt["ensemble"]:
                self.test_selfensemble()
            else:
                self.test()
            ender.record()
            torch.cuda.synchronize()
            memorys[idx] = (
                torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024**2
            )
            self.post_test()

            curr_time = starter.elapsed_time(ender)
            timings[idx] = curr_time

        logger.info(f"The average test time is {timings.mean()} ms.")
        logger.info(f"The max allocated mem is {memorys.mean()} M.")

    def get_current_visuals(self):
        out_dict = OrderedDict()

        out_dict["result"] = self.output.float().detach().cpu()

        return out_dict

    def get_current_visuals_acc(self):
        out_dict = OrderedDict()

        # out_dict["lq"] = self.lq.detach().cpu()
        # out_dict["result"] = self.output.float().detach().cpu()
        if hasattr(self, "gt"):
            out_dict["lq"], out_dict["result"], out_dict["gt"] = self.accelerator.gather_for_metrics(
                (self.lq, self.output, self.gt))
            # out_dict["gt"] = self.gt.detach().cpu()
            out_dict["lq"], out_dict["result"], out_dict["gt"] = out_dict["lq"].detach().cpu(), out_dict["result"].detach().cpu(), out_dict["gt"].detach().cpu()
        else:
            out_dict["lq"], out_dict["result"] = self.accelerator.gather_for_metrics(
                (self.lq, self.output))
            out_dict["lq"], out_dict["result"] = out_dict["lq"].detach().cpu(), out_dict["result"].detach().cpu()
        return out_dict
    def save(self, epoch, current_iter):
        if hasattr(self, "net_g_ema"):
            self.save_network(
                [self.net_g, self.net_g_ema],
                "net_g",
                current_iter,
                param_key=["params", "params_ema"],
            )
        else:
            self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)
