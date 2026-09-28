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
import kornia
from .base_model import BaseModel
import pyiqa
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
@MODEL_REGISTRY.register()
class DeblurModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(DeblurModel, self).__init__(opt)

        # define network
        in_channels = opt["network_g"].get("inp_channels", 3)
        self.align_list = opt["align_list"].get("names", [])
        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model_to_device(self.net_g)
        h = opt["network_g"].get("h", 256)
        self.print_network(self.net_g, (1, in_channels, h, h), get_inference_speed=opt["val"].get("speed", False))

        self.grad_clip = opt.get("grad_clip", 0)
        self.accelerator = None
        self.slides = self.opt["slide"].get("slide_datasets", [])
        # load pretrained models
        self.iqa_metrics = {
            'PSNR': pyiqa.create_metric('psnr').to(self.device),
            # 'SSIM': pyiqa.create_metric('ssim').to(self.device),
            # 'LPIPS': pyiqa.create_metric('lpips', device=self.device),
            # 'NIMA': pyiqa.create_metric('nima', device=self.device),
            # 'CLIPIQA': pyiqa.create_metric('clipiqa+_vitL14_512', device=self.device),
            # 'NIQE': pyiqa.create_metric('niqe', device=self.device),
            'MUSIQ': pyiqa.create_metric('musiq', device=self.device),
            # 'NRQM': pyiqa.create_metric('nrqm', device=self.device),
            # 'MANIQA': pyiqa.create_metric('maniqa-pipal', device=self.device)
        }

        # Initialize FID separately
        # self.fid_metric = pyiqa.create_metric('fid', device=self.device)
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

        # define losses
        if train_opt.get("pixel_opt"):
            self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pix = None
        if train_opt.get("fft_opt"):
            self.cri_fft = build_loss(train_opt["fft_opt"]).to(self.device)
        else:
            self.cri_fft = None
        if train_opt.get("ldl_opt"):
            self.cri_ldl = build_loss(train_opt["ldl_opt"]).to(self.device)
        else:
            self.cri_ldl = None

        if train_opt.get("perceptual_opt"):
            self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(
                self.device
            )
        else:
            self.cri_perceptual = None

        if (
            self.cri_pix is None
            and self.cri_perceptual is None
            and self.cri_ldl is None
        ):
            raise ValueError("Both pixel and perceptual losses are None.")

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

            if isinstance(preds, dict):
                # l_deblur = 0.
                # if 'loss_deblur' in preds.keys():
                #     # print(preds['loss'])
                #     l_deblur += preds['loss_deblur']
                #     loss_dict['l_deblur'] = l_deblur
                #     # l_deblur *= preds['deblur_loss_weight']
                #     l_total += l_deblur
                preds_img = preds['restored']
                if 'loss_model' in preds.keys():
                    # print('loss_model: ', preds['loss_model'])
                    l_model = preds['loss_model']
                    loss_dict["l_model"] = l_model
                    l_total += l_model
            else:
                preds_img = preds

            self.output = preds_img

            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(self.output, self.gt)
                l_total += l_pix
                loss_dict["l_pix"] = l_pix
            if self.cri_fft:
                l_fft = self.cri_fft(self.output, self.gt)
                l_total += l_fft
                loss_dict["l_fft"] = l_fft
            if self.cri_ldl:
                pixel_weight = get_refined_artifact_map(
                    self.gt,
                    self.output,
                    std=self.opt["train"]["ldl_std"],
                )
                l_ldl = pixel_weight * self.cri_ldl(self.output, self.gt)
                l_total += l_ldl.mean()
                loss_dict["l_ldl"] = l_ldl.mean()
            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict["l_percep"] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict["l_style"] = l_style
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
    def optimize_parameters_acc(self, current_iter):
        self.net_g.train()
        self.optimizer_g.zero_grad()
        # ================= 1. 前向传播与 Loss 计算 =================
        # 使用 autocast 上下文管理器，如果 self.use_amp 为 False，则相当于正常执行
        # print(self.use_amp)
        # with autocast(enabled=self.use_amp):
        # self.output = self.net_g(self.lq)
        preds = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        if isinstance(preds, dict):
            # l_deblur = 0.
            # if 'loss_deblur' in preds.keys():
            #     # print(preds['loss'])
            #     l_deblur += preds['loss_deblur']
            #     loss_dict['l_deblur'] = l_deblur
            #     # l_deblur *= preds['deblur_loss_weight']
            #     l_total += l_deblur
            preds_img = preds['restored']

        else:
            preds_img = preds

        self.output = preds_img

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix
        if self.cri_fft:
            l_fft = self.cri_fft(self.output, self.gt)
            l_total += l_fft
            loss_dict["l_fft"] = l_fft
        if self.cri_ldl:
            pixel_weight = get_refined_artifact_map(
                self.gt,
                self.output,
                std=self.opt["train"]["ldl_std"],
            )
            l_ldl = pixel_weight * self.cri_ldl(self.output, self.gt)
            l_total += l_ldl.mean()
            loss_dict["l_ldl"] = l_ldl.mean()
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict["l_style"] = l_style

        self.accelerator.backward(l_total)

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
                    pred = self.net_g_ema(self.lq)
                    if isinstance(pred, list):
                        pred = pred[-1]
                    elif isinstance(pred, dict):
                        pred = pred['restored']
                        if isinstance(pred, list):
                            pred = pred[-1]
                            if isinstance(pred, list):
                                pred = pred[-1]
                    self.output = pred
            else:
                self.net_g.eval()
                with torch.no_grad():
                    pred = self.net_g(self.lq)
                    if isinstance(pred, list):
                        pred = pred[-1]
                    elif isinstance(pred, dict):
                        pred = pred['restored']
                        if isinstance(pred, list):
                            pred = pred[-1]
                            if isinstance(pred, list):
                                pred = pred[-1]
                    self.output = pred
                self.net_g.train()

    def test_lq(self):
        self.output = self.lq
    def test_patch(self, patch):
        with autocast(enabled=self.val_amp):
            if hasattr(self, "net_g_ema"):
                # self.net_g_ema.eval()
                with torch.no_grad():
                    pred = self.net_g_ema(patch)
                    if isinstance(pred, list):
                        pred = pred[-1]
                    elif isinstance(pred, dict):
                        pred = pred['restored']
                        if isinstance(pred, list):
                            pred = pred[-1]
                            if isinstance(pred, list):
                                pred = pred[-1]
                    output = pred
            else:

                with torch.no_grad():
                    pred = self.net_g(patch)
                    if isinstance(pred, list):
                        pred = pred[-1]
                    elif isinstance(pred, dict):
                        pred = pred['restored']
                        if isinstance(pred, list):
                            pred = pred[-1]
                            if isinstance(pred, list):
                                pred = pred[-1]
                    output = pred
                # self.net_g.train()
            return output
    def test_slide(self):
        inp_img_ = self.grids(self.lq)
        all_batch = inp_img_.shape[0]
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
        else:
            self.net_g.eval()
        # batch_num = all_batch // batch
        if self.mini_bs is not None:
            batchs = range(0, all_batch, self.mini_bs)

            out_parts = []
            # print(inp_img_.shape)
            for batch_z in batchs:
                if batch_z + self.mini_bs <= all_batch:
                    in_x = inp_img_[batch_z:batch_z + self.mini_bs, ...]
                    if len(in_x.shape) == 3:
                        in_x = in_x.unsqueeze(0)
                    out_parts.append(self.test_patch(in_x))
                else:
                    in_x = inp_img_[batch_z:, ...]
                    if len(in_x.shape) == 3:
                        in_x = in_x.unsqueeze(0)
                    out_parts.append(self.test_patch(in_x))
            out_parts = torch.cat(out_parts, dim=0)
        else:
            out_parts = self.test_patch(inp_img_)
        # print(out_parts.shape)

        self.output = self.grids_inverse(out_parts)

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == "v":
                tfnp = v2np[..., ::-1].copy()
            elif op == "h":
                tfnp = v2np[..., ::-1, :].copy()
            elif op == "t":
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in "v", "h", "t":
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        out_list = []
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                for aug in lq_list:
                    pred = self.net_g_ema(aug)
                    if isinstance(pred, list):
                        pred = pred[-1]
                    elif isinstance(pred, dict):
                        pred = pred['restored']
                        if isinstance(pred, list):
                            pred = pred[-1]
                    out_list.append(pred)
                # out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                for aug in lq_list:
                    pred = self.net_g(aug)
                    if isinstance(pred, list):
                        pred = pred[-1]
                    elif isinstance(pred, dict):
                        pred = pred['restored']
                        if isinstance(pred, list):
                            pred = pred[-1]
                    out_list.append(pred)
                # out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], "t")
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], "h")
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], "v")
            out_list[i] = out_list[i].unsqueeze(0)
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0)

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

    def test_tile(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.lq.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.lq.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt["tile"]["infer_size"])
        tiles_y = math.ceil(height / self.opt["tile"]["infer_size"])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt["tile"]["infer_size"]
                ofs_y = y * self.opt["tile"]["infer_size"]
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt["tile"]["infer_size"], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt["tile"]["infer_size"], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt["tile"]["tile_pad"], 0)
                input_end_x_pad = min(input_end_x + self.opt["tile"]["tile_pad"], width)
                input_start_y_pad = max(input_start_y - self.opt["tile"]["tile_pad"], 0)
                input_end_y_pad = min(
                    input_end_y + self.opt["tile"]["tile_pad"], height
                )

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                input_tile = self.lq[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                # upscale tile
                output_tile = None
                try:
                    if hasattr(self, "net_g_ema"):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_tile = self.net_g(input_tile)
                        self.net_g.train()
                except RuntimeError as error:
                    raise error

                # output tile area on total image
                output_start_x = input_start_x * self.opt["scale"]
                output_end_x = input_end_x * self.opt["scale"]
                output_start_y = input_start_y * self.opt["scale"]
                output_end_y = input_end_y * self.opt["scale"]

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt[
                    "scale"
                ]
                output_end_x_tile = (
                    output_start_x_tile + input_tile_width * self.opt["scale"]
                )
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt[
                    "scale"
                ]
                output_end_y_tile = (
                    output_start_y_tile + input_tile_height * self.opt["scale"]
                )

                # put tile into output image
                self.output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]
    def grids(self, x, level=0):
        b, c, h, w = x.shape

        k1, k2 = self.ker_size[0] , self.ker_size[1]
        # print(h, w, k1, k2)
        k1 = min(h, k1)
        k2 = min(w, k2)

        overlap_size = [self.overlap_size[0], self.overlap_size[1]]  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])

        num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - overlap_size[1] - 1) // stride[1] + 1

        # import math
        step_j = k2 if num_col == 1 else stride[1]  # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0]  # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        # self.ek1, self.ek2 = None, None
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                # if not self.ek1:
                #     # print(step_i, i, k1, h)
                #     self.ek1 = i + k1 - h # - self.overlap_size[0]
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    # if not self.ek2:
                    #     self.ek2 = j + k2 - w # + self.overlap_size[1]
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i * self.up_scale, 'j': j * self.up_scale})
                j = j + step_j
            i = i + step_i

        self.num_patchs = len(parts)
        # print(len(parts))
        parts = torch.stack(parts, dim=1)
        # print(parts.shape)
        parts = parts.view(-1, c, k1, k2)
        # print(parts.shape)
        # parts = parts.view(b*self.num_patchs, -1, k1, k2)
        # if level == 0:

        self.original_size = (b, self.out_channels, h * self.up_scale , w * self.up_scale)
        self.stride = (stride[0] * self.up_scale, stride[1] * self.up_scale)
        self.nr = num_row
        self.nc = num_col
        self.idxes = idxes
        return parts

    def get_overlap_matrix(self, h, w, level=0):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        # print( self.ek1,  self.ek2)
        self.ek1 = self.nr * self.stride[0] + self.overlap_size[0] * 2 - h
        self.ek2 = self.nc * self.stride[1] + self.overlap_size[1] * 2 - w
        # print(self.ek1, self.ek2, h, w, self.nr, self.nc, self.stride, self.overlap_size, h, w)
        if self.h != h or self.w != w:

            self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size[1]).view(1, 1, self.overlap_size[1]).unsqueeze(0)
            self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size[1]).view(1, 1, self.overlap_size[1]).unsqueeze(0)
            self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size[0]).view(1, self.overlap_size[0], 1).unsqueeze(0)
            self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size[0]).view(1, self.overlap_size[0], 1).unsqueeze(0)
        if self.h != h:
            self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, self.ek1, 1).unsqueeze(0)
            self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, self.ek1, 1).unsqueeze(0)
        if self.w != w:
            self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, self.ek2).unsqueeze(0)
            self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, self.ek2).unsqueeze(0)
        self.h = h
        self.w = w

    def grids_inverse(self, outs, level=0, out_device=None, pix_max=1., size=None):
        if out_device is None:
            out_device = outs.device
        # print(out_device)
        # print(outs.shape)
        type_out = torch.uint8 if pix_max == 255. else torch.float32
        # type_out = torch.int16
        if size is None:
            preds = torch.zeros(self.original_size, device=out_device, dtype=type_out)  # .to(out_device)
            b, c, h, w = self.original_size
        else:
            preds = torch.zeros(size, device=out_device, dtype=type_out)  # .to(out_device)
            b, c, h, w = size
        # print(self.original_size)
        # outs = torch.clamp(outs, 0, 1) # * pix_max
        # outs = outs.type(type_out)
        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size_up
        k1 = min(h, k1)
        k2 = min(w, k2)
        # print(outs.shape, self.num_patchs)
        outs = outs.clone().view(b, self.num_patchs, -1, k1, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w, level)
        # rounding_mode = 'floor'
        # print(self.idxes)
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            if self.overlap_size_up[0] > 0:
                if i != 0 and i + k1 != h:
                    # print(1)
                    outs[:, cnt, :, :self.overlap_size_up[0], :] = torch.mul(outs[:, cnt, :, :self.overlap_size_up[0], :],
                                                                          self.fuse_matrix_h2.to(outs.device))
                    # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                    #       self.fuse_matrix_h1.shape)
                if i + k1 * 2 - self.ek1 < h:
                    # print(2)
                    # print(self.fuse_matrix_h1.shape, outs[:, cnt, :, -self.overlap_size_up[0]:, :].shape)
                    # print(torch.mul(outs[:, cnt, :, -self.overlap_size_up[0]:, :],
                    #                                                        self.fuse_matrix_h1.to(outs.device)).shape)
                    outs[:, cnt, :, -self.overlap_size_up[0]:, :] = torch.mul(outs[:, cnt, :, -self.overlap_size_up[0]:, :],
                                                                           self.fuse_matrix_h1.to(outs.device))
            # print(self.fuse_matrix_eh1.dtype)
            if self.ek1 > 0:
                if i + k1 == h and i !=0:
                    # print(3)
                    outs[:, cnt, :, :self.ek1, :] = torch.mul(outs[:, cnt, :, :self.ek1, :], self.fuse_matrix_eh2.to(outs.device))
                if i + k1 * 2 - self.ek1 == h:
                    # print(4)
                    outs[:, cnt, :, -self.ek1:, :] = torch.mul(outs[:, cnt, :, -self.ek1:, :],
                                                            self.fuse_matrix_eh1.to(outs.device))

            if self.overlap_size_up[1] > 0:
                if j != 0 and j + k2 != w:
                    # print(1)
                    outs[:, cnt, :, :, :self.overlap_size_up[1]] = torch.mul(outs[:, cnt, :, :, :self.overlap_size_up[1]],
                                                                          self.fuse_matrix_w2.to(outs.device))
                if j + k2 * 2 - self.ek2 < w:
                    # print(2)
                    # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                    # print(self.fuse_matrix_w1.shape)
                    outs[:, cnt, :, :, -self.overlap_size_up[1]:] = torch.mul(outs[:, cnt, :, :, -self.overlap_size_up[1]:],
                                                                           self.fuse_matrix_w1.to(outs.device))
            if self.ek2 > 0:
                if j + k2 == w and j != 0:
                    # print(3)
                    # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                    outs[:, cnt, :, :, :self.ek2] = torch.mul(outs[:, cnt, :, :, :self.ek2], self.fuse_matrix_ew2.to(outs.device))
                if j + k2 * 2 - self.ek2 == w:
                    # print(4)
                    # print('j + k2*2 - self.ek2 == w: ')
                    # print(self.ek2, self.fuse_matrix_ew1.shape,outs[:, cnt, :, :, -self.ek2:].shape )
                    outs[:, cnt, :, :, -self.ek2:] = torch.mul(outs[:, cnt, :, :, -self.ek2:],
                                                            self.fuse_matrix_ew1.to(outs.device))
            # print(i, j)
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            # pred_win = outs[cnt, :, :, :].clone() * pix_max
            # pred_win = pred_win.type(type_out)
            # print(pred_win.dtype, preds.dtype, type_out, outs.dtype)
            # print(outs[cnt, :, :, :].max())
            # print(preds[:, :, i:i + k1, j:j + k2].shape, j, outs.shape)
            # print(preds.shape, outs.shape)
            preds[:, :, i:i + k1, j:j + k2] += outs[:, cnt, :, :, :].type(type_out).to(out_device)
            # preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].to(out_device) * pix_max
            # count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds  # / count_mt
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
    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if (
            hasattr(self, "best_metric_results")
            and dataset_name in self.best_metric_results
        ):
            return
        elif not hasattr(self, "best_metric_results"):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric in self.iqa_metrics.keys():
            better = "better" if metric in ["NIQE", "LPIPS"] else "higher" # content.get("better", "higher")
            init_val = float("-inf") if better == "higher" else float("inf")
            record[metric] = dict(better=better, val=init_val, iter=-1)
            # init_val = float("-inf")
            # record[metric] = dict(better='higher', val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record
    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        if self.best_metric_results[dataset_name][metric]["better"] == "higher":
            if val >= self.best_metric_results[dataset_name][metric]["val"]:
                self.best_metric_results[dataset_name][metric]["val"] = val
                self.best_metric_results[dataset_name][metric]["iter"] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]["val"]:
                self.best_metric_results[dataset_name][metric]["val"] = val
                self.best_metric_results[dataset_name][metric]["iter"] = current_iter
    # def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
    #
    #     self.best_metric_results[dataset_name][metric]["val"] = val
    #     self.best_metric_results[dataset_name][metric]["iter"] = current_iter

    def image_align(self, deblurred, gt):
        # this function is based on kohler evaluation code
        z = deblurred
        # c = np.ones_like(z)
        x = gt

        zs = (np.sum(x * z) / np.sum(z * z)) * z  # simple intensity matching

        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.eye(3, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = 100

        termination_eps = 0

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    number_of_iterations, termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY),
                                                 cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria,
                                                 inputMask=None, gaussFiltSize=5)

        target_shape = x.shape
        shift = warp_matrix

        zr = cv2.warpPerspective(
            zs,
            warp_matrix,
            (target_shape[1], target_shape[0]),
            flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT)

        cr = cv2.warpPerspective(
            np.ones_like(zs, dtype='float32'),
            warp_matrix,
            (target_shape[1], target_shape[0]),
            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0)

        zr = zr * cr
        xr = x * cr

        return zr, xr, cr, shift
    def nondist_validation(
        self, dataloader, current_iter, tb_logger, save_img, clamp=True
    ):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        use_pbar = self.opt["val"].get("pbar", False)

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                # self.metric_results = {
                #     metric: 0 for metric in self.opt["val"]["metrics"].keys()
                # }
                self.metric_results = {
                    metric: 0 for metric in self.iqa_metrics.keys()
                }
                # self.metric_results['fid'] = 0
                # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")
        max_idx = self.opt["val"].get("max_idx", 1e9) # 1200
        # max_idx = 2
        idx_ = 0
        for idx, val_data in enumerate(dataloader):
            if idx_ > max_idx:
                break
            self.feed_data(val_data)

            self.pre_test()
            if "tile" in self.opt:
                self.test_tile()
            elif "ensemble" in self.opt and self.opt["ensemble"]:
                self.test_selfensemble()
            elif "slide" in self.opt and self.opt["slide"] and dataset_name in self.slides:
                self.test_slide()
            else:
                self.test()
                # self.test_lq()
            self.post_test()

            visuals = self.get_current_visuals()
            if clamp:
                visuals["result"] = visuals["result"].clamp(0, 1)
                visuals["gt"] = visuals["gt"].clamp(0, 1)
            idx_ += visuals["result"].shape[0]
            # print(visuals["result"].shape[0])


            del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gt = visuals["gt"] # .clone()
            result = visuals["result"] # .clone()
            if dataset_name in self.align_list:
                device = result.device
                result = result.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                gt = gt.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                result, gt, cr1, shift = self.image_align(result, gt)
                result = torch.from_numpy(result).permute(2, 0, 1).unsqueeze(0).to(device)
                gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device)
            # gt = kornia.color.rgb_to_bgr(visuals["gt"])
            # result = kornia.color.rgb_to_bgr(visuals["result"])
            if with_metrics:
                # calculate metrics
                for name, metric in self.iqa_metrics.items():
                # for name, opt_ in self.opt["val"]["metrics"].items():
                    with torch.no_grad():
                        if name in ['NIMA', 'CLIPIQA', 'NIQE', 'MUSIQ', 'MANIQA', 'PAQ2PIQ']:
                            self.metric_results[name] += metric(visuals["result"]).item()
                        else:
                            self.metric_results[name] += metric(result, gt).item()
                        # print(name, metric(result, gt).item())
                    # self.metric_results[name] += calculate_metric(
                    #     {
                    #         "img": visuals["result"],
                    #         "img2": visuals["gt"],
                    #     },
                    #     opt_,
                    # )
                    # print(calculate_metric(
                    #     {
                    #         "img": visuals["result"],
                    #         "img2": visuals["gt"],
                    #     },
                    #     opt_,
                    # ))

                if not clamp:
                    try:
                        assert np.isnan(visuals["result"]).sum() == 0
                    except:
                        visuals["result"][np.isnan(visuals["result"])] = 0

                    self.metric_results["mae"] = 255 * np.abs(
                        np.mean(visuals["result"].clip(0, 1) - visuals["gt"])
                    )
            visuals["result"] = visuals["result"].numpy()
            visuals["gt"] = visuals["gt"].numpy()
            if save_img:
                for i, img_path in enumerate(val_data["lq_path"]):
                    depth = self.opt["depth"] if "depth" in self.opt else 8
                    if depth == 16:
                        sr_img = (
                            (visuals["result"][i, ...] * 65535.0)
                            .round()
                            .astype(np.uint16)
                        )
                    else:
                        sr_img = (
                            (visuals["result"][i, ...] * 255.0).round().astype(np.uint8)
                        )
                    if sr_img.shape[0] == 3:
                        sr_img = cv2.cvtColor(
                            sr_img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR
                        )
                    if sr_img.shape[-1] == 1:
                        sr_img = sr_img[..., 0]
                    img_name = osp.splitext(osp.basename(img_path))[0]
                    if self.opt["is_train"]:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            img_name,
                            f"{img_name}_{current_iter}.png",
                        )
                    else:
                        if self.opt["val"]["suffix"]:
                            save_img_path = osp.join(
                                self.opt["path"]["visualization"],
                                dataset_name,
                                f'{img_name}_{self.opt["val"]["suffix"]}.png',
                            )
                        else:
                            save_img_path = osp.join(
                                self.opt["path"]["visualization"],
                                dataset_name,
                                f'{img_name}.png',
                            )
                            # save_img_path = osp.join(
                            #     self.opt["path"]["visualization"],
                            #     dataset_name,
                            #     f'{img_name}_{self.opt["name"]}.png',
                            # )
                    imwrite(sr_img, save_img_path, depth=depth)

            if use_pbar:
                pbar.update(1)
                # pbar.set_description(f"Test {img_name}")
        if use_pbar:
            pbar.close()
        print('num_images: ', idx_)
        if with_metrics:
            # idx = min(idx, max_idx)
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= idx_ # idx + 1
                # update the best metric result
                if clamp:
                    self._update_best_metric_result(
                        dataset_name, metric, self.metric_results[metric], current_iter
                    )
            if clamp:
                self._log_validation_metric_values(
                    current_iter, dataset_name, tb_logger
                )
    def validation_acc(
        self, dataloader, current_iter, tb_logger, save_img, clamp=True
    ):
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

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")
        max_idx = self.opt["val"].get("max_idx", 1e9) # 1200
        idx_ = 0
        for idx, val_data in enumerate(dataloader):
            if idx_ > max_idx:
                break
            self.feed_data(val_data)

            self.pre_test()
            if "tile" in self.opt:
                self.test_tile()
            elif "ensemble" in self.opt and self.opt["ensemble"]:
                self.test_selfensemble()
            elif "slide" in self.opt and self.opt["slide"]:
                self.test_slide()
            else:
                self.test()
            self.post_test()

            visuals = self.get_current_visuals_acc()

            if clamp:
                visuals["result"] = visuals["result"].clamp(0, 1)
                visuals["gt"] = visuals["gt"].clamp(0, 1)
            idx_ += visuals["result"].shape[0]
            # print(visuals["result"].shape[0])
            visuals["result"] = visuals["result"].numpy()
            visuals["gt"] = visuals["gt"].numpy()

            del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            if with_metrics and self.accelerator.is_main_process:
                # calculate metrics
                for name, opt_ in self.opt["val"]["metrics"].items():
                    self.metric_results[name] += calculate_metric(
                        {
                            "img": visuals["result"],
                            "img2": visuals["gt"],
                        },
                        opt_,
                    )
                if not clamp:
                    try:
                        assert np.isnan(visuals["result"]).sum() == 0
                    except:
                        visuals["result"][np.isnan(visuals["result"])] = 0

                    self.metric_results["mae"] = 255 * np.abs(
                        np.mean(visuals["result"].clip(0, 1) - visuals["gt"])
                    )

            if save_img and self.accelerator.is_main_process:
                for i, img_path in enumerate(val_data["lq_path"]):
                    depth = self.opt["depth"] if "depth" in self.opt else 8
                    if depth == 16:
                        sr_img = (
                            (visuals["result"][i, ...] * 65535.0)
                            .round()
                            .astype(np.uint16)
                        )
                    else:
                        sr_img = (
                            (visuals["result"][i, ...] * 255.0).round().astype(np.uint8)
                        )
                    if sr_img.shape[0] == 3:
                        sr_img = cv2.cvtColor(
                            sr_img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR
                        )
                    if sr_img.shape[-1] == 1:
                        sr_img = sr_img[..., 0]
                    img_name = osp.splitext(osp.basename(img_path))[0]
                    if self.opt["is_train"]:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            img_name,
                            f"{img_name}_{current_iter}.png",
                        )
                    else:
                        if self.opt["val"]["suffix"]:
                            save_img_path = osp.join(
                                self.opt["path"]["visualization"],
                                dataset_name,
                                f'{img_name}_{self.opt["val"]["suffix"]}.png',
                            )
                        else:
                            save_img_path = osp.join(
                                self.opt["path"]["visualization"],
                                dataset_name,
                                f'{img_name}_{self.opt["name"]}.png',
                            )
                    imwrite(sr_img, save_img_path, depth=depth)

            if use_pbar:
                pbar.update(1)
                # pbar.set_description(f"Test {img_name}")
        if use_pbar:
            pbar.close()

        if with_metrics and self.accelerator.is_main_process:
            # idx = min(idx, max_idx)
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= idx_ # (idx_ + 1) # idx + 1
                # update the best metric result
                if clamp:
                    self._update_best_metric_result(
                        dataset_name, metric, self.metric_results[metric], current_iter
                    )
            if clamp:
                self._log_validation_metric_values(
                    current_iter, dataset_name, tb_logger
                )

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += (
                    f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                    f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                )
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
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.float().detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
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
