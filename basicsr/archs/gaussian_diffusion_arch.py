
# from typing import Tuple
from typing import Optional, Tuple, Dict
import torch

from tqdm import tqdm
from basicsr.utils.registry import ARCH_REGISTRY
from .cldm_arch import ControlLDM
from basicsr.archs.DeblurDiff.cond_fn import Guidance, MSEGuidance, WeightedMSEGuidance
from basicsr.archs.DeblurDiff.common import sliding_windows, gaussian_weights, wavelet_decomposition, wavelet_reconstruction, count_vram_usage
from torch import nn
from torch.nn import functional as F
import numpy as np
from PIL import Image

def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def bicubic_resize(img, scale: float) -> np.ndarray:
    # pil = Image.fromarray(img)
    # res = pil.resize(tuple(int(x * scale) for x in pil.size), Image.BICUBIC)
    res = F.interpolate(img, scale_factor=scale, mode='bicubic')
    return res


def resize_short_edge_to(imgs: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h == w:
        new_h, new_w = size, size
    elif h < w:
        new_h, new_w = size, int(w * (size / h))
    else:
        new_h, new_w = int(h * (size / w)), size
    return F.interpolate(imgs, size=(new_h, new_w), mode="bicubic", antialias=True)


def pad_to_multiples_of(imgs: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h % multiple == 0 and w % multiple == 0:
        return imgs.clone()
    # get_pad = lambda x: (x // multiple + 1) * multiple - x
    get_pad = lambda x: (x // multiple + int(x % multiple != 0)) * multiple - x
    ph, pw = get_pad(h), get_pad(w)
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
            np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = np.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "sqrt":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedSampler(nn.Module):
    """
    Implementation for spaced sampling schedule proposed in IDDPM. This class is designed
    for sampling ControlLDM.

    https://arxiv.org/pdf/2102.09672.pdf
    """

    def __init__(self, betas: np.ndarray) -> "SpacedSampler":
        super().__init__()
        self.num_timesteps = len(betas)
        self.original_betas = betas
        self.original_alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
        self.context = {}


    def register(self, name: str, value: np.ndarray) -> None:
        self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    def make_schedule(self, num_steps: int) -> None:
        # calcualte betas for spaced sampling
        # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
        used_timesteps = space_timesteps(self.num_timesteps, str(num_steps))
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(self.original_alphas_cumprod):
            if i in used_timesteps:
                # marginal distribution is the same as q(x_{S_t}|x_0)
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        assert len(betas) == num_steps
        self.timesteps = np.array(sorted(list(used_timesteps)), dtype=np.int32)  # e.g. [0, 10, 20, ...]

        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # print(f"sampler sqrt_alphas_cumprod: {np.sqrt(alphas_cumprod)[-1]}")
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )
        posterior_mean_coef1 = (
                betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
                (1.0 - alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - alphas_cumprod)
        )

        self.register("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register("posterior_mean_coef1", posterior_mean_coef1)
        self.register("posterior_mean_coef2", posterior_mean_coef2)

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[
        torch.Tensor]:
        """
        Implement the posterior distribution q(x_{t-1}|x_t, x_0).

        Args:
            x_start (torch.Tensor): The predicted images (NCHW) in timestep `t`.
            x_t (torch.Tensor): The sampled intermediate variables (NCHW) of timestep `t`.
            t (torch.Tensor): Timestep (N) of `x_t`. `t` serves as an index to get
                parameters for each timestep.

        Returns:
            posterior_mean (torch.Tensor): Mean of the posterior distribution.
            posterior_variance (torch.Tensor): Variance of the posterior distribution.
            posterior_log_variance_clipped (torch.Tensor): Log variance of the posterior distribution.
        """
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def apply_cond_fn(
            self,
            model: ControlLDM,
            pred_x0: torch.Tensor,
            t: torch.Tensor,
            index: torch.Tensor,
            cond_fn: Guidance
    ) -> torch.Tensor:
        t_now = int(t[0].item()) + 1
        if not (cond_fn.t_stop < t_now and t_now < cond_fn.t_start):
            # stop guidance
            self.context["g_apply"] = False
            return pred_x0
        grad_rescale = 1 / extract_into_tensor(self.posterior_mean_coef1, index, pred_x0.shape)
        # apply guidance for multiple times
        loss_vals = []
        for _ in range(cond_fn.repeat):
            # set target and pred for gradient computation
            target, pred = None, None
            if cond_fn.space == "latent":
                target = model.vae_encode(cond_fn.target)
                pred = pred_x0
            elif cond_fn.space == "rgb":
                # We need to backward gradient to x0 in latent space, so it's required
                # to trace the computation graph while decoding the latent.
                with torch.enable_grad():
                    target = cond_fn.target
                    pred_x0_rg = pred_x0.detach().clone().requires_grad_(True)
                    pred = model.vae_decode(pred_x0_rg)
                    assert pred.requires_grad
            else:
                raise NotImplementedError(cond_fn.space)
            # compute gradient
            delta_pred, loss_val = cond_fn(target, pred, t_now)
            loss_vals.append(loss_val)
            # update pred_x0 w.r.t gradient
            if cond_fn.space == "latent":
                delta_pred_x0 = delta_pred
                pred_x0 = pred_x0 + delta_pred_x0 * grad_rescale
            elif cond_fn.space == "rgb":
                pred.backward(delta_pred)
                delta_pred_x0 = pred_x0_rg.grad
                pred_x0 = pred_x0 + delta_pred_x0 * grad_rescale
            else:
                raise NotImplementedError(cond_fn.space)
        self.context["g_apply"] = True
        self.context["g_loss"] = float(np.mean(loss_vals))
        return pred_x0

    def predict_noise(
            self,
            model: ControlLDM,
            x: torch.Tensor,
            t: torch.Tensor,
            cond: Dict[str, torch.Tensor],
            uncond: Optional[Dict[str, torch.Tensor]],
            cfg_scale: float
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.:
            model_output, kpn = model(x, t, cond)
        else:
            # apply classifier-free guidance
            model_cond,kpn = model(x, t, cond)
            model_uncond,_ = model(x, t, uncond)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        return model_output, kpn

    @torch.no_grad()
    def predict_noise_tiled(
            self,
            model: ControlLDM,
            x: torch.Tensor,
            t: torch.Tensor,
            cond: Dict[str, torch.Tensor],
            uncond: Optional[Dict[str, torch.Tensor]],
            cfg_scale: float,
            tile_size: int,
            tile_stride: int
    ):
        _, _, h, w = x.shape
        tiles = tqdm(sliding_windows(h, w, tile_size // 8, tile_stride // 8), unit="tile", leave=False)
        eps = torch.zeros_like(x)
        kpn = torch.zeros_like(x)
        count = torch.zeros_like(x, dtype=torch.float32)
        weights = gaussian_weights(tile_size // 8, tile_size // 8)[None, None]
        weights = torch.tensor(weights, dtype=torch.float32, device=x.device)
        for hi, hi_end, wi, wi_end in tiles:
            tiles.set_description(f"Process tile ({hi} {hi_end}), ({wi} {wi_end})")
            tile_x = x[:, :, hi:hi_end, wi:wi_end]
            tile_cond = {
                "c_img": cond["c_img"][:, :, hi:hi_end, wi:wi_end],
                "c_txt": cond["c_txt"]
            }
            if uncond:
                tile_uncond = {
                    "c_img": uncond["c_img"][:, :, hi:hi_end, wi:wi_end],
                    "c_txt": uncond["c_txt"]
                }
            tile_eps,tile_kpn = self.predict_noise(model, tile_x, t, tile_cond, tile_uncond, cfg_scale)
            kpn[:, :, hi:hi_end, wi:wi_end] = tile_kpn
            # accumulate noise
            eps[:, :, hi:hi_end, wi:wi_end] += tile_eps * weights
            count[:, :, hi:hi_end, wi:wi_end] += weights
        # average on noise (score)
        eps.div_(count)
        return eps,kpn

    @torch.no_grad()
    def p_sample(
            self,
            model: ControlLDM,
            x: torch.Tensor,
            t: torch.Tensor,
            index: torch.Tensor,
            cond: Dict[str, torch.Tensor],
            uncond: Optional[Dict[str, torch.Tensor]],
            cfg_scale: float,
            cond_fn: Optional[Guidance],
            tiled: bool,
            tile_size: int,
            tile_stride: int
    ) -> torch.Tensor:
        if tiled:
            eps, kpn = self.predict_noise_tiled(model, x, t, cond, uncond, cfg_scale, tile_size, tile_stride)
        else:
            eps, kpn = self.predict_noise(model, x, t, cond, uncond, cfg_scale)
        pred_x0 = self._predict_xstart_from_eps(x, index, eps)
        if cond_fn:
            assert not tiled, f"tiled sampling currently doesn't support guidance"
            pred_x0 = self.apply_cond_fn(model, pred_x0, t, index, cond_fn)
        model_mean, model_variance, _ = self.q_posterior_mean_variance(pred_x0, x, index)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (index != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        x_prev = model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
        return x_prev, kpn

    @torch.no_grad()
    def sample(
            self,
            model: ControlLDM,
            device: str,
            steps: int,
            batch_size: int,
            x_size: Tuple[int],
            cond: Dict[str, torch.Tensor],
            uncond: Dict[str, torch.Tensor],
            cfg_scale: float,
            cond_fn: Optional[Guidance] = None,
            tiled: bool = False,
            tile_size: int = -1,
            tile_stride: int = -1,
            x_T: Optional[torch.Tensor] = None,
            progress: bool = True,
            progress_leave: bool = True,
    ) -> torch.Tensor:
        self.make_schedule(steps)
        self.to(device)
        if x_T is None:
            # TODO: not convert to float32, may trigger an error
            img = torch.randn((batch_size, *x_size), device=device)
        else:
            img = x_T
        timesteps = np.flip(self.timesteps)  # [1000, 950, 900, ...]
        total_steps = len(self.timesteps)
        iterator = tqdm(timesteps, total=total_steps, leave=progress_leave, disable=not progress)
        for i, step in enumerate(iterator):
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            index = torch.full_like(ts, fill_value=total_steps - i - 1)
            img, kpn = self.p_sample(
                model, img, ts, index, cond, uncond, cfg_scale, cond_fn,
                tiled, tile_size, tile_stride
            )
            if cond_fn and self.context["g_apply"]:
                loss_val = self.context["g_loss"]
                desc = f"Spaced Sampler With Guidance, Loss: {loss_val:.6f}"
            else:
                desc = "Spaced Sampler"
            iterator.set_description(desc)
        return img

@ARCH_REGISTRY.register()
class Diffusion(nn.Module):

    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        parameterization="eps",
        guidance=False,
        g_loss='w_mse'
    ):
        super().__init__()
        self.num_timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.cosine_s = cosine_s
        assert parameterization in ["eps", "x0", "v"], "currently only supporting 'eps' and 'x0' and 'v'"
        self.parameterization = parameterization
        self.loss_type = loss_type

        self.steps = 50
        self.tiled = False
        self.tile_size = 512
        self.tile_stride = 256
        self.pos_prompt = ""
        self.neg_prompt = "low quality, blurry, low-resolution, noisy, unsharp, weird textures"
        self.cfg_scale = 1.0
        self.better_start = False

        self.guidance = guidance
        self.g_loss = g_loss
        self.g_scale = 0.0
        self.g_start = 1001
        self.g_stop = -1
        self.g_space = "latent"
        self.g_repeat = 1
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                   cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

        self.betas = betas
        self.register("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

        self.init_cond_fn()
    def init_cond_fn(self) -> None:
        if not self.guidance:
            self.cond_fn = None
            return
        if self.g_loss == "mse":
            cond_fn_cls = MSEGuidance
        elif self.g_loss == "w_mse":
            cond_fn_cls = WeightedMSEGuidance
        else:
            raise ValueError(self.g_loss)
        self.cond_fn = cond_fn_cls(
            scale=self.g_scale, t_start=self.g_start, t_stop=self.g_stop,
            space=self.g_space, repeat=self.g_repeat
        )
    def register(self, name: str, value: np.ndarray) -> None:
        self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    def q_sample(self, x_start, t, noise):
        # print(x_start.device, t.device, noise.device)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod.to(x.device), t, x.shape) * noise -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod.to(x.device), t, x.shape) * x
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, model, x_start, t, cond):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output, lr_kpn = model(x_noisy, t, cond)

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean()
        loss_kpn = self.get_loss(lr_kpn, x_start, mean=False).mean()
        loss = loss_kpn+loss_simple
        return loss

    def set_final_size(self, lq: torch.Tensor) -> None:
        h, w = lq.shape[2:]
        self.final_size = (h, w)


    @count_vram_usage
    def run_diff(
        self,
        cldm,
        clean: torch.Tensor,
        steps: int,
        strength: float,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
        pos_prompt: str,
        neg_prompt: str,
        cfg_scale: float,
        better_start: float
    ) -> torch.Tensor:
        ### preprocess
        bs, _, ori_h, ori_w = clean.shape
        # pad: ensure that height & width are multiples of 64
        pad_clean = pad_to_multiples_of(clean, multiple=64)
        h, w = pad_clean.shape[2:]
        # prepare conditon
        cldm = cldm.module if hasattr(cldm, 'module') else cldm
        if not tiled:
            cond = cldm.prepare_condition(pad_clean, [pos_prompt] * bs)
            uncond = cldm.prepare_condition(pad_clean, [neg_prompt] * bs)
        else:
            cond = cldm.prepare_condition_tiled(pad_clean, [pos_prompt] * bs, tile_size, tile_stride)
            uncond = cldm.prepare_condition_tiled(pad_clean, [neg_prompt] * bs, tile_size, tile_stride)
        if self.cond_fn:
            self.cond_fn.load_target(pad_clean * 2 - 1)
        old_control_scales = cldm.control_scales
        cldm.control_scales = [strength] * 13
        if better_start:
            # using noised low frequency part of condition as a better start point of
            # reverse sampling, which can prevent our model from generating noise in
            # image background.
            _, low_freq = wavelet_decomposition(pad_clean)
            if not tiled:
                x_0 = cldm.vae_encode(low_freq)
            else:
                x_0 = cldm.vae_encode_tiled(low_freq, tile_size, tile_stride)
            x_T = self.q_sample(
                x_0,
                torch.full((bs, ), self.num_timesteps - 1, dtype=torch.long, device=pad_clean.device),
                torch.randn(x_0.shape, dtype=torch.float32, device=pad_clean.device)
            )
            # print(f"diffusion sqrt_alphas_cumprod: {self.sqrt_alphas_cumprod[-1]}")
        else:
            x_T = torch.randn((bs, 4, h // 8, w // 8), dtype=torch.float32, device=pad_clean.device)
        ### run sampler
        sampler = SpacedSampler(self.betas)
        z = sampler.sample(
            model=cldm, device=x_T.device, steps=steps, batch_size=bs, x_size=(4, h // 8, w // 8),
            cond=cond, uncond=uncond, cfg_scale=cfg_scale, x_T=x_T, progress=True,
            progress_leave=True, cond_fn=self.cond_fn, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        if not tiled:
            x = cldm.vae_decode(z)
        else:
            x = cldm.vae_decode_tiled(z, tile_size // 8, tile_stride // 8)
        ### postprocess
        cldm.control_scales = old_control_scales
        sample = x[:, :, :ori_h, :ori_w]
        return sample

    @torch.no_grad()
    def run(
        self,
        cldm,
        lq
    ) -> np.ndarray:
        # image to tensor
        # lq = torch.tensor((lq / 255.).clip(0, 1), dtype=torch.float32, device=self.device)
        # lq = rearrange(lq, "n h w c -> n c h w").contiguous()
        # set pipeline output size
        self.set_final_size(lq)
        clean = lq
        sample = self.run_diff(
            cldm, clean, self.steps, 1.0, self.tiled,
        self.tile_size, self.tile_stride,
        self.pos_prompt, self.neg_prompt, self.cfg_scale,
        self.better_start
        )
        # colorfix (borrowed from StableSR, thanks for their work)
        sample = (sample + 1) / 2
        sample = adaptive_instance_normalization(sample, clean)

        # sample = rearrange(sample * 255., "n c h w -> n h w c")
        # sample = sample.contiguous().clamp(0, 255).to(torch.uint8).cpu().numpy()
        return sample.contiguous().clamp(0., 1.)


