import numpy as np
from cleanfid import fid
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mode 保持默认的 "clean" 以保证与 pyiqa 的计算结果对齐
feat_model = fid.build_feature_extractor(mode="clean", device=device)


# # =========================================================
# # 评测时，直接使用 clean-fid 的底层接口计算生成图像
# # =========================================================

stats_path = "./1080_gt_stats.npz"

# 使用 with 语句，读取完后会自动释放文件资源
with np.load(stats_path) as data:
    mu_real = data['mu']
    sigma_real = data['sigma']
gen_dir = "/data/mxt_code/AllinOne/results/UniMotionDeblur_EVSSM/visualization/GoPro"
gen_features = fid.get_folder_features(
    gen_dir,
    model=feat_model,  # <--- 解决 NoneType 报错的关键
    batch_size=32,
    device=device
)
print(gen_features.shape)
score = fid.frechet_distance(
    np.mean(gen_features, axis=0),
    np.cov(gen_features, rowvar=False),
    mu_real,
    sigma_real
)
print(f"最终 FID Score: {score}")