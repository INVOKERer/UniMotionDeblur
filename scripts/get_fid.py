import numpy as np
from cleanfid import fid
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mode 保持默认的 "clean" 以保证与 pyiqa 的计算结果对齐
feat_model = fid.build_feature_extractor(mode="clean", device=device)

gt_folders = [
    "/data/mxt_data/GoPro/train/sharp",
    "/home/ubuntu/150t/personal_data/mxt/Dataset/HIDE_dataset/GT"
]

all_features = []

print("开始逐个文件夹提取特征...")
for i, folder in enumerate(gt_folders):
    print(f"[{i+1}/{len(gt_folders)}] 正在处理: {folder}")
    # 提取单个文件夹的特征 (shape: [N, 2048])
    features = fid.get_folder_features(
        folder,
        model=feat_model,  # <--- 解决 NoneType 报错的关键
        batch_size=32,
        device=device,
        verbose=True  # 关掉内部进度条，保持终端整洁
    )
    all_features.append(features)

# 沿着第 0 维度将所有特征拼接起来
print("正在合并特征...")
merged_features = np.concatenate(all_features, axis=0)
print(f"总计提取了 {merged_features.shape[0]} 张图像的特征")

# 计算总特征的均值和协方差
mu_real = np.mean(merged_features, axis=0)
sigma_real = np.cov(merged_features, rowvar=False)

# 保存到本地项目目录
np.savez("./1080_gt_stats.npz", mu=mu_real, sigma=sigma_real)
print("统计量已保存！")

# # =========================================================
# # 评测时，直接使用 clean-fid 的底层接口计算生成图像
# # =========================================================
# gen_dir = "./path/to/my_deeprftv2_results"
# gen_features = fid.get_folder_features(gen_dir, batch_size=32)
#
# score = fid.frechet_distance(
#     np.mean(gen_features, axis=0),
#     np.cov(gen_features, rowvar=False),
#     mu_real,
#     sigma_real
# )
# print(f"最终 FID Score: {score}")