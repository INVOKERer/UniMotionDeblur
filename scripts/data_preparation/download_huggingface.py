from huggingface_hub import snapshot_download
import os
import glob
# 核心：代码内配置国内镜像源（hf-mirror.com），无需终端额外配置
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# export HF_ENDPOINT=https://hf-mirror.com
# 下载数据集到指定本地目录
# dir_name = '30000_39999/blurred' # blurred groundtruth
# dir_name = ''
snapshot_download(
    repo_id="timseizinger/RealBokeh_3MP",  # 数据集在Hugging Face的仓库ID "masterda/SloMoBlur"
    # allow_patterns=dir_name + '/*.png',
    repo_type="dataset",  # 明确指定下载的是数据集（而非模型）
    # cache_dir="/data/mxt_data/SloMoBlur/0_9999/groundtruth",
    local_dir="/data/mxt_data/Bokeh",  # 本地存储路径，可自行修改
    # local_dir="/home/nora/150t/personal_data/mxt/Dataset/SloMoBlur",

    # local_dir_use_symlinks=False,  # 禁用软链接，避免跨平台/权限问题
    # resume_download=True  # 断点续传，网络中断后无需重新下载
)

# start = 30000
# existed_files = glob.glob(os.path.join('/data/mxt_data/SloMoBlur', dir_name, '*.png'))
# n_start = max(0, len(existed_files)-5)
# for i in range(n_start, 10000):
#     k = str(start + i).zfill(5)
#     files = glob.glob(os.path.join('/data/mxt_data/SloMoBlur', dir_name, '*{}.png'.format(k)))
#     if len(files) > 0:
#         print(files[0], ' exists.')
#         continue
#     snapshot_download(
#         repo_id="masterda/SloMoBlur",  # 数据集在Hugging Face的仓库ID
#     allow_patterns=dir_name+'/*{}.png'.format(k),
#         repo_type="dataset",  # 明确指定下载的是数据集（而非模型）
#     # cache_dir="/data/mxt_data/SloMoBlur/0_9999/groundtruth",
#         local_dir="/data/mxt_data/SloMoBlur",  # 本地存储路径，可自行修改
#         # local_dir_use_symlinks=False,  # 禁用软链接，避免跨平台/权限问题
#         # resume_download=True  # 断点续传，网络中断后无需重新下载
#     )

# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download --resume-download masterda/SloMoBlur --repo-type dataset --local-dir /data/mxt_data/SloMoBlur/
# git lfs clone https://hf-mirror.com/datasets/masterda/SloMoBlur