import logging

# isort: off
import sys

sys.path.append(".")
# isort: on
import warnings
from os import path as osp

import torch

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
# import torch.nn as nn
warnings.filterwarnings("ignore")
import os
# os.environ['RANK'] = str(0)
# os.environ['WORLD_SIZE'] = str(4)
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = str(9990)
def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt["path"]["log"], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name="basicsr", log_level=logging.INFO, log_file=log_file
    )
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt["datasets"].items()):
        print(dataset_opt['name'])
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt["num_gpu"],
            dist=opt["dist"],
            sampler=None,
            seed=opt["manual_seed"],
        )
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)
    # model = nn.DataParallel(model)
    metric_results_dict = {}
    metric_avg_lists = opt["metric_avg_list"]
    i = 0
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt["name"]
        logger.info(f"Testing {test_set_name}...")
        model.validation(
            test_loader,
            current_iter=opt["name"],
            tb_logger=None,
            save_img=opt["val"]["save_img"],
        )
        metric_results_dict[test_set_name] = model.metric_results
        # print( model.metric_results)
        if i == 0:
            metric_results = model.metric_results.copy()
        else:
            for metric in metric_results.keys():
                metric_results[metric] += model.metric_results[metric]
        i += 1
    log_str = f"Average Metric \n"
    for metric, value in metric_results.items():
        log_str += f"\t # Average {metric}: {value / float(i):.4f}"
        log_str += "\n"
    # print(metric_results_dict['LOLBlur'])
    align_list = opt["align_list"]['names']
    if len(metric_avg_lists) > 0:
        for i, metric_avg_list in enumerate(metric_avg_lists):
            values_save = ['PSNR', 'SSIM', 'CLIPIQA', 'MA-CLIP', 'MUSIQ']
            if metric_avg_list[0] in align_list:
                values_save = ['PSNR', 'SSIM', 'CLIPIQA', 'MA-CLIP', 'MUSIQ', 'PSNR*', 'SSIM*']
            values = ''
            names = ''
            for n, name in enumerate(metric_avg_list):
                # print(n, name, metric_results_dict[name])
                if n == 0:
                    metric_result_ = metric_results_dict[name].copy()
                else:
                    for metric in metric_result_.keys():
                        metric_result_[metric] += metric_results_dict[name][metric]
                names += name + ', '
            log_str += f"Average Metric  {names}\n"
            for metric, value in metric_result_.items():
                # print(value, len(metric_avg_list))
                avg_value = value / float(len(metric_avg_list))
                log_str += f"\t # Average {metric}: {avg_value:.4f}"
                log_str += "\n"

                if metric in values_save:
                    values += f"&{avg_value:.4f} "
            log_str += values
            log_str += "\n"
    logger.info(log_str)


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
