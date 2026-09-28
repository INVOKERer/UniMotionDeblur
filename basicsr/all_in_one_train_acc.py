import datetime
import logging
import math

# isort: off
import sys

import torch.utils
import torch.utils.data

sys.path.append(".")
# isort: on

import time
import warnings
from os import path as osp

import torch
from accelerate import Accelerator
warnings.filterwarnings("ignore")

from basicsr.data import build_dataloader2, build_dataset
from basicsr.data.concat_dataset import ConcatDataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (
    AvgTimer,
    MessageLogger,
    check_resume,
    check_resume_all_in_one,
    get_env_info,
    get_root_logger,
    get_time_str,
    init_tb_logger,
    init_wandb_logger,
    make_exp_dirs,
    mkdir_and_rename,
    scandir,
)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (
        (opt["logger"].get("wandb") is not None)
        and (opt["logger"]["wandb"].get("project") is not None)
        and ("debug" not in opt["name"])
    ):
        assert (
            opt["logger"].get("use_tb_logger") is True
        ), "should turn on tensorboard when using wandb"
        init_wandb_logger(opt)
    tb_logger = None
    if opt["logger"].get("use_tb_logger") and "debug" not in opt["name"]:
        tb_logger = init_tb_logger(
            log_dir=osp.join(opt["root_path"], "tb_logger", opt["name"])
        )
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader, train_sets, val_sets, enlarge_ratios = (
        None,
        None,
        [],
        [],
        [],
    )
    train_class = []
    val_class = []
    for phase, dataset_opt in opt["datasets"].items():
        if phase.split("_")[0] == "train":
            train_set = build_dataset(dataset_opt)
            train_sets.append(train_set)
            train_class.append(dataset_opt.get("dataset_class_idx"))
            enlarge_ratios.append(dataset_opt.get("dataset_enlarge_ratio", 1))
            logger.info(
                f'Number / Enlarged of train images/folders in {dataset_opt["name"]}: {len(train_set)}, {len(train_set) * enlarge_ratios[-1]}'
            )
        elif phase.split("_")[0] == "val":
            val_set = build_dataset(dataset_opt)
            val_sets.append(val_set)
            val_class.append(dataset_opt.get("dataset_class_idx"))
        else:
            raise ValueError(f"Dataset phase {phase} is not recognized.")

    dataloader_opt = opt["dataloader"]
    dataset_enlarge_ratio = dataloader_opt.get("dataset_enlarge_ratio", 1)

    train_set = ConcatDataset(train_sets, enlarge_ratios, train_class)
    train_sampler = EnlargedSampler(
        train_set, opt["world_size"], opt["rank"], dataset_enlarge_ratio
    )
    train_loader = build_dataloader2(
        train_set,
        dataloader_opt,
        num_gpu=opt["num_gpu"],
        dist=opt["dist"],
        sampler=train_sampler,
        seed=opt["manual_seed"],
    )

    val_set = ConcatDataset(val_sets, [1] * len(val_sets), val_class)
    val_dataloader_opt = (
        opt["dataloader_val"] if hasattr(opt, "dataloader_val") else dataloader_opt
    )
    val_loader = build_dataloader2(
        val_set,
        val_dataloader_opt,
        num_gpu=opt["num_gpu"],
        dist=opt["dist"],
        sampler=None,
        seed=opt["manual_seed"],
    )

    num_iter_per_epoch = math.ceil(
        len(train_set)
        * dataset_enlarge_ratio
        / (dataloader_opt["batch_size_per_gpu"] * opt["world_size"])
    )
    total_iters = int(opt["train"]["total_iter"])
    total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
    logger.info(
        "Training statistics:"
        f"\n\tNumber of train images: {len(train_set)}"
        f"\n\tDataset enlarge ratio: {dataset_enlarge_ratio}"
        f'\n\tBatch size per gpu: {dataloader_opt["batch_size_per_gpu"]}'
        f'\n\tWorld size (gpu number): {opt["world_size"]}'
        f"\n\tRequire iter number per epoch: {num_iter_per_epoch}"
        f"\n\tTotal epochs: {total_epochs}; iters: {total_iters}."
    )

    logger.info(f"Number of val images/folders: {len(val_set)}")

    return (
        train_set,
        train_loader,
        train_sampler,
        val_loader,
        total_epochs,
        total_iters,
    )


def create_val_dataloaders(opt, logger, acc):
    # create val dataloaders
    val_loaders = []
    for phase, dataset_opt in opt["datasets"].items():
        if phase.split("_")[0] == "val":
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader2(
                val_set,
                dataset_opt,
                num_gpu=opt["num_gpu"],
                dist=opt["dist"],
                sampler=None,
                seed=opt["manual_seed"],
            )
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}'
            )
            val_loader = acc.prepare(val_loader)
            val_loaders.append(val_loader)

    return val_loaders


def load_resume_state(opt):
    resume_state_path = None
    if opt["auto_resume"]:
        state_path = osp.join("experiments", opt["name"], "training_states")
        if osp.isdir(state_path):
            states = list(
                scandir(state_path, suffix="state", recursive=False, full_path=False)
            )
            if len(states) != 0:
                states = [float(v.split(".state")[0]) for v in states]
                resume_state_path = osp.join(state_path, f"{max(states):.0f}.state")
                opt["path"]["resume_state"] = resume_state_path
    else:
        if opt["path"].get("resume_state"):
            resume_state_path = opt["path"]["resume_state"]

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id)
        )
        if opt.get("classify", False):
            check_resume_all_in_one(opt, resume_state["iter"])
        else:
            check_resume(opt, resume_state["iter"], opt.get("resume_remove_dc", False))
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt["root_path"] = root_path

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.deterministic = True
    accelerator = Accelerator(mixed_precision=opt['Accelerator']['mixed_precision'], split_batches=opt['Accelerator']['split_batches'])
    opt["dist"] = accelerator.num_processes > 1
    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None and accelerator.is_main_process:
        make_exp_dirs(opt)
        if (
            opt["logger"].get("use_tb_logger")
            and "debug" not in opt["name"]
            and opt["rank"] == 0
        ):
            mkdir_and_rename(osp.join(opt["root_path"], "tb_logger", opt["name"]))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt["path"]["experiments_root"])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt["path"]["log"], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name="basicsr", log_level=logging.INFO, log_file=log_file
    )
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    (
        train_set,
        train_loader,
        train_sampler,
        val_loader,
        total_epochs,
        total_iters,
    ) = result
    classify = opt.get("classify", False)
    if not classify:
        del val_loader
        val_loaders = create_val_dataloaders(opt, logger, accelerator)

    # create model
    model = build_model(opt)
    if resume_state and accelerator.is_main_process:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(
            f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}."
        )
        start_epoch = resume_state["epoch"]
        start_iter = current_iter = resume_state["iter"]
    else:
        start_epoch = 0
        start_iter = current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    net_g = model.net_g
    optim_g = model.optimizer_g
    scheduler_g = model.schedulers[0]  # 取出 BasicSR 内部生成的 scheduler

    # 将它们全部交给 Accelerate 一次性包装
    if not classify:
        prepared_objects = accelerator.prepare(
            net_g, optim_g, scheduler_g, train_loader
        )

    else:
        prepared_objects = accelerator.prepare(
            net_g, optim_g, scheduler_g, train_loader, val_loader
        )
        val_loader = prepared_objects[4]

    # 按照顺序将包装后的对象原路放回模型实例中
    model.net_g = prepared_objects[0]
    model.optimizer_g = prepared_objects[1]
    model.schedulers[0] = prepared_objects[2]  # [关键] 将包装后的 scheduler 放回原处
    train_loader = prepared_objects[3]


    model.accelerator = accelerator
    # ================= 修改结束 =================
    # dataloader prefetcher
    # prefetch_mode = opt["dataloader"].get("prefetch_mode")
    # if prefetch_mode is None or prefetch_mode == "cpu":
    #     prefetcher = CPUPrefetcher(train_loader)
    # elif prefetch_mode == "cuda":
    #     prefetcher = CUDAPrefetcher(train_loader, opt)
    #     logger.info(f"Use {prefetch_mode} prefetch dataloader")
    #     if opt["dataloader"].get("pin_memory") is not True:
    #         raise ValueError("Please set pin_memory=True for CUDAPrefetcher.")
    # else:
    #     raise ValueError(
    #         f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'."
    #     )

    # training
    logger.info(f"Start training from epoch: {start_epoch}, iter: {current_iter}")
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    # Progressive learning
    flag_progressive, stage = False, 0
    if opt.get("progressive_learning", None) is not None:
        flag_progressive = True
        progressive_iters = opt["progressive_learning"].get("iters", None)
        progressive_gtsize = opt["progressive_learning"].get("gtsize", None)
        progressive_bs = opt["progressive_learning"].get("bs", None)
        for progressive_iter in progressive_iters:
            if progressive_iter < current_iter:
                stage += 1

        opt["dataloader"]["batch_size_per_gpu"] = progressive_bs[stage]
        train_set.gt_size = progressive_gtsize[stage]
        del train_loader
        train_loader = build_dataloader2(
            train_set,
            opt["dataloader"],
            num_gpu=opt["num_gpu"],
            dist=opt["dist"],
            sampler=train_sampler,
            seed=opt["manual_seed"],
            phase="train",
        )
        prefetcher.ori_loader = train_loader
        prefetcher.reset()

        logger.info(f"current stage is: {stage}")
        logger.info(f"current gt_size is: {prefetcher.ori_loader.dataset.gt_size}")
        logger.info(f"current batch size is: {prefetcher.ori_loader.batch_size}")

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        # prefetcher.reset()
        # train_data = prefetcher.next()

        # while train_data is not None:
        for i, train_data in enumerate(train_loader):
            data_timer.record()

            # # validation
            # if opt.get("val") is not None and current_iter == start_iter:
            #     if classify:
            #         model.validation(val_loader, current_iter, tb_logger, False)
            #     else:
            #         if len(val_loaders) > 1:
            #             logger.warning(
            #                 "Multiple validation datasets are *only* supported by SRModel."
            #             )
            #         for i, val_loader in enumerate(val_loaders):
            #             model.validation(
            #                 val_loader,
            #                 current_iter,
            #                 tb_logger,
            #                 opt["val"]["save_img"],
            #                 # dataset_idx=i,
            #             )

            if flag_progressive:
                # print(current_iter, stage, progressive_iters[stage])
                if current_iter == progressive_iters[stage]:
                    # print("what's up?")
                    stage += 1
                    current_gt_size = progressive_gtsize[stage]
                    current_batch_size = progressive_bs[stage]
                    opt["dataloader"]["batch_size_per_gpu"] = current_batch_size
                    train_set.gt_size = current_gt_size
                    del train_loader

                    train_loader = build_dataloader2(
                        train_set,
                        opt["dataloader"],
                        num_gpu=opt["num_gpu"],
                        dist=opt["dist"],
                        sampler=train_sampler,
                        seed=opt["manual_seed"],
                        phase="train",
                    )
                    # prefetcher.ori_loader = train_loader
                    # prefetcher.reset()

                    logger.info(f"current stage is: {stage}")
                    # logger.info(
                    #     f"current gt_size is: {prefetcher.ori_loader.dataset.gt_size}"
                    # )
                    # logger.info(
                    #     f"current batch size is: {prefetcher.ori_loader.batch_size}"
                    # )

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt["train"].get("warmup_iter", -1)
            )
            # training
            model.feed_data(train_data)
            model.optimize_parameters_acc(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt["logger"]["print_freq"] == 0:
                log_vars = {"epoch": epoch, "iter": current_iter}
                log_vars.update({"lrs": model.get_current_learning_rate()})
                log_vars.update(
                    {
                        "time": iter_timer.get_avg_time(),
                        "data_time": data_timer.get_avg_time(),
                    }
                )
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving models and training states.")
                model.save(epoch, current_iter)

            # validation
            if opt.get("val") is not None and (
                current_iter % opt["val"]["val_freq"] == 0
            ):
                if classify:
                    model.validation_acc(val_loader, current_iter, tb_logger, False)
                else:
                    if len(val_loaders) > 1:
                        logger.warning(
                            "Multiple validation datasets are *only* supported by SRModel."
                        )
                    for i, val_loader in enumerate(val_loaders):
                        model.validation_acc(
                            val_loader,
                            current_iter,
                            tb_logger,
                            opt["val"]["save_img"],
                            # dataset_idx=i,
                        )

            data_timer.start()
            iter_timer.start()
            # train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"End of training. Time consumed: {consumed_time}")
    logger.info("Save the latest model.")
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest

    if tb_logger:
        tb_logger.close()


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
