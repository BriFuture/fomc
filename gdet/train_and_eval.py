# coding=utf-8
import logging

from argparse import ArgumentParser
import sys, os, os.path as osp
from bfcommon.config import Config
from bfcommon.utils import set_random_seed
from bfcommon.logger import setup_logger

import torch
from gdet.registries import EXPS
from gdet.engine import ExpBase
from gdet.structures.configure import ConfigType, DistConfigType, ExpTrainConfigType
from gdet.parallel.dist_utils import get_dist_info

logger = logging.getLogger("gdet.train")
CURR_DIR = osp.dirname(__file__)
ROOT_DIR = osp.abspath(osp.join(CURR_DIR, ".."))

def construct_parser():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--launcher", type=str, default="")
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--calc_flops", action='store_true')
    parser.add_argument("--no_eval", action='store_true')
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--load_from", type=str, default="")
    parser.add_argument("--copy_best", action='store_true')
    parser.add_argument("--no_save_best", action="store_true")
    parser.add_argument("--no_save_val_best", action="store_true")
    parser.add_argument("--no_save_train_best", action="store_true")
    parser.add_argument("--no_eval_during_train", action="store_true")
    return parser



def prepare_args_cfg(args=None, local_rank=0):
    parser = construct_parser()
    args = parser.parse_args(args)
    set_random_seed(args.seed)
    # torch.autograd.set_detect_anomaly(True)
    ExpBase.register_config_path()
    cfg: "ConfigType" = Config.fromfile(args.config)
    cfg.experiment.seed = args.seed
    if args.load_from:
        cfg.experiment.load_from = args.load_from
    if args.launcher:
        cfg.dist.launcher = args.launcher

    cfg.train.auto_copy_best = args.copy_best
    cfg.freeze()

    cfg = ExpBase.filter_config(cfg)
    ExpBase.load_spec_module(cfg)
    ### setup logger 
    wd = cfg.experiment.work_dir
    os.makedirs(wd, exist_ok=True)
    if args.train:
        log_name = "train.log"
    else:
        log_name = "eval.log"
    if local_rank != 0:
        log_name = f"rank{local_rank}_{log_name}"
        setup_logger(name='fvlm', output=osp.join(wd, log_name), stdout=False)
        setup_logger(name='gdet', output=osp.join(wd, log_name), stdout=False)
        ### for mmcv repo
        log_name = f"rank{local_rank}_mmcv.log"
        setup_logger(name='mmcv', output=osp.join(wd, "mmcv.log"), stdout=False)
    else:
        setup_logger(name='fvlm', output=osp.join(wd, log_name))
        setup_logger(name='gdet', output=osp.join(wd, log_name))
        setup_logger(name='mmcv', output=osp.join(wd, "mmcv.log"))
    test_ind = "train" if args.train else "test"
    cfg_dump_path = osp.join(wd, f"config_{test_ind}.py")
    if local_rank == 0:
        print(f"[{local_rank}] Save config into: {cfg_dump_path}")
        cfg.dump(cfg_dump_path)
    return args, cfg


import psutil
def check_run_programe_name(dist=False):
    """
    """
    # 获取当前进程的父进程信息
    parent_process = psutil.Process().parent()
    executable_name = parent_process.name()
    dist_programs = ['torchrun']
    if dist:
        assert executable_name in dist_programs, f"{executable_name} dist: {dist}"
    else:
        assert executable_name not in dist_programs, f"{executable_name} dist: {dist}"

def is_programe_dist():
    """
    """
    # 获取当前进程的父进程信息
    parent_process = psutil.Process().parent()
    executable_name = parent_process.name()
    dist_programs = ['torchrun']
    return executable_name in dist_programs


def main(args=None):
    input_args = sys.argv if args is None else args
    check_run_programe_name(False)
    args, cfg = prepare_args_cfg(args)
    exp_type = cfg.experiment.type
    exp_cls = EXPS.get(exp_type)
    exp: "ExpTrainConfigType" = exp_cls(cfg)
    logger.info(f"Sys args: {input_args}")
    exp.merge_args(args)
    exp.init()

    if args.train:
        exp.train()
    if args.calc_flops:
        exp.calc_flops()
    # rank, _ = get_dist_info()
    if not args.no_eval:
        exp.evaluate_val()
        # exp.evaluate_test()
