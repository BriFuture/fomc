import torch
import numpy as np
from bfcommon.config import Config
import os
import os.path as osp
import copy
import importlib
import logging
from typing import Sequence

from gdet.structures.configure import ConfigType, SolverConfigType, TrainConfigType, DistConfigType
from gdet.registries import EXPS, MODULE_BUILD_FUNCS
from gdet.parallel import MMDataParallel, MMDistributedDataParallel, init_dist
logger = logging.getLogger("gdet.engine.eval")

@EXPS.register_module()
class ExpEval():
    def __init__(self, config: "ConfigType") -> None:
        self.m_config = config
        self.m_save_train_best = False
        self.m_save_val_best = True
        self.m_eval_during_train = True
        self.m_iter = 0
        self.m_train_mode = False
        self.preprocess_config()

    def merge_args(self, args: dict):
        self.m_save_val_best = not args.no_save_best and not args.no_save_val_best
        self.m_save_train_best = not args.no_save_best and not args.no_save_train_best
        # print(f"Save best of train: {self.m_save_train_best} val: {self.m_save_val_best} ")
        self.m_eval_during_train = not args.no_eval_during_train
        
    def init(self):
        cfg = self.m_config
        try:
            self.device = torch.device('cuda', torch.cuda.current_device())
        except:
            self.device = torch.device('cpu', )

        model_builder = cfg.model.builder
        construct_func = MODULE_BUILD_FUNCS.get(model_builder)
        assert construct_func is not None, model_builder
        
        self.m_model: "torch.nn.Module" = construct_func(copy.deepcopy(cfg))
        ### assert self.m_model is not None
        
        self.m_model.init()
        self.m_val_dataset = None
        self.m_model = self.m_model.to(device=self.device)
        self.load_model_state()
        self.setup_dist_model()
    
    def setup_dist_model(self):
        """ note train only
        """
        dist_cfg: "DistConfigType" = self.m_config.dist
        # put model on gpus
        if dist_cfg.distributed:
            find_unused_parameters = dist_cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            # if dist_cfg.gpu_nums > 0:
            #     gpu_nums = dist_cfg.gpu_nums
            #     gpu_ids = dist_cfg.gpu_ids[:gpu_nums]
            # else:
            #     gpu_nums = torch.cuda.device_count()
            #     gpu_ids = [i for i in range(gpu_nums)]

            gpu_ids = [self.device]
            model = MMDistributedDataParallel(
                self.m_model.cuda(),
                device_ids=gpu_ids,
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            # gpu_ids = dist_cfg.gpu_ids
            gpu_ids = [self.device]
            model = MMDataParallel(self.m_model.cuda(torch.cuda.current_device()), device_ids=gpu_ids)
        self.m_dist_model = model

    def load_model_state(self):
        exp_cfg = self.m_config.experiment
        load_from = exp_cfg.load_from
        if load_from:
            if not osp.exists(load_from):
                raise ValueError(f"Checkpoint {load_from} not exists, please change another one")
            saved_data = torch.load(load_from, map_location="cpu")
            state_dict = saved_data['state_dict']
            self.m_model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded From : {load_from}")

    @property
    def model(self):
        return self.m_model

    def preprocess_config(self):
        pass

    def evaluate_val(self):
        pass 

    def evaluate_test(self):
        pass 


import torch.distributed as dist
def setup_distributed(gpu: "int", n_gpus: "int", cfg):
    torch.cuda.set_device(gpu)
    dist_cfg = cfg.dist
    # dist_cfg.gpu = gpu
    dist_cfg.rank = dist_cfg.rank * n_gpus + gpu
    ws = dist_cfg.world_size
    dist.init_process_group(
        backend=dist_cfg.backend,
        init_method='env://',
        world_size=ws,
        rank=dist_cfg.rank
    )
    dist.barrier()
    ## batch size 设置的是每个gpu 的 batch size
    # cfg.train.batch_size = int(cfg.train.batch_size / dist_cfg.world_size)
    # cfg.train.num_workers = int((cfg.train.num_workers + n_gpus - 1) / n_gpus)

    # suppress printing
    if dist_cfg.gpu != 0 or dist_cfg.rank != 0:
        # cfg.base.progress = False
        def print_pass(*args):
            pass
        # builtins.print = print_pass