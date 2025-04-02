import numpy as np
import copy
import random
from functools import partial

import logging
import torch
import torch.utils.data.distributed as torch_data_dist
from torch.utils.data import DataLoader

from gdet.structures.configure import TrainConfigType, DatasetConfigType, ConfigType
from gdet.registries import DATASETS
from gdet.datasets.transforms_factory import construct_transforms
from gdet.engine import schedulers
from .datasets.collate import collate

logger = logging.getLogger("gdet.data")

def init_train_sampler_loader(cfg: "ConfigType", train_dataset):
    cfg_train: "TrainConfigType" = cfg.train
    dl_cfg = cfg_train.data_loader
    sampling_strategy = dl_cfg.sampling_strategy
    samples_per_gpu = dl_cfg.samples_per_gpu
    workers_per_gpu = dl_cfg.workers_per_gpu
    dist_cfg = cfg.dist

    if dist_cfg.distributed:
        if sampling_strategy != 'instance_balanced':
            msg = 'Resampling is not allowed when distributed parallel is applied. \
                   Please set sampling_strategy to instance_balanced.'
            logger.warning(msg)
            exit()

        train_sampler = torch_data_dist.DistributedSampler(
            train_dataset,
            num_replicas=dist_cfg.world_size,
            rank=dist_cfg.rank
        )
    else:
        if sampling_strategy == 'class_balanced':
            train_sampler = schedulers.ScheduledWeightedSampler(train_dataset, 1)
        elif sampling_strategy == 'progressively_balanced':
            train_sampler = schedulers.ScheduledWeightedSampler(train_dataset, cfg.data.sampling_weights_decay_rate)
        elif sampling_strategy == 'instance_balanced':
            train_sampler = None
        else:
            raise NotImplementedError('Not implemented resampling strategy.')
    # batch_size  = cfg_train.batch_size
    # num_workers = cfg_train.num_workers

    num_gpus        = len(dist_cfg.gpu_ids)
    batch_size  = num_gpus * samples_per_gpu
    num_workers = num_gpus * workers_per_gpu
    pin_memory = dl_cfg.pin_memory
    seed       = cfg.experiment.seed
    generator = torch.Generator()
    
    if seed is not None:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=0, seed=seed)  
        generator.manual_seed(seed)  # 固定生成器的种子
    else:
        init_fn = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        worker_init_fn=init_fn,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        drop_last=True,
        pin_memory=pin_memory,
        generator=generator,
    )

    return train_loader, train_sampler

def init_val_sampler_loader(cfg, val_dataset):
    cfg_val = cfg.val
    dist_cfg = cfg.dist
    dl_cfg = cfg_val.data_loader
    sampling_strategy = dl_cfg.sampling_strategy
    samples_per_gpu = dl_cfg.samples_per_gpu
    workers_per_gpu = dl_cfg.workers_per_gpu
    
    if dist_cfg.distributed:
        if sampling_strategy != 'instance_balanced':
            msg = 'Resampling is not allowed when distributed parallel is applied. \
                   Please set sampling_strategy to instance_balanced.'
            logger.warning(msg)
            exit()
        val_sampler = torch_data_dist.DistributedSampler(
            val_dataset,
            num_replicas=dist_cfg.world_size,
            rank=dist_cfg.rank
        )
    else:
        val_sampler = None

    num_gpus        = len(dist_cfg.gpu_ids)
    batch_size  = num_gpus * samples_per_gpu
    num_workers = num_gpus * workers_per_gpu
    pin_memory = dl_cfg.pin_memory
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=dl_cfg.shuffle,
        sampler=val_sampler,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )
    return val_loader, val_sampler

def worker_init_fn(worker_id, num_workers: int, rank: int, seed: int, disable_subprocess_warning: bool = False):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    # if disable_subprocess_warning and worker_id != 0:
    #     warnings.simplefilter('ignore')

def construct_dataset(cfg: "DatasetConfigType"):
    data_cfg = cfg.clone()
    data_cls_type = data_cfg.pop('type')
    data_cls = DATASETS.get(data_cls_type)
    assert data_cls is not None, f"data_cls_type: {data_cls_type}"

    train_transform = construct_transforms(data_cfg.transforms)
    dataset = data_cls(data_cfg, transforms=train_transform)
    dataset.init()
    return dataset

# define data loader
def construct_labeled_dataset(cfg, dstype: str):
    """cfg: root config
    """
    data_cfg = cfg.dataset
    label_data_cfg = data_cfg[dstype]
    # label_data_cfg = copy.deepcopy(label_data_cfg)
    dataset = construct_dataset(label_data_cfg)
    if dstype == "train":
        dloader, data_sampler = init_train_sampler_loader(cfg, dataset)
    else:
        dloader, data_sampler = init_val_sampler_loader(cfg, dataset)

    return dataset, dloader

def construct_trainval_dataset(cfg):
    train_dataset, train_loader = construct_labeled_dataset(cfg, 'train')
    val_dataset, val_loader = construct_labeled_dataset(cfg, 'val')
    
    return train_dataset, train_loader, val_dataset, val_loader

def construct_test_dataset(cfg):
    data_cfg = cfg.dataset.copy()
    
    test_data_cfg = data_cfg['test'].clone()
    dtype = test_data_cfg.pop('type')
    data_cls = DATASETS.get(dtype)
    
    test_transform = construct_transforms(test_data_cfg.transforms)

    test_dataset = data_cls(test_data_cfg, test_transform)
    if hasattr(cfg, "categories"):
        test_dataset.categories = copy.deepcopy(cfg.categories)
    test_dataset.init()
    
    test_loader, test_sampler = init_val_sampler_loader(cfg, test_dataset)

    return test_dataset, test_loader
