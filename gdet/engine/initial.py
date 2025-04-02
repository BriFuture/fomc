import tqdm
import os, os.path as osp
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

from torch.utils.data import DataLoader

from gdet.engine import schedulers
from gdet.losses import losses
from gdet.structures.configure import ConfigType, SolverConfigType, TrainConfigType

import logging

logger = logging.getLogger("gdet.engine.train")


def initialize_loss(cfg: "ConfigType", train_dataset):
    train_cfg: "TrainConfigType" = cfg.train
    criterion = train_cfg.criterion
    criterion_args = cfg.criterion_args[criterion]

    weight = None
    loss_weight_scheduler = None
    loss_weight = train_cfg.loss_weight
    if criterion == 'cross_entropy':
        if loss_weight == 'balance':
            loss_weight_scheduler = schedulers.LossWeightsScheduler(train_dataset, 1)
        elif loss_weight == 'dynamic':
            loss_weight_scheduler = schedulers.LossWeightsScheduler(train_dataset, train_cfg.loss_weight_decay_rate)
        elif isinstance(loss_weight, list):
            assert len(loss_weight) == len(train_dataset.classes)
            weight = torch.as_tensor(loss_weight, dtype=torch.float32, device=cfg.base.device)
        loss = nn.CrossEntropyLoss(weight=weight, **criterion_args)
    elif criterion == 'mean_square_error':
        loss = nn.MSELoss(**criterion_args)
    elif criterion == 'mean_absolute_error':
        loss = nn.L1Loss(**criterion_args)
    elif criterion == 'smooth_L1':
        loss = nn.SmoothL1Loss(**criterion_args)
    elif criterion == 'kappa_loss':
        loss = losses.KappaLoss(**criterion_args)
    elif criterion == 'focal_loss':
        loss = losses.FocalLoss(**criterion_args)
    else:
        raise NotImplementedError('Not implemented loss function.')

    loss_function = losses.WarpedLoss(loss, criterion)
    return loss_function, loss_weight_scheduler

def collect_model_parameters(model: "nn.Module", learning_rate, cfg_solver):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = learning_rate
        weight_decay = cfg_solver.weight_decay
        if key.endswith("norm.weight") or key.endswith("norm.bias"):
            weight_decay = cfg_solver.weight_decay_norm
        elif key.endswith(".bias"):
            # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
            # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
            # hyperparameters are by default exactly the same as for regular
            # weights.
            lr = learning_rate
            weight_decay = weight_decay
        train_p = {
            "params": [value], "lr": lr, "weight_decay": weight_decay,
            "name": key.replace("m_task_heads.detection.", ""),
        }
        # params += [train_p]
        params.append(train_p)
    return params

# define optimizer
def initialize_optimizer(cfg: "ConfigType", model: "nn.Module", params: "list" = None):
    cfg_solver: "SolverConfigType" = cfg.train.solver
    optimizer_strategy = cfg_solver.optimizer
    learning_rate = cfg_solver.learning_rate / cfg_solver.warmup_steps
    weight_decay = cfg_solver.weight_decay
    momentum = cfg_solver.momentum
    nesterov = cfg_solver.nesterov
    
    if params is None:
        params = collect_model_parameters(model, learning_rate, cfg_solver)
    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAM':
        adamw_betas = cfg_solver.adam_betas
        optimizer = torch.optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=adamw_betas,
            # nesterov = nesterov,
        )
    elif optimizer_strategy == 'ADAMW':
        adamw_betas = cfg_solver.adamw_betas
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            betas=adamw_betas,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    return optimizer

