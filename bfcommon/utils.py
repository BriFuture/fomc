import random
import torch
import torch.backends.cudnn
import numpy as np
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    print(f"Random seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

import datetime
def convert_doy_to_date(year: int, doy: int):
    d = datetime.datetime(year, 1, 1)
    d += datetime.timedelta(days = doy - 1)
    return (year, d.month, d.day)

def convert_date_to_doy(year: "int", month: "int", day: "int") -> "int":
    dt = datetime.datetime(year= year, month=month, day=day)
    base_dt = datetime.datetime(year= year, month=1, day=1)
    doy = (dt - base_dt).days + 1
    return doy

def convert_doy_as_float(year: int, doy: int):
    return year + doy / 400

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn as nn
from PIL import Image
import os.path as osp, os
import torch

import numpy as np
import torch

import logging
logger = logging.getLogger("bf.core.utils")
        
def sim_matrix(a: "torch.Tensor", b: "torch.Tensor", eps=1e-8):
    """a : T[N, d] b: T[M, d]
    added eps for numerical stability
    """
    a_n = a.norm(dim=-1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_n = b.norm(dim=1)[:, None]
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def convert_tensor_as_npint8(xim: "torch.Tensor"):
    """xim: T[3, h, w]"""
    assert xim.dim() == 3
    arr = xim.detach().cpu().permute(1, 2, 0).numpy()
    m1 = arr.max()
    m2 = arr.min()
    diff = m1 - m2
    if diff > 1e-3:
        arr = (arr - m2) / (m1 - m2) * 255
    else:
        arr *= 255
    
    arr = arr.astype(np.uint8)
    return arr, m1, m2

def get_pred_from_cls_score(cls_score: "torch.Tensor"):
    cs = cls_score.sigmoid()
    max_scores, _ = cs.max(dim=-1)
    _, topk_inds = max_scores.topk(1000)
    scores = cs[topk_inds, :]
    pred_cls = scores.argmax(-1)
