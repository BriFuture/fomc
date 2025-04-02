import tqdm
import os, os.path as osp
import numpy as np
import torch
import torch.nn as nn
import logging

from .exp_mmdet import ExpMMDetTrain, EXPS

logger = logging.getLogger("gdet.engine.fs")

@EXPS.register_module()
class ExpFewshotTrain(ExpMMDetTrain):
   val_metric_key = "nAP"

    