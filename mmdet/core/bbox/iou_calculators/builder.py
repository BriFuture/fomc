# Copyright (c) OpenMMLab. All rights reserved.
from gdet.registries import Registry
from gdet.utils.builder_utils import build_from_cfg

IOU_CALCULATORS = Registry('IoU calculator')


def build_iou_calculator(cfg, default_args=None):
    """Builder of IoU calculator."""
    return build_from_cfg(cfg, IOU_CALCULATORS, default_args)
