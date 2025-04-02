# Copyright (c) OpenMMLab. All rights reserved.
from bfcommon.registry import build_from_cfg

def build_with_registry(reg, cfg):
    """pass all configs except type
    """
    cls_type = cfg.pop("type")
    reg_cls = reg.get(cls_type)
    assert reg_cls is not None, cls_type
    obj = reg_cls(**cfg)
    return obj