# Copyright (c) OpenMMLab. All rights reserved.
from gdet.registries import MODELS as ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector


@ROTATED_DETECTORS.register_module()
class RotatedRetinaNet(RotatedSingleStageDetector):
    """Implementation of Rotated `RetinaNet.`__

    __ https://arxiv.org/abs/1708.02002
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedRetinaNet,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained, init_cfg)
