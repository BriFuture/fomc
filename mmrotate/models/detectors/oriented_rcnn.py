# Copyright (c) OpenMMLab. All rights reserved.
import torch

from gdet.registries import MODELS as ROTATED_DETECTORS
from .two_stage import RotatedTwoStageDetector


@ROTATED_DETECTORS.register_module()
class OrientedRCNN(RotatedTwoStageDetector):
    """Implementation of `Oriented R-CNN for Object Detection.`__

    __ https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf  # noqa: E501, E261.
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None, **kwargs):
        config = kwargs.pop("config", dict())
        super(OrientedRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.freeze_rpn = config.get("freeze_rpn", False)
        self.freeze_backbone = config.get("freeze_backbone", False)
        self.freeze_neck = config.get("freeze_neck", False)
        if self.freeze_neck :
            self.neck.requires_grad_(False)
        if self.freeze_rpn :
            self.rpn_head.requires_grad_(False)

    def train(self, mode = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        # if self.freeze_neck:
        #     self.neck.eval()
        # if self.freeze_rpn:
        #     self.rpn_head.eval()
        return self
    
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmrotate/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 6).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs
