# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn
from .base import RotatedBaseDetector
from .utils import AlignConvModule
from ..dense_heads.rotated_retina_head import RotatedRetinaHead
from mmrotate.core import rbbox2result
from gdet.registries import MODELS as ROTATED_DETECTORS

@ROTATED_DETECTORS.register_module()
class S2ANet(RotatedBaseDetector):
    """Implementation of `Align Deep Features for Oriented Object Detection.`__

    __ https://ieeexplore.ieee.org/document/9377550
    """

    def __init__(self,
                 backbone: "nn.Module",
                 neck=None,
                 fam_head=None,
                 align_conv=None,
                 odm_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        config = kwargs.pop("config", dict())
        super(S2ANet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.fam_head: "RotatedRetinaHead" = fam_head
        self.align_conv: "AlignConvModule" = align_conv
        self.odm_head = odm_head
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.freeze_rpn = config.get("freeze_rpn", False)
        self.freeze_backbone = config.get("freeze_backbone", False)
        self.freeze_neck = config.get("freeze_neck", False)

    def init(self):
        pass
    
    def train(self, mode = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        if self.freeze_neck:
            self.neck.eval()
        return self
    
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.fam_head(x)
        rois = self.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)
        outs = self.odm_head(align_feat)

        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """Forward function of S2ANet."""
        losses = dict()
        x = self.extract_feat(img)

        outs = self.fam_head(x)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_base = self.fam_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in loss_base.items():
            losses[f'fam.{name}'] = value

        rois = self.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)
        outs = self.odm_head(align_feat)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_refine = self.odm_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
        for name, value in loss_refine.items():
            losses[f'odm.{name}'] = value

        return losses

    def simple_test(self, img, img_meta, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.fam_head(x)
        rois = self.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)
        outs = self.odm_head(align_feat)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.odm_head.get_bboxes(*bbox_inputs, rois=rois)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.odm_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
