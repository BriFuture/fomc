# Copyright (c) SJTU. All rights reserved.
from torch import nn
from mmcv.runner import ModuleList

from mmrotate.core import rbbox2result
from gdet.registries import MODELS as  ROTATED_DETECTORS
# , build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from .utils import FeatureRefineModule


@ROTATED_DETECTORS.register_module()
class R3Det(RotatedBaseDetector):
    """Rotated Refinement RetinaNet."""

    def __init__(self,
                 num_refine_stages,
                 backbone: "nn.Module",
                 neck=None,
                 bbox_head=None,
                 frm_cfgs=None,
                 refine_heads=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None, **kwargs):
        super(R3Det, self).__init__(init_cfg)
        # self.backbone = build_backbone(backbone)
        self.backbone = backbone
        self.num_refine_stages = num_refine_stages
        if neck is not None:
            self.neck = neck

        self.bbox_head = bbox_head
        self.feat_refine_module = ModuleList()
        self.refine_head = ModuleList(refine_heads)
        for frm_cfg in frm_cfgs:
            self.feat_refine_module.append(FeatureRefineModule(**frm_cfg))

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

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
        outs = self.bbox_head(x)
        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """Forward function."""
        losses = dict()
        x = self.extract_feat(img)

        outs = self.bbox_head(x)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_base = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in loss_base.items():
            losses[f's0.{name}'] = value

        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            lw = self.train_cfg.stage_loss_weights[i]
            feat_refine_mod = self.feat_refine_module[i]
            x_refine = feat_refine_mod(x, rois)
            refine_head = self.refine_head[i]
            outs = refine_head(x_refine)
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            loss_refine = refine_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
            for name, value in loss_refine.items():
                losses[f'sr{i}.{name}'] = ([v * lw for v in value]
                                           if 'loss' in name else value)

            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

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
        outs = self.bbox_head(x)
        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.refine_head[-1].get_bboxes(*bbox_inputs, rois=rois)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels,
                         self.refine_head[-1].num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass
