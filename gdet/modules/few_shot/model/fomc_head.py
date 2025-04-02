import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Callable

import logging

from gdet.registries import MODEL_HEADS as HEADS
from gdet.structures.models import *
from gdet.structures.configure import *
from mmdet.core.anchor import (images_to_levels, )
from mmdet.core.utils import multi_apply, unmap
from mmrotate.core import (obb2hbb, rotated_anchor_inside_flags)
from mmrotate.models.dense_heads.odm_refine_head import ODMRefineHead
from mmrotate.models.dense_heads.rotated_retina_head import RotatedRetinaHead
from bfcommon.fp16_utils import convert_fp32
from gdet.engine.env import ctx_mgr
from . import contrast_loss as ConLossModule # import ContrastiveHead, WeightedSupConLoss
# from .s2anet_head import AlignConv, S2ANetHead, bbox_decode, delta2bbox_rotated

logger = logging.getLogger("gdet.model.mems2anet")

@HEADS.register_module()
class fomc_RotatedRetinaHead(RotatedRetinaHead):
    
    def collect_fam_anchors(self, outs: "S2ANetHeadForwardResult", img_metas: "list[TrainImageMeta]", 
            gt_bboxes:"torch.Tensor", gt_labels:"torch.Tensor", ):
        fam_bbox_preds = convert_fp32(outs["fam_bbox_preds"])
        featmap_sizes = [featmap.size()[-2:] for featmap in fam_bbox_preds]
        device = fam_bbox_preds[0].device
        # [[(16384, K_5), ]]
        anchor_list, valid_flag_list = self.get_init_anchors(featmap_sizes, img_metas, device=device)   
        # Feature Alignment Module
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else self.cls_out_channels -1
        self.train_cfg = self.train_fam_cfg; self.assigner = self.fam_assigner; self.bbox_coder = self.fam_bbox_coder

        cls_reg_targets = self.get_targets(
            anchor_list, valid_flag_list, gt_bboxes, img_metas,
            gt_bboxes_ignore_list=None,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
        )
        if cls_reg_targets is None:
            return None, None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, 
            num_total_pos, num_total_neg, sample_results) = cls_reg_targets
        label = labels_list[0][0]
        mask = label < self.num_classes
        idx, = torch.where(mask)
        return idx, labels_list

@HEADS.register_module()
class ODM_Head(ODMRefineHead):
    def __init__(self, num_classes, in_channels, stacked_convs=2, conv_cfg=None, norm_cfg=None, anchor_generator=None, init_cfg=None, **kwargs):
        cfg: "dict" = kwargs.pop("cfg", dict())
        self.odm_cls_kernel = cfg.get("odm_cls_kernel", 3)
        self.omit_fam_cls   = cfg.get("omit_fam_cls", False)
        self.omit_fam_box   = cfg.get("omit_fam_box", False)
        self.use_mlp_cls    = cfg.get("use_mlp_cls", False)
        super().__init__(num_classes, in_channels, stacked_convs, conv_cfg, norm_cfg, anchor_generator, init_cfg, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""

        super()._init_layers()
        if self.use_mlp_cls:
            self.odm_cls = nn.Linear(
                self.feat_channels,
                self.num_anchors * self.cls_out_channels,
                )
        else:
            self.odm_cls = nn.Conv2d(
                self.feat_channels,
                self.num_anchors * self.cls_out_channels,
                3,
                padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple (torch.Tensor):

                - cls_score (torch.Tensor): Cls scores for a single scale \
                    level the channels number is num_anchors * num_classes.
                - bbox_pred (torch.Tensor): Box energies / deltas for a \
                    single scale level, the channels number is num_anchors * 4.
        """
        or_feat = self.or_conv(x)
        reg_feat = or_feat
        cls_feat = self.or_pool(or_feat)

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        ## bsf.p1
        # bsf.b-e1 ConvContrastiveHead ## train stage only
        if self.use_mlp_cls:
            B, C, H, W = cls_feat.shape
            cls_feat = cls_feat.permute(0, 2, 3, 1)
            cls_feat = cls_feat.reshape(B, -1, C)
            cls_score = self.odm_cls(cls_feat)
            cls_score = cls_score.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        else:
            cls_score = self.odm_cls(cls_feat)
        bbox_pred = self.odm_reg(reg_feat)
        return cls_score, bbox_pred
    
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             rois=None,
             gt_bboxes_ignore=None):
        """Loss function of ODMRefineHead."""
        cls_scores = convert_fp32(cls_scores)
        bbox_preds = convert_fp32(bbox_preds)
        assert rois is not None
        self.bboxes_as_anchors = rois

        cls_scores = convert_fp32(cls_scores)
        bbox_preds = convert_fp32(bbox_preds)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox) 

@HEADS.register_module()
class fomc_ODMRefinedHead(ODM_Head):
    def __init__(self, num_classes, in_channels, stacked_convs=2, conv_cfg=None, norm_cfg=None, anchor_generator=None, init_cfg=None, **kwargs):
        self.base_classes = kwargs.pop("base_classes", 0)
        cfg: "dict" = kwargs.get("cfg", dict())
        super().__init__(num_classes, in_channels, stacked_convs, conv_cfg, norm_cfg, anchor_generator, init_cfg, **kwargs)
        

        self.class_specific_bbox  = cfg.get("class_specific_bbox", False)

        con_cfg = cfg["contrast"].clone()
        self.contrast_weight = con_cfg["contrast_weight"]
        
        if self.contrast_weight > 0:
            con_head_param = cfg["contrast_head"]
            con_head_type = con_head_param.pop("type")
            if con_head_type is None:
                self.encoder = lambda x: x ## no encoder used
            else:
                con_head_cls = getattr(ConLossModule, con_head_type)
                self.encoder = con_head_cls(**con_head_param) 

            con_loss = cfg["contrast_loss"]
            self.iou_threshold = con_loss['iou_threshold']
            self.bg_iou_threshold = con_loss['bg_iou_threshold']
            if self.use_sigmoid_cls:
                num_classes = self.cls_out_channels + 1 
            else:
                num_classes = self.cls_out_channels
            self.criterion = getattr(ConLossModule, con_loss.pop("type"))(cls_channels=num_classes,
                                                                          **con_loss)
            self.criterion.num_classes = self.num_classes
        else:
            self.encoder = lambda x: x ## no encoder used
            logger.warning("Skip contrast encoder because weight is 0!")
        # self.criterion.num_classes = cfg.num_classes
        self.decay_steps: list = con_cfg["decay_steps"]
        self.decay_rate = con_cfg["decay_rate"]
        self.feature_norm = cfg.get("feature_norm")

    def test_odm_anchors(self, outs: "S2ANetHeadForwardResult", img_metas: "list[TrainImageMeta]", 
                      gt_bboxes: "list[torch.Tensor]", gt_labels: "list[torch.Tensor]", im_idx=0, ):
        # Oriented Detection Module targets
        refine_anchors = convert_fp32(outs["refine_anchors"])
        odm_bbox_preds = convert_fp32(outs["odm_bbox_preds"])
        
        device = odm_bbox_preds[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in odm_bbox_preds]
               # Oriented Detection Module targets
        refine_anchors_list, valid_flag_list = self.get_refine_anchors(
            featmap_sizes, refine_anchors, img_metas, device=device)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else self.cls_out_channels-1
        cls_reg_targets = self.get_targets(
            refine_anchors_list,    valid_flag_list,    gt_bboxes,  img_metas,
            gt_bboxes_ignore_list=None,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            )
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg, sample_results) = cls_reg_targets
        idx_list = []
        for l in range(len(labels_list)):
            label = labels_list[l][im_idx]
            mask = label < self.num_classes
            idx, = torch.where(mask)
            idx_list.append(idx)
        print(idx)
        return idx_list, labels_list

    def collect_odm_anchors(self, outs: "S2ANetHeadForwardResult", img_metas: "list[TrainImageMeta]", 
                      gt_bboxes: "list[torch.Tensor]", gt_labels: "list[torch.Tensor]", im_idx=0, ):
        # Oriented Detection Module targets
        refine_anchors = convert_fp32(outs["refine_anchors"])
        odm_bbox_preds = convert_fp32(outs["odm_bbox_preds"])
        
        device = odm_bbox_preds[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in odm_bbox_preds]
               # Oriented Detection Module targets
        refine_anchors_list, valid_flag_list = self.get_refine_anchors(
            featmap_sizes, refine_anchors, img_metas, device=device)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else self.cls_out_channels-1
        # gt_bboxes, gt_labels = self.extract_gt(gt_bboxes, gt_labels)
        cls_reg_targets = self.get_targets(
            refine_anchors_list,    valid_flag_list,    gt_bboxes,  img_metas,
            gt_bboxes_ignore_list=None,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            )
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg, sample_results) = cls_reg_targets
        feats = outs['odm_contrast_feats']
        idx_list = []
        for l in range(len(labels_list)):
            label = labels_list[l][im_idx]
            mask = label < self.num_classes
            idx, = torch.where(mask)
            idx_list.append(idx)
        print(idx)
        return idx_list, labels_list
    
    def extract_gt(self, gt_bboxes: "list[torch.Tensor]", gt_labels: "list[torch.Tensor]",):
        pass
        ### shrink gt_bboxes as ratio less than 3:1
        batch = len(gt_bboxes)
        egt_bboxes = []
        egt_labels = []
        for bboxes, labels in zip(gt_bboxes, gt_labels):
            ratio = bboxes[:, 2] / bboxes[:, 3]
            level2 = (ratio > 3) 
            level3 = (ratio > 6) 
            mask1 = level2
            mask2 = level3
            result_box = []
            result_lbl = []
            for i in range(bboxes.shape[0]):
                if mask2[i]:
                    x, y, w, h, a = bboxes[i]
                    third_w = w / 3

                    dx = (third_w * torch.cos(a)) / 2
                    dy = (third_w * torch.sin(a)) / 2

                    # Segment 1: Shift x to the left
                    segment1 = torch.tensor([x - dx * 2, y - dy * 2, third_w, h, a], device=bboxes.device)
                    # Segment 2 (centered)
                    segment2 = torch.tensor([x, y, third_w, h, a], device=bboxes.device)
                    # Segment 3: Shift x to the right
                    segment3 = torch.tensor([x + dx * 2, y + dy * 2, third_w, h, a], device=bboxes.device)

                    # Append all segments to the result_box list
                    result_box.append(segment1)
                    result_box.append(segment2)
                    result_box.append(segment3)
                    result_lbl.append(labels[i])
                    result_lbl.append(labels[i])
                    result_lbl.append(labels[i])
                elif mask1[i]:  
                    x, y, w, h, a = bboxes[i]
                    half_w = w / 2
                    dx = (half_w / 2) * torch.cos(a)
                    dy = (half_w / 2) * torch.sin(a)

                    # Segment 1: Shift x to the left by dx
                    segment1 = torch.tensor([x - dx, y - dy, half_w, h, a], device=bboxes.device)
                    # Segment 2: Shift x to the right by dx
                    segment2 = torch.tensor([x + dx, y + dy, half_w, h, a], device=bboxes.device)
                    result_box.append(segment1)
                    result_box.append(segment2)
                    result_lbl.append(labels[i])
                    result_lbl.append(labels[i])
                # else:
                result_lbl.append(labels[i])
                result_box.append(bboxes[i])
            result_bboxes = torch.stack(result_box)
            result_labels = torch.stack(result_lbl)
            egt_bboxes.append(result_bboxes)
            egt_labels.append(result_labels)
        return egt_bboxes, egt_labels
    
    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (torch.Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape
                (num_anchors, 5)
            valid_flags (torch.Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (torch.Tensor): Ground truth bboxes of the image,
                shape (num_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (torch.Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_labels (torch.Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple (list[Tensor]):

                - labels_list (list[Tensor]): Labels of each level
                - label_weights_list (list[Tensor]): Label weights of each \
                  level
                - bbox_targets_list (list[Tensor]): BBox targets of each level
                - bbox_weights_list (list[Tensor]): BBox weights of each level
                - num_total_pos (int): Number of positive samples in all images
                - num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = rotated_anchor_inside_flags(
            flat_anchors, valid_flags, img_meta['img_shape'][:2],
            self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        if self.assign_by_circumhbbox is not None:
            gt_bboxes_assign = obb2hbb(gt_bboxes, self.assign_by_circumhbbox)
            assign_result = self.assigner.assign(
                anchors, gt_bboxes_assign, gt_bboxes_ignore,
                None if self.sampling else gt_labels)
        else:
            assign_result = self.assigner.assign(
                anchors, gt_bboxes, gt_bboxes_ignore,
                None if self.sampling else gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        sampling_result.max_overlaps = assign_result.max_overlaps
        s_labels = assign_result.labels[:]
        s_labels[s_labels == -1] = self.num_classes
        sampling_result.labels = s_labels
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)
    
    def get_targets(self,
                    anchor_list: "list[list[torch.Tensor]]",
                    valid_flag_list: "list[list[torch.Tensor]]",
                    gt_bboxes_list: "list[torch.Tensor]",
                    img_metas: "list[TrainImageInfo]",
                    gt_bboxes_ignore_list: "list[torch.Tensor]"=None,
                    gt_labels_list: "list[torch.Tensor]"=None,
                    label_channels: "int"=1,
                    unmap_outputs=True,
                    ):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results: "list[BBoxHeadTarget]" = multi_apply(
            self._get_targets_single,
            concat_anchor_list, concat_valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list,
            gt_labels_list, img_metas,
            label_channels=label_channels, unmap_outputs=unmap_outputs)

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
            pos_inds_list, neg_inds_list) = results[:6] # len(list) is batch


        rest_results = list(results[6:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list        = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list  = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list  = images_to_levels(all_bbox_weights, num_level_anchors)

        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
 
        for i, r in enumerate(rest_results):  # user-added return values
            if type(r[0]) is torch.Tensor:    # check if tensor
                rest_results[i] = images_to_levels(r, num_level_anchors)

        target = BBoxHeadTarget(*res, *rest_results)
        return target

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple (torch.Tensor):

                - cls_score (torch.Tensor): Cls scores for a single scale \
                    level the channels number is num_anchors * num_classes.
                - bbox_pred (torch.Tensor): Box energies / deltas for a \
                    single scale level, the channels number is num_anchors * 4.
        """
        or_feat = self.or_conv(x)
        reg_feat = or_feat
        cls_feat = self.or_pool(or_feat)

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        ## bsf.p1
        odm_cont_feat = cls_feat  ## NOTE.1 previous or_feat, channels is 32 if self.with_or_conv
        # bsf.b-e1 ConvContrastiveHead ## train stage only
        if self.training:
            if self.use_mlp_cls:
                odm_cont_feat = cls_feat.permute(0, 2, 3, 1)
                odm_cont_feat = odm_cont_feat.reshape(odm_cont_feat.shape[0], -1, odm_cont_feat.shape[3])
                odm_cont_feat = self.encoder(odm_cont_feat)
            else:
                odm_cont_feat = self.encoder(cls_feat)
                odm_cont_feat = odm_cont_feat.permute(0, 2, 3, 1)
                odm_cont_feat = odm_cont_feat.reshape(odm_cont_feat.shape[0], -1, odm_cont_feat.shape[3])
        if self.use_mlp_cls:
            B, C, H, W = cls_feat.shape
            cls_feat = cls_feat.permute(0, 2, 3, 1)
            cls_feat = cls_feat.reshape(B, -1, C)
            cls_score = self.odm_cls(cls_feat)
            cls_score = cls_score.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        else:
            cls_score = self.odm_cls(cls_feat)
        bbox_pred = self.odm_reg(reg_feat)
        return cls_score, bbox_pred, odm_cont_feat
        
    def forward(self, feats: tuple):
        map_results = map(self.forward_single, feats)
        v = tuple(map(list, zip(*map_results)))
        ret = ConS2ANetHeadForwardResult()
        ret["cls_scores"] = v[0]
        ret["bbox_preds"] = v[1]
        ret["contrast_feats"] = v[2]
        return ret
    
    def loss(self, outs: "dict",
             gt_bboxes,
             gt_labels,
             img_metas,
             rois=None,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        cls_scores = convert_fp32(outs["cls_scores"])
        bbox_preds = convert_fp32(outs["bbox_preds"])
        assert rois is not None
        self.bboxes_as_anchors = rois
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg, sample_results) = cls_reg_targets

        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        loss_dict = {}
        contrast_loss = None
        if self.able_contrast():
            outs['sample_results'] = sample_results
            odm_cls_feats: "list[torch.Tensor]" = convert_fp32(outs["contrast_feats"])
            cls_feats, labels, ious, scores = self._collect_proposal_features(odm_cls_feats, outs)
            if cls_feats is not None:
                # prediction encoder
                encode_cls_feats = cls_feats

                contrast_loss = self.criterion(encode_cls_feats, labels, ious, scores) 
                contrast_loss = contrast_loss * self.contrast_weight
                if int(ctx_mgr.get_iter()) in self.decay_steps:
                    self.contrast_weight *= self.decay_rate
                    self.decay_steps.remove(ctx_mgr.get_iter())
                    print("========== contrast weight", self.contrast_weight)
                loss_dict['contrast'] = contrast_loss
            
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)

        loss_dict.update(dict(loss_cls=losses_cls, loss_bbox=losses_bbox))
        return loss_dict
    
    def able_contrast(self):
        return self.decay_rate > -1 and self.contrast_weight > 0
        
    def _collect_proposal_features(self, odm_cls_feats: "list[torch.Tensor]", outs: "ConS2ANetHeadForwardResult"):
        odm_sample_results: "SamplingResult"    = outs['sample_results'] 
        odm_cls_scores: "list[torch.Tensor]"    = outs['cls_scores'] 
        B, C, H, W = odm_cls_scores[0].shape
        odm_cls_scores = [cs.reshape(B, C, -1).permute(0, 2, 1) for cs in odm_cls_scores]
        # if self.use_mlp_cls:
        m_cls_feats = torch.cat(odm_cls_feats, dim=1)
        odm_cls_scores = torch.cat(odm_cls_scores, dim=1)

        batch_size = len(odm_sample_results)

        feature_list = []; label_list = []; ious_list = []; score_list = []
        num_samples = 256
        for bid in range(batch_size):
            
            odm_batch_feat = m_cls_feats[bid]
            odm_sample_result = odm_sample_results[bid]
            odm_cls_score = odm_cls_scores[bid]
  
            s_iou = odm_sample_result.max_overlaps
            if not hasattr(odm_sample_result, 'all_inds'):
                mask = (s_iou > self.iou_threshold)
                bg_mask = (s_iou < self.bg_iou_threshold)
                all_inds = torch.where(mask > 0)[0]
                bg_inds = torch.where(bg_mask > 0.01)[0]
                setattr(odm_sample_result, "all_inds", all_inds)
                setattr(odm_sample_result, "bg_inds", bg_inds)
            else:
                all_inds = odm_sample_result.all_inds # 使用所有的 pos sample
                bg_inds = odm_sample_result.bg_inds # 使用所有的 neg sample
            
            if all_inds.any() > 0:
                sr_label = odm_sample_result.labels[all_inds]
                sr_score = odm_cls_score[all_inds]
                sr_iou = s_iou[all_inds]
                feat = odm_batch_feat[all_inds]
                feature_list.append(feat)
                label_list.append(sr_label)
                ious_list.append(sr_iou)
                score_list.append(sr_score.sigmoid())
            
                # selected_indices = torch.randperm(len(bg_inds))[:num_bg_samples]
                # bg_feats = odm_batch_feat[selected_indices]
                # feature_list.append(bg_feats)
                # label_list.append(torch.full((num_bg_samples, ), self.num_classes).to(device=m_cls_feats.device))
                # ious_list.append(torch.full((num_bg_samples, ), 1.).to(device=m_cls_feats.device))
                
                bg_feats = odm_batch_feat[bg_inds]
                mbg_feats = torch.mean(bg_feats, dim=0, keepdim=True)
                feature_list.append(mbg_feats)
                label_list.append(torch.Tensor([self.num_classes]).to(device=m_cls_feats.device))
                ious_list.append(torch.Tensor([1.]).to(device=m_cls_feats.device))
                
        if len(score_list) == 0:
            return None, None, None, None
        cls_feats = torch.cat(feature_list)
        labels = torch.cat(label_list)
        ious = torch.cat(ious_list)
        
        scores = torch.cat(score_list)
        return cls_feats, labels, ious, scores

    def get_bboxes(self,
                   outs: "dict",
                   img_metas,
                   cfg=None,
                   rescale=False,
                   rois=None):
        """Transform network output for a batch into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            img_metas (list[dict]): size / scale info for each image
            cfg (mmcv.Config): test / postprocessing configuration
            rois (list[list[Tensor]]): input rbboxes of each level of
            each image. rois output by former stages and are to be refined
            rescale (bool): if True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (xc, yc, w, h, a) and the
                6-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the class index of the
                corresponding box.
        """
        cls_scores = convert_fp32(outs["cls_scores"])
        bbox_preds = convert_fp32(outs["bbox_preds"])
        num_levels = len(cls_scores)
        assert len(cls_scores) == len(bbox_preds)
        assert rois is not None

        result_list = []

        for img_id, _ in enumerate(img_metas):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                rois[img_id], img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

@HEADS.register_module()
class fomc_ODMRefinedHeadv2(fomc_ODMRefinedHead):
    def __init__(self, num_classes, in_channels, stacked_convs=2, conv_cfg=None, norm_cfg=None, anchor_generator=None, init_cfg=None, **kwargs):
        cfg = kwargs.get("cfg", dict())
        super().__init__(num_classes, in_channels, stacked_convs, conv_cfg, norm_cfg, anchor_generator, init_cfg, **kwargs)
        self.ignore_unused_object_loss = cfg.get("ignore_unused_object_loss", False)

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (torch.Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape
                (num_anchors, 5)
            valid_flags (torch.Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (torch.Tensor): Ground truth bboxes of the image,
                shape (num_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (torch.Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_labels (torch.Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple (list[Tensor]):

                - labels_list (list[Tensor]): Labels of each level
                - label_weights_list (list[Tensor]): Label weights of each \
                  level
                - bbox_targets_list (list[Tensor]): BBox targets of each level
                - bbox_weights_list (list[Tensor]): BBox weights of each level
                - num_total_pos (int): Number of positive samples in all images
                - num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = rotated_anchor_inside_flags(
            flat_anchors, valid_flags, img_meta['img_shape'][:2],
            self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        if self.assign_by_circumhbbox is not None:
            gt_bboxes_assign = obb2hbb(gt_bboxes, self.assign_by_circumhbbox)
            assign_result = self.assigner.assign(
                anchors, gt_bboxes_assign, gt_bboxes_ignore,
                None if self.sampling else gt_labels)
        else:
            assign_result = self.assigner.assign(
                anchors, gt_bboxes, gt_bboxes_ignore,
                None if self.sampling else gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
                    
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        sampling_result.max_overlaps = assign_result.max_overlaps
        s_labels = assign_result.labels
        s_labels[s_labels == -1] = self.num_classes
        sampling_result.labels = s_labels
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)
    
    def get_targets(self,
                    anchor_list: "list[list[torch.Tensor]]",
                    valid_flag_list: "list[list[torch.Tensor]]",
                    gt_bboxes_list: "list[torch.Tensor]",
                    img_metas: "list[TrainImageInfo]",
                    gt_bboxes_ignore_list: "list[torch.Tensor]"=None,
                    gt_labels_list: "list[torch.Tensor]"=None,
                    label_channels: "int"=1,
                    unmap_outputs=True,
                    ):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results: "list[BBoxHeadTarget]" = multi_apply(
            self._get_targets_single,
            concat_anchor_list, concat_valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list,
            gt_labels_list, img_metas,
            label_channels=label_channels, unmap_outputs=unmap_outputs)

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
            pos_inds_list, neg_inds_list) = results[:6] # len(list) is batch


        rest_results = list(results[6:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list        = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list  = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list  = images_to_levels(all_bbox_weights, num_level_anchors)

        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
 
        for i, r in enumerate(rest_results):  # user-added return values
            if type(r[0]) is torch.Tensor:    # check if tensor
                rest_results[i] = images_to_levels(r, num_level_anchors)

        target = BBoxHeadTarget(*res, *rest_results)
        return target    
    
from mmrotate.models.roi_heads.bbox_heads.convfc_rbbox_head import RotatedShared2FCBBoxHead
from mmdet.models.losses.accuracy import accuracy

@HEADS.register_module()
class fomc_RotatedShared2FCBBoxHead(RotatedShared2FCBBoxHead):
    """Shared2FC RBBox head."""
   
    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x  ### 
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        
        return cls_score, bbox_pred, x_cls
    
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        cls_score = convert_fp32(cls_score)
        bbox_pred = convert_fp32(bbox_pred)
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override
                )
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
        
from mmrotate.models.roi_heads.oriented_standard_roi_head import OrientedStandardRoIHead    
from mmrotate.core import rbbox2roi
@HEADS.register_module()    
class fomc_OrientedStandardRoIHead(OrientedStandardRoIHead):
    def __init__(self, bbox_roi_extractor=None, bbox_head=None, shared_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None, version='oc', **kwargs):
        cfg: "dict" = kwargs.get("cfg", dict())
        super().__init__(bbox_roi_extractor, bbox_head, shared_head, train_cfg, test_cfg, pretrained, init_cfg, version)
        con_cfg = cfg["contrast"].clone()
        self.contrast_weight = con_cfg["contrast_weight"]
        self.num_classes = self.bbox_head.num_classes
        if self.contrast_weight > 0:
            con_head_param = cfg["contrast_head"]
            con_head_type = con_head_param.pop("type")
            if con_head_type is None:
                self.encoder = lambda x: x ## no encoder used
            else:
                con_head_cls = getattr(ConLossModule, con_head_type)
                self.encoder = con_head_cls(**con_head_param) 

            con_loss = cfg["contrast_loss"]
            self.iou_threshold = con_loss['iou_threshold']
            self.bg_iou_threshold = con_loss['bg_iou_threshold']
            num_classes = self.num_classes
            self.criterion = getattr(ConLossModule, con_loss.pop("type"))(cls_channels=num_classes,
                                                                          **con_loss)
            self.criterion.num_classes = self.num_classes
        else:
            self.encoder = lambda x: x ## no encoder used
            logger.warning("Skip contrast encoder because weight is 0!")
        self.decay_steps: list = con_cfg["decay_steps"]
        self.decay_rate = con_cfg["decay_rate"]
        self.feature_norm = cfg.get("feature_norm")

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing.

        Args:
            x (list[Tensor]): list of multi-level img features.
            rois (list[Tensors]): list of region of interests.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, x_cls = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, cls_feats=x_cls)
        return bbox_results    
    
    def able_contrast(self):
        return self.decay_rate > -1 and self.contrast_weight > 0
 
    
    def _collect_proposal_features(self, con_cls_feats: "list[torch.Tensor]", outs: "ConS2ANetHeadForwardResult"):
        sample_results: "SamplingResult"    = outs['sample_results'] 
        # if self.use_mlp_cls:

        batch_size = len(sample_results)
        m_cls_feats = con_cls_feats.reshape(batch_size, -1, con_cls_feats.shape[-1])

        feature_list = []; label_list = []; ious_list = []; score_list = []
        pos = False
        for bid in range(batch_size):
            con_batch_feat = m_cls_feats[bid]
            ### encode into embedding space
            con_batch_feat = self.encoder(con_batch_feat)
            

            con_sample_result = sample_results[bid]
            pos_inds = con_sample_result.pos_inds
            bg_inds  = con_sample_result.neg_inds
            all_inds = torch.cat([pos_inds, bg_inds])
            s_iou = con_sample_result.max_overlaps[all_inds]
            all_labels = con_sample_result.labels[all_inds]

            pos_mask = (s_iou > self.iou_threshold)
            feat = con_batch_feat[pos_mask]
            sr_label = all_labels[pos_mask]
            sr_iou = s_iou[pos_mask]

            feature_list.append(feat)
            label_list.append(sr_label)
            ious_list.append(sr_iou)

            bg_mask = (s_iou < self.bg_iou_threshold)
            bg_feats = con_batch_feat[bg_mask]
            mbg_feats = torch.mean(bg_feats, dim=0, keepdim=True)
            
            feature_list.append(mbg_feats)
            label_list.append(torch.Tensor([self.num_classes]).to(device=m_cls_feats.device))
            ious_list.append(torch.Tensor([1.]).to(device=m_cls_feats.device))

        cls_feats = torch.cat(feature_list)
        
        labels = torch.cat(label_list)
        ious = torch.cat(ious_list)
        
        return cls_feats, labels, ious
        
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. Always
                set to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        losses = dict()
        if self.with_bbox:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                        (0, gt_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = \
                        gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]
                ###
                # inds = torch.cat([sampling_result.pos_inds, sampling_result.neg_inds])
                sampling_result.max_overlaps = assign_result.max_overlaps
                s_labels = assign_result.labels[:]
                s_labels[s_labels == -1] = self.num_classes
                sampling_result.labels = s_labels
                sampling_results.append(sampling_result)

            # bbox head forward and loss
        
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])
            ## contrast
            if self.able_contrast():
                con_cls_feats = convert_fp32(bbox_results['cls_feats'])
                bbox_results['sample_results'] = sampling_results
                cls_feats, labels, ious = self._collect_proposal_features(con_cls_feats, bbox_results)
                if cls_feats is not None:
                    # prediction encoder
                    encode_cls_feats = cls_feats

                    contrast_loss = self.criterion(encode_cls_feats, labels, ious, None) 
                    contrast_loss = contrast_loss * self.contrast_weight
                    if int(ctx_mgr.get_iter()) in self.decay_steps:
                        self.contrast_weight *= self.decay_rate
                        self.decay_steps.remove(ctx_mgr.get_iter())
                        print("========== contrast weight", self.contrast_weight)
                    losses['contrast'] = contrast_loss
        return losses    