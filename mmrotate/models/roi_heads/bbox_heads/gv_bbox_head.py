# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.utils import to_2tuple
from mmdet.core import multi_apply
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer

from mmrotate.core import (build_bbox_coder, hbb2obb, multiclass_nms_rotated,
                           obb2xyxy)
from ...builder import ROTATED_HEADS, build_loss


@ROTATED_HEADS.register_module()
class GVBBoxHead(BaseModule):
    """Gliding Vertex's RoI bbox head.

    Args:
        with_avg_pool (bool, optional): If True, use ``avg_pool``.
        num_shared_fcs (int, optional): number of ``shared_fcs``.
        roi_feat_size (int, optional): Size of RoI features.
        in_channels (int, optional): Input channels.
        fc_out_channels (int, optional): output channels of fc.
        num_classes (int, optional): Number of classes.
        ratio_thr (float, optional): threshold of ratio.
        bbox_coder (dict, optional): Config of bbox coder.
        fix_coder (dict, optional): Config of fix coder.
        ratio_coder (dict, optional): Config of ratio coder.
        reg_class_agnostic (bool, optional): If True, regression branch are
            class agnostic.
        reg_decoded_bbox (bool, optional): If True, regression branch use
            decoded bbox to compute loss.
        reg_predictor_cfg (dict, optional): Config of regression predictor.
        cls_predictor_cfg (dict, optional): Config of classification predictor.
        fix_predictor_cfg (dict, optional): Config of fix predictor.
        ratio_predictor_cfg (dict, optional): Config of ratio predictor.
        loss_cls (dict, optional): Config of classification loss.
        loss_bbox (dict, optional): Config of regression loss.
        loss_fix (dict, optional): Config of fix loss.
        loss_ratio (dict, optional): Config of ratio loss.
        version (str, optional): Angle representations. Defaults to 'oc'.
        init_cfg (dict, optional): Config of initialization.
    """

    def __init__(
            self,
            with_avg_pool=False,
            num_shared_fcs=2,
            roi_feat_size=7,
            in_channels=256,
            fc_out_channels=1024,
            num_classes=80,
            ratio_thr=0.8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                clip_border=True,
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            fix_coder=dict(type='GVFixCoder'),
            ratio_coder=dict(type='GVRatioCoder'),
            reg_class_agnostic=False,
            reg_decoded_bbox=False,
            reg_predictor_cfg=dict(type='Linear'),
            cls_predictor_cfg=dict(type='Linear'),
            fix_predictor_cfg=dict(type='Linear'),
            ratio_predictor_cfg=dict(type='Linear'),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            loss_fix=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            loss_ratio=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            version='oc',
            init_cfg=None):
        super(GVBBoxHead, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        self.num_shared_fcs = num_shared_fcs
        self.roi_feat_size = to_2tuple(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.ratio_thr = ratio_thr
        self.reg_decoded_bbox = reg_decoded_bbox
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg
        self.fix_predictor_cfg = fix_predictor_cfg
        self.ratio_predictor_cfg = ratio_predictor_cfg
        self.fp16_enabled = False

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.fix_coder = build_bbox_coder(fix_coder)
        self.ratio_coder = build_bbox_coder(ratio_coder)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_fix = build_loss(loss_fix)
        self.loss_ratio = build_loss(loss_ratio)
        self.version = version

        self.relu = nn.ReLU(inplace=True)
        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area

        self.shared_fcs = nn.ModuleList()
        for i in range(self.num_shared_fcs):
            fc_in_channels = (in_channels if i == 0 else self.fc_out_channels)
            self.shared_fcs.append(
                nn.Linear(fc_in_channels, self.fc_out_channels))

        last_dim = in_channels if self.num_shared_fcs == 0 \
            else self.fc_out_channels

        self.fc_cls = build_linear_layer(
            self.cls_predictor_cfg,
            in_features=last_dim,
            out_features=num_classes + 1)
        out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
        self.fc_reg = build_linear_layer(
            self.reg_predictor_cfg,
            in_features=last_dim,
            out_features=out_dim_reg)
        out_dim_fix = 4 if reg_class_agnostic else 4 * num_classes
        self.fc_fix = build_linear_layer(
            self.fix_predictor_cfg,
            in_features=last_dim,
            out_features=out_dim_fix)
        out_dim_ratio = 1 if reg_class_agnostic else num_classes
        self.fc_ratio = build_linear_layer(
            self.ratio_predictor_cfg,
            in_features=last_dim,
            out_features=out_dim_ratio)

        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            self.init_cfg += [
                dict(type='Normal', std=0.01, override=dict(name='fc_cls'))
            ]
            self.init_cfg += [
                dict(type='Normal', std=0.001, override=dict(name='fc_reg'))
            ]
            self.init_cfg += [
                dict(type='Normal', std=0.001, override=dict(name='fc_fix'))
            ]
            self.init_cfg += [
                dict(type='Normal', std=0.001, override=dict(name='fc_ratio'))
            ]

    @property
    def custom_cls_channels(self):
        """The custom cls channels."""
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    @property
    def custom_activation(self):
        """The custom activation."""
        return getattr(self.loss_cls, 'custom_activation', False)

    @property
    def custom_accuracy(self):
        """The custom accuracy."""
        return getattr(self.loss_cls, 'custom_accuracy', False)

    # @auto_fp16()
    def forward(self, x):
        """Forward function."""
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        for fc in self.shared_fcs:
            x = self.relu(fc(x))

        cls_score = self.fc_cls(x)
        bbox_pred = self.fc_reg(x)
        fix_pred = torch.sigmoid(self.fc_fix(x))
        ratio_pred = torch.sigmoid(self.fc_ratio(x))
        return cls_score, bbox_pred, fix_pred, ratio_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (torch.Tensor): Contains all the positive boxes,
                has shape (num_pos, 5), the last dimension 5
                represents [cx, cy, w, h, a].
            neg_bboxes (torch.Tensor): Contains all the negative boxes,
                has shape (num_neg, 5), the last dimension 5
                represents [cx, cy, w, h, a].
            pos_gt_bboxes (torch.Tensor): Contains all the gt_boxes,
                has shape (num_gt, 5), the last dimension 5
                represents [cx, cy, w, h, a].
            pos_gt_labels (torch.Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(torch.Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
                - bbox_weights(torch.Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 5).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        fix_targets = pos_bboxes.new_zeros(num_samples, 4)
        fix_weights = pos_bboxes.new_zeros(num_samples, 4)
        ratio_targets = pos_bboxes.new_zeros(num_samples, 1)
        ratio_weights = pos_bboxes.new_zeros(num_samples, 1)

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, obb2xyxy(pos_gt_bboxes, self.version))
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1

            pos_fix_targets = self.fix_coder.encode(pos_gt_bboxes)
            fix_targets[:num_pos, :] = pos_fix_targets
            fix_weights[:num_pos, :] = 1

            pos_ratio_targets = self.ratio_coder.encode(pos_gt_bboxes)
            ratio_targets[:num_pos, :] = pos_ratio_targets
            ratio_weights[:num_pos, :] = 1

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, fix_targets,
                fix_weights, ratio_targets, ratio_weights)

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 5),  the last dimension 5
                represents [cx, cy, w, h, a].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 5) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 5), the last dimension 5 represents
                  [cx, cy, w, h, a].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        (labels, label_weights, bbox_targets, bbox_weights, fix_targets,
         fix_weights, ratio_targets, ratio_weights) = multi_apply(
             self._get_target_single,
             pos_bboxes_list,
             neg_bboxes_list,
             pos_gt_bboxes_list,
             pos_gt_labels_list,
             cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            fix_targets = torch.cat(fix_targets, 0)
            fix_weights = torch.cat(fix_weights, 0)
            ratio_targets = torch.cat(ratio_targets, 0)
            ratio_weights = torch.cat(ratio_weights, 0)
        return (labels, label_weights, bbox_targets, bbox_weights, fix_targets,
                fix_weights, ratio_targets, ratio_weights)

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'fix_pred', 'ratio_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             fix_pred,
             ratio_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             fix_targets,
             fix_weights,
             ratio_targets,
             ratio_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            fix_pred (Tensor, optional): Shape (num_boxes, num_classes * 4).
            ratio_pred (Tensor, optional): Shape (num_boxes, num_classes * 1).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 5 represents [cx, cy, w, h].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
            fix_targets (torch.Tensor): Fix target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 5 represents [a1, a2, a3, a4].
            fix_weights (list[tensor],Tensor): Fix weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
            ratio_targets (torch.Tensor): Ratio target for all
                  proposals, has shape (num_proposals, 1).
            ratio_weights (list[tensor],Tensor): Ratio weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 1) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 1).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
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
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                    pos_fix_pred = fix_pred.view(fix_pred.size(0),
                                                 4)[pos_inds.type(torch.bool)]
                    pos_ratio_pred = ratio_pred.view(
                        ratio_pred.size(0), 1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                    pos_fix_pred = fix_pred.view(
                        fix_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                    pos_ratio_pred = ratio_pred.view(
                        ratio_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
                losses['loss_fix'] = self.loss_bbox(
                    pos_fix_pred,
                    fix_targets[pos_inds.type(torch.bool)],
                    fix_weights[pos_inds.type(torch.bool)],
                    avg_factor=fix_targets.size(0),
                    reduction_override=reduction_override)
                losses['loss_ratio'] = self.loss_bbox(
                    pos_ratio_pred,
                    ratio_targets[pos_inds.type(torch.bool)],
                    ratio_weights[pos_inds.type(torch.bool)],
                    avg_factor=ratio_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
                losses['loss_fix'] = fix_pred[pos_inds].sum()
                losses['loss_ratio'] = ratio_pred[pos_inds].sum()

        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'fix_pred', 'ratio_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   fix_pred,
                   ratio_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 6) and last
                dimension 6 represent (cx, cy, w, h, a, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        rbboxes = self.fix_coder.decode(bboxes, fix_pred)

        bboxes = bboxes.view(*ratio_pred.size(), 4)
        rbboxes = rbboxes.view(*ratio_pred.size(), 5)
        try:
            rbboxes[ratio_pred > self.ratio_thr] = \
                hbb2obb(bboxes[ratio_pred > self.ratio_thr], self.version)
        except:  # noqa: E722
            pass

        if rescale and rbboxes.size(0) > 0:
            scale_factor = rbboxes.new_tensor(scale_factor)
            rbboxes[..., :4] = rbboxes[..., :4] / scale_factor
            rbboxes = rbboxes.view(rbboxes.size(0), -1)

        if cfg is None:
            return rbboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms_rotated(
                rbboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (torch.Tensor): Shape (n*bs, 5), where n is image number per
                GPU, and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (torch.Tensor): Shape (n*bs, ).
            bbox_preds (torch.Tensor): Shape (n*bs, 5) or (n*bs, 5*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i, _ in enumerate(img_metas):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (torch.Tensor): shape (n, 4) or (n, 5)
            label (torch.Tensor): shape (n, )
            bbox_pred (torch.Tensor): shape (n, 5*(#class)) or (n, 5)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 5

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=img_meta['img_shape'])
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
