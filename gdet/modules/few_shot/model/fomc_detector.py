import numpy as np
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from gdet.registries import MODELS as DETECTORS, VISER

from mmdet.core.bbox.transforms import bbox2result
from gdet.structures import *

# from mmdet.models import FocalLoss, SmoothL1Loss
from gdet.structures.configure import ConfigType
from gdet.structures.models import S2ANetHeadLossResult, S2ANetHeadForwardResult, TrainImageMeta, ProcessedImageInfo
from mmrotate.models.detectors.s2anet import S2ANet
from mmrotate.core import rbbox2result

from .fomc_head import fomc_ODMRefinedHead, fomc_RotatedRetinaHead
# from fs.core import utils as cutil
# from fs.events import get_event_storage

    
@DETECTORS.register_module()
class FomcDetector(S2ANet):
    def __init__(self, *args, **kwargs):
        config = kwargs.pop("config", dict())
        super().__init__(*args, **kwargs)
        self.collect = False
        self.few_shot_train = config.get("few_shot_train", False)

    def train(self, mode = True):
        super().train(mode)
        if self.few_shot_train:
            self.backbone.eval()
            self.neck.eval()
        return self
    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    #     sd = state_dict.copy()
    #     for k, v in sd.items():    
    #         if k.startswith("bbox_head.odm_cls_convs.0.conv"):
    #             k = k.replace("bbox_head.odm_cls_convs.0.conv", "bbox_head.encoder.layer1")
    #             state_dict[k] = copy.deepcopy(v)
    #         elif k.startswith("bbox_head.odm_cls_convs.1.conv"):
    #             k = k.replace("bbox_head.odm_cls_convs.1.conv", "bbox_head.encoder.layer2")
    #             state_dict[k] = copy.deepcopy(v)                

    #     return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    # def forward_train(self, img: "torch.Tensor",
    #                   img_metas: "list[TrainImageMeta]",
    #                   gt_bboxes: "list[torch.Tensor]",             gt_labels: "list[torch.Tensor]",
    #                   gt_bboxes_ignore: "list[torch.Tensor]"=None, ids: "list[torch.Tensor]"=None
    #         ):
        
    #     backbone_x = self.backbone(img)
    #     x = self.neck(backbone_x)
    #     outs = self.bbox_head(x) # list(5, list(5, Tensor(3, 15, 132, 132)))
        
    #     losses = self.bbox_head.loss(outs,
    #             gt_bboxes, gt_labels, img_metas, 
    #             gt_bboxes_ignore=gt_bboxes_ignore, ids=ids)
    #     addtion_count = sum(im['addtion'] for im in img_metas)
        
    #     if addtion_count > 0:
    #         ## remove cls loss 
    #         remove_keys = [k for k in losses.keys() if 'cls' in k]
    #         for rk in remove_keys:
    #             losses.pop(rk)
    #     ### visual ###
    #     self.forward_for_visual(outs, img, img_metas, gt_bboxes, gt_labels)

    #     return losses
    # def forward_train(self, img: "torch.Tensor",
    #                   img_metas: "list[TrainImageMeta]",
    #                   gt_bboxes: "list[torch.Tensor]",             gt_labels: "list[torch.Tensor]",
    #                   gt_bboxes_ignore: "list[torch.Tensor]"=None, ids: "list[torch.Tensor]"=None
    #         ):
    #     if self.collect:
    #         return self.forward_collect(img, img_metas, gt_bboxes, gt_labels)

    #     x = self.extract_feat(img)
    #     # x = self.ref_neck(backbone_x)
    #     outs: "S2ANetHeadForwardResult" = self.bbox_head(x) # list(5, list(5, Tensor(3, 15, 132, 132)))
    #     # im_idx = 0
    #     # layer = 0
    #     # idx_list, labels_list = self.bbox_head.test_odm_anchors(outs, img_metas, 
    #     #     gt_bboxes, gt_labels, im_idx=im_idx, )
    #     # sim_vis_cls = VISER.get("SimViser")
    #     # sim_viser = sim_vis_cls(img_root="checkpoints/sim_viser")
    #     # for layer in range(5):
    #     #     idx = idx_list[layer]
    #     #     if idx.numel():
    #     #         hm, norm_hm = sim_viser.plot_hm(img, x[layer], idx[0], im_idx=im_idx, skip_self=False)

    #     # gt_bboxes, gt_labels = self.bbox_head.extract_gt(gt_bboxes, gt_labels)
        
    #     losses: "dict" = self.bbox_head.loss(outs,
    #             gt_bboxes, gt_labels, img_metas, 
    #             gt_bboxes_ignore=gt_bboxes_ignore, ids=ids)
        
    #     # self.forward_for_visual(outs, img, img_metas, gt_bboxes, gt_labels)
    #     return losses
    
    def forward_train(self, img: "torch.Tensor",
            img_metas: "list[TrainImageMeta]",
            gt_bboxes: "list[torch.Tensor]",             gt_labels: "list[torch.Tensor]",
            gt_bboxes_ignore=None):
        """Forward function of S2ANet."""
        losses = dict()
        x = self.extract_feat(img)

        outs = self.fam_head(x)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        # loss_base = self.fam_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        loss_base = self.fam_head.loss(*loss_inputs, gt_bboxes_ignore=None)
        for name, value in loss_base.items():
            losses[f'fam.{name}'] = value

        rois = self.fam_head.refine_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        align_feat = self.align_conv(x, rois)
        outs = self.odm_head(align_feat)
        loss_inputs = dict(gt_bboxes=gt_bboxes, gt_labels=gt_labels, img_metas=img_metas, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
        loss_refine = self.odm_head.loss(outs, **loss_inputs,)
        for name, value in loss_refine.items():
            losses[f'odm.{name}'] = value

        return losses
    
    def forward_collect(self, img: "torch.Tensor",
                      img_metas: "list[TrainImageMeta]",
                      gt_bboxes: "list[torch.Tensor]",             gt_labels: "list[torch.Tensor]",
                      gt_bboxes_ignore: "list[torch.Tensor]"=None, ids: "list[torch.Tensor]"=None
            ):
        backbone_x = self.backbone(img)
        x = self.neck(backbone_x)
        # x = self.ref_neck(backbone_x)
        
        outs: "S2ANetHeadForwardResult" = self.bbox_head(x) # list(5, list(5, Tensor(3, 15, 132, 132)))
        

        im_idx = 0
        idx_list, labels_list = self.bbox_head.collect_odm_anchors(outs, img_metas, gt_bboxes, gt_labels, im_idx=im_idx, )
        # outs = {k: v for k, v in outs.items() if 'feats' in k}
        return outs
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

        bbox_inputs = dict(img_metas=img_meta, cfg=self.test_cfg, rescale=rescale, rois=rois)
        bbox_list = self.odm_head.get_bboxes(outs, **bbox_inputs)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.odm_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
        
    def train_step(self, data: "ProcessedImageInfo", optimizer: "torch.optim.Optimizer"):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        if self.collect:
            feat = self(**data)
            return feat
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs
        
    @torch.no_grad()
    def _momentum_update_weight(self):
        """
        Momentum update on the backbone.
        """
        for param, param_mmt in zip(
            self.bbox_head.parameters(), self.base_bbox_head.parameters()
        ):
            param_mmt.data = param_mmt.data * self.mmt + param.data * (1.0 - self.mmt)
        for param, param_mmt in zip(
            self.neck.parameters(), self.base_neck.parameters()
        ):
            param_mmt.data = param_mmt.data * self.mmt + param.data * (1.0 - self.mmt)
