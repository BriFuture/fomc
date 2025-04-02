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
# from mmdet.core import (anchor_target, images_to_levels, multi_apply, SamplingResult)
from mmrotate.models.dense_heads.odm_refine_head import ODMRefineHead
from mmrotate.models.dense_heads.rotated_retina_head import RotatedRetinaHead
from bfcommon.fp16_utils import convert_fp32
# from fs.events import get_event_storage
from . import contrast_loss as ConLossModule # import ContrastiveHead, WeightedSupConLoss
# from .s2anet_head import AlignConv, S2ANetHead, bbox_decode, delta2bbox_rotated
#    
logger = logging.getLogger("gdet.model.mems2anet")

@HEADS.register_module()
class Contrast_S2ANetHead(Collect_S2ANetHead):
    def __init__(self, num_classes, in_channels, cfg: "HeadConfigType", base_classes: "int"=12, **kwargs):
        super().__init__(num_classes, in_channels, cfg,  **kwargs)
        self.base_classes = base_classes
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
            self.criterion = getattr(ConLossModule, con_loss.pop("type"))(cls_channels=self.cls_out_channels,
                                                                          **con_loss)
            self.criterion.num_classes = self.num_classes
        else:
            self.encoder = lambda x: x ## no encoder used
            logger.warning("Skip contrast encoder because weight is 0!")
        # self.criterion.num_classes = cfg.num_classes
        self.decay_steps: list = con_cfg["decay_steps"]
        self.decay_rate = con_cfg["decay_rate"]
        self.feature_norm = cfg.get("feature_norm")
        # self.mem_bank = MemoryBank(bank_size=102400, dim=self.feat_channels, mmt=0.99)
        # self.omit_fam_cls = True

    def get_targets(self,
                    anchor_list: "list[list[torch.Tensor]]",
                    valid_flag_list: "list[list[torch.Tensor]]",
                    gt_bboxes_list: "list[torch.Tensor]",
                    img_metas: "list[TrainImageInfo]",
                    gt_bboxes_ignore_list: "list[torch.Tensor]"=None,
                    gt_labels_list: "list[torch.Tensor]"=None,
                    label_channels: "int"=1,
                    unmap_outputs=True,
                    ids_list: "list[torch.Tensor]"=None):
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
        if ids_list is None:
            results: "list[BBoxHeadTarget]" = multi_apply(
                self._get_targets_single,
                concat_anchor_list, concat_valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list,
                gt_labels_list, img_metas,
                label_channels=label_channels, unmap_outputs=unmap_outputs)
        else:
            results: "list[BBoxHeadTarget]" = multi_apply(
                self._get_targets_single_with_id,
                concat_anchor_list, concat_valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list,
                gt_labels_list, img_metas, ids_list,
                label_channels=label_channels, unmap_outputs=unmap_outputs, )
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
    def forward_odm_part(self, or_feat) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        ### 设置 regression feature 并输出 contrast feat
        odm_reg_feat = or_feat
        for odm_reg_conv in self.odm_reg_convs:
            odm_reg_feat = odm_reg_conv(odm_reg_feat)  # BFP_SHAPE T[B 16, C 256, H, W]
        odm_bbox_pred = self.odm_reg(odm_reg_feat) # BFP_SHAPE T[B, 5, H, W]
        ##  测试 or_feat 的特征，NOTE 可以使用 non-max 函数对class特征进行修正
        # fam_contrast_feat = or_feat ## 获取旋转前的特征
        # if self.use_mlp_cls:
        #     fam_contrast_feat = fam_contrast_feat.permute(0, 2, 3, 1)
        #     fam_contrast_feat = fam_contrast_feat.reshape(fam_contrast_feat.shape[0], -1, self.feat_channels)
        # class branch        
        if self.with_orconv:
            odm_cls_feat = self.or_pool(or_feat) # or_pool 提取最大的特征
        else:
            odm_cls_feat = or_feat
        
        for oi, odm_cls_conv in enumerate(self.odm_cls_convs):
            odm_cls_feat = odm_cls_conv(odm_cls_feat)
        
        ## bsf.p1
        odm_cont_feat = odm_cls_feat  ## NOTE.1 previous or_feat, channels is 32 if self.with_or_conv
        # bsf.b-e1 ConvContrastiveHead ## train stage only
        if self.training:
            odm_cont_feat = self.encoder(odm_cls_feat)
            if self.use_mlp_cls:
                odm_cont_feat = odm_cont_feat.permute(0, 2, 3, 1)
                odm_cont_feat = odm_cont_feat.reshape(odm_cont_feat.shape[0], -1, odm_cont_feat.shape[3])
        ## permute tensor
        if self.use_mlp_cls:
            odm_cls_feat = odm_cls_feat.permute(0, 2, 3, 1)
            odm_cls_feat = odm_cls_feat.reshape(odm_cls_feat.shape[0], -1, self.feat_channels)

        if not self.cosine_on:
            odm_cls_score = self.odm_cls(odm_cls_feat) # BFP_SHAPE [B 16, K 10, H, W]
        else:
            odm_cls_score = self.forward_cosine_weight(self.odm_cls, odm_cls_feat)
        return odm_cls_score, odm_bbox_pred, odm_cont_feat

    def forward_single(self, x, stride): # all returns are features
        ## pass fam
        fam_cls_score, fam_bbox_pred, fam_contrast_feat = self.forward_fam_part(x)   # BFP_SHAPE T(B, 15, H, W)
        
        num_level = self.anchor_strides.index(stride) # level idx
        featmap_size = fam_bbox_pred.shape[-2:]
        device = fam_bbox_pred.device
        init_anchors = self.anchor_generators.single_grid_anchors(featmap_size, num_level, device=device)
        
        refine_anchor = bbox_decode( fam_bbox_pred.detach(), init_anchors, self.target_means, self.target_stds)

        align_feat = self.align_conv(x, refine_anchor.clone(), stride) # ARN BFP_SHAPE T(B, 256, H, W)

        or_feat = self.or_conv(align_feat)  # BFP_SHAPE T(B, 256, H, W) # ARF
        odm_cls_score, odm_bbox_pred, odm_contrast_feat = self.forward_odm_part(or_feat)
        
        return (fam_cls_score, fam_bbox_pred, refine_anchor,
                odm_cls_score, odm_bbox_pred, 
                odm_contrast_feat, 
        )
    
    def forward(self, feats: tuple):
        # return multi_apply(self.forward_single, feats, self.anchor_strides)
        map_results = map(self.forward_single, feats, self.anchor_strides)
        v = tuple(map(list, zip(*map_results)))
        ret = ConS2ANetHeadForwardResult()
        ret["fam_cls_scores"] = v[0]
        ret["fam_bbox_preds"] = v[1]
        ret["refine_anchors"] = v[2]
        ret["odm_cls_scores"] = v[3]
        ret["odm_bbox_preds"] = v[4]
        ret["odm_contrast_feats"] = v[5]
        return ret
    # end of forward
    def loss_part_odm(self, outs, img_metas: "list[TrainImageMeta]", 
                      gt_bboxes: "list[torch.Tensor]", gt_labels: "list[torch.Tensor]", gt_bboxes_ignore: "list[torch.Tensor]", 
                      ids_list: "list[torch.Tensor]"=None):
        # Oriented Detection Module targets
        refine_anchors = convert_fp32(outs["refine_anchors"])
        odm_cls_scores = convert_fp32(outs["odm_cls_scores"]) # SHAPE [K_5 level, T(3 batch, 15 class, 144 width, 144 height)]
        odm_bbox_preds = convert_fp32(outs["odm_bbox_preds"])
        
        device = odm_cls_scores[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in odm_bbox_preds]
               # Oriented Detection Module targets
        refine_anchors_list, valid_flag_list = self.get_refine_anchors(
            featmap_sizes, refine_anchors, img_metas, device=device)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in refine_anchors_list[0]]

        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(refine_anchors_list)):
            concat_anchor_list.append(torch.cat(refine_anchors_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)  # [K_5 level, T(3, 24336, K_5) ]


        label_channels = self.cls_out_channels if self.use_sigmoid_cls else self.cls_out_channels-1
        self.train_cfg = self.train_odm_cfg; self.assigner = self.odm_assigner; self.bbox_coder = self.odm_bbox_coder
        cls_reg_targets = self.get_targets(
            refine_anchors_list,    valid_flag_list,    gt_bboxes,  img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            ids_list=ids_list
            )
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg, sample_results) = cls_reg_targets

        contrast_loss = None
        outs['odm_sample_results'] = sample_results
        if self.able_contrast():
            odm_cls_feats: "list[torch.Tensor]" = convert_fp32(outs["odm_contrast_feats"])
            cls_feats, labels, ious, scores = self._collect_proposal_features(odm_cls_feats, outs)
            # prediction encoder
            encode_cls_feats = cls_feats

            contrast_loss = self.criterion(encode_cls_feats, labels, ious, scores) 
            contrast_loss = contrast_loss * self.contrast_weight

        num_total_samples = (num_total_pos + num_total_neg if self.sampling else num_total_pos)

        losses_odm_cls, losses_odm_bbox = multi_apply(
            self.loss_odm_single,
            odm_cls_scores,     odm_bbox_preds,         all_anchor_list,
            labels_list,        label_weights_list,     bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=self.train_odm_cfg
        )
        return OdmLossResult(losses_odm_cls, losses_odm_bbox, contrast_loss)

    def loss(self,
             outs: "S2ANetHeadForwardResult", 
             gt_bboxes: "list[torch.Tensor]", gt_labels: "list[torch.Tensor]", 
             img_metas: "TrainImageMeta",
             gt_bboxes_ignore: "list[torch.Tensor]"=None, ids: "list[torch.Tensor]"=None):
        
        ## 如果 omit fam box 和 cls 的 loss，那么不需要经过 fam part
        if not (self.omit_fam_cls and self.omit_fam_box):
            fam_ret = self.loss_part_fam(outs, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, ids_list=ids)
            if fam_ret is None:
                return None
            losses_fam_cls, losses_fam_bbox, fam_sample_results = fam_ret
            outs['fam_sample_results'] = fam_sample_results
        else:
            losses_fam_cls, losses_fam_bbox = None, None
        ## 
        odm_ret = self.loss_part_odm(outs, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, ids_list=ids)
        if odm_ret is None:
            return None
        losses_odm_cls, losses_odm_bbox, constrast_loss = odm_ret
        loss_dict = dict()
        if not self.omit_fam_cls:
            loss_dict['loss_fam_cls']  = losses_fam_cls
        if not self.omit_fam_box:
            loss_dict['loss_fam_bbox'] = losses_fam_bbox
        # outs['fam_sample_results']  = fam_sample_results
        loss_dict['loss_odm_cls']   = losses_odm_cls
        loss_dict['loss_odm_bbox']  = losses_odm_bbox
        if constrast_loss is not None:
            loss_dict['loss_contrastive']  = constrast_loss

        storage = get_event_storage()

        if self.able_contrast():
            if int(storage.iter) in self.decay_steps:
                self.contrast_weight *= self.decay_rate
                self.decay_steps.remove(storage.iter)
                print("========== contrast weight", self.contrast_weight)
            
        return loss_dict
    
    def able_contrast(self):
        return self.decay_rate > -1 and self.contrast_weight > 0
    
    def _collect_proposal_features(self, odm_cls_feats: "list[torch.Tensor]", outs: "ConS2ANetHeadForwardResult"):
        odm_sample_results: "SamplingResult"    = outs['odm_sample_results'] 
        odm_cls_scores: "list[torch.Tensor]"    = outs['odm_cls_scores'] 
        if self.use_mlp_cls:
            m_cls_feats = torch.cat(odm_cls_feats, dim=1)
            odm_cls_scores = torch.cat(odm_cls_scores, dim=1)

        batch_size = len(odm_sample_results)

        feature_list = []; label_list = []; ious_list = []; score_list = []
            
        for bid in range(batch_size):
            
            odm_batch_feat = m_cls_feats[bid]
            odm_sample_result = odm_sample_results[bid]
            odm_cls_score = odm_cls_scores[bid]
  
            s_iou = odm_sample_result.max_overlaps
            if not hasattr(odm_sample_result, 'all_inds'):
                mask = (s_iou > self.iou_threshold)
                bg_mask = (s_iou < self.bg_iou_threshold)
                all_inds = torch.where(mask > 0)[0]
                bg_inds = torch.where(bg_mask > 0)[0]
                setattr(odm_sample_result, "all_inds", all_inds)
                setattr(odm_sample_result, "bg_inds", bg_inds)
            else:
                all_inds = odm_sample_result.all_inds # 使用所有的 pos sample
                bg_inds = odm_sample_result.bg_inds # 使用所有的 pos sample
            
            if all_inds.any() > 0:
                sr_label = odm_sample_result.labels[all_inds]
                sr_score = odm_cls_score[all_inds]
                sr_iou = s_iou[all_inds]
                feat = odm_batch_feat[all_inds]
                feature_list.append(feat)
                label_list.append(sr_label)
                ious_list.append(sr_iou)
                score_list.append(sr_score.sigmoid())
            # bg_feats = odm_batch_feat[bg_inds]
            # mbg_feats = torch.mean(bg_feats, dim=0, keepdim=True)
            # # num_bg = bg_inds.shape[0]
            # feature_list.append(mbg_feats)
            # label_list.append(torch.Tensor([self.num_classes]).to(device=m_cls_feats.device))
            # ious_list.append(torch.Tensor([1.]).to(device=m_cls_feats.device))
        cls_feats = torch.cat(feature_list)
        labels = torch.cat(label_list)
        ious = torch.cat(ious_list)
        
        scores = torch.cat(score_list)
        return cls_feats, labels, ious, scores

    
    def collect_proposal_scores(self, outs, base_outs, name, sample_name, reshape=False):
        cls_feats       = convert_fp32(outs[name])
        base_cls_feats  = convert_fp32(base_outs[name])
        sample_results  = outs[sample_name] 
        base_sample_results = base_outs[sample_name] 
        num_level_anchors_ori = outs["num_level_anchors"] 

        feature_list = []; base_feature_list = []
        num_level_anchors = np.asarray(num_level_anchors_ori).cumsum()
        num_level_anchors = np.insert(num_level_anchors, 0, 0)
        for layerid, (cls_feat, base_cls_feat) in enumerate(zip(cls_feats, base_cls_feats)):
            if reshape:
                cls_feat = cls_feat.reshape(*cls_feat.shape[:2], -1) # T [B, C, H*W]
                cls_feat = cls_feat.permute(0, 2, 1)  # T[B, H*W, C]
                base_cls_feat = base_cls_feat.reshape(*base_cls_feat.shape[:2], -1) # T [B, C, H*W]
                base_cls_feat = base_cls_feat.permute(0, 2, 1)  # T[B, H*W, C]
            
            for i, (sample_result, base_sample_result) in enumerate(zip(sample_results, base_sample_results)):   ## len is Batch
                nla_min = num_level_anchors[layerid]
                nla_max = num_level_anchors[layerid + 1]
                # all_inds = sample_result.pos_inds # 使用所有的 pos sample
                all_inds = base_sample_result.pos_inds # 使用所有的 pos sample
                # all_inds = torch.cat([all_inds, base_all_inds]).unique()
                mask = (all_inds >= nla_min) & ( all_inds < nla_max)
                if mask.any() > 0:
                    all_inds = all_inds[mask]
                    label = base_sample_result.labels[all_inds]

                    all_inds -= nla_min
                    feat = cls_feat[i][all_inds]
                    ma = label < self.base_classes
                    feature_list.append(feat[ma])
                    feat = base_cls_feat[i][all_inds]
                    base_feature_list.append(feat[ma])

        cls_feats = torch.cat(feature_list)
        base_cls_feats = torch.cat(base_feature_list)
        return cls_feats, base_cls_feats
    
    def collect_proposal_box(self, outs, base_outs, name):
        box_regs       = convert_fp32(outs[name])
        base_box_regs  = convert_fp32(base_outs[name])
        anchors        = convert_fp32(base_outs["refine_anchors"])
        base_anchors   = convert_fp32(base_outs["refine_anchors"])
       

        feature_list = []; base_feature_list = []
        for layerid, (box_reg, anchor, base_box_reg, base_anchor ) in enumerate(zip(box_regs, anchors, base_box_regs, base_anchors)):
            box_reg = box_reg.reshape(*box_reg.shape[:2], -1) # T [B, C, H*W]
            box_reg = box_reg.permute(0, 2, 1)  # T[B, H*W, C]
            batch_size = len(box_reg)
            base_box_reg = base_box_reg.reshape(*base_box_reg.shape[:2], -1) # T [B, C, H*W]
            base_box_reg = base_box_reg.permute(0, 2, 1)  # T[B, H*W, C]
            anchor      = anchor.reshape(batch_size,        -1, self.box_dim)
            base_anchor = base_anchor.reshape(batch_size,   -1, self.box_dim)
            for i in range(batch_size):
                feat = box_reg[i]
                anc = anchor[i]
                feature_list.append(delta2bbox_rotated(anc, feat))

                feat = base_box_reg[i]
                anc = base_anchor[i]
                base_feature_list.append(delta2bbox_rotated(anc, feat))

        box_regs = torch.cat(feature_list)
        base_box_regs = torch.cat(base_feature_list)
        return box_regs, base_box_regs


@HEADS.register_module()
class ContrastId_S2ANetHead(Contrast_S2ANetHead):

    def loss_contrastive_feat(self, cls_feats, labels, ious):
        if len(cls_feats) == 0:
            contrast_loss = torch.tensor(0.0, device=cls_feats.device)
            return contrast_loss
        
        cls_feats = self.encoder(cls_feats)
        contrast_loss = self.criterion(cls_feats, labels, ious)
        return contrast_loss 
    def origin_collect_sample_features(self, outs: "S2ANetHeadLossResult",
            gt_bboxes: "list[torch.Tensor]", gt_labels: "list[torch.Tensor]", img_metas: "list[TrainImageMeta]",
            gt_bboxes_ignore: "list[torch.Tensor]"=None, ids=None):
        ### 类似 loss() 但不输出loss bf04.23 添加 fam part
        fam_ret = self.loss_part_fam(outs, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        if fam_ret is None:
                return None
        losses_fam_cls, losses_fam_bbox, fam_sample_results = fam_ret
        outs['fam_sample_results'] = fam_sample_results
        odm_ret = self.loss_part_odm(outs, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        if odm_ret is None:
            return None
        losses_odm_cls, losses_odm_bbox, odm_sample_results = odm_ret

        # if odm_sample_results is not None:
        outs['odm_sample_results'] = odm_sample_results

        return self._collect_proposal_features(outs)

    def _collect_proposal_features(self, odm_cls_feats: "list[torch.Tensor]", outs: "ConS2ANetHeadForwardResult"):
        
        odm_sample_results: "SamplingResult"  = outs['odm_sample_results'] 
        m_cls_feats = torch.cat(odm_cls_feats, dim=1)
        batch_size = len(odm_sample_results)

        feature_list = []; label_list = []; ious_list = []
        for bid in range(batch_size):
            # bsf.t 测试无 encoder 情况
            # odm_batch_feat = self.encoder(odm_cls_feat)
            odm_batch_feat = m_cls_feats[bid]
            odm_sample_result = odm_sample_results[bid]
            if not self.use_mlp_cls: # 使用 MLP Conv卷积层
                odm_batch_feat = odm_batch_feat.reshape(*odm_batch_feat.shape[:2], -1) # T [B, C, H*W]
                odm_batch_feat = odm_batch_feat.permute(0, 2, 1)  # T[B, H*W, C]
  
            s_iou = odm_sample_result.max_overlaps
            if not hasattr(odm_sample_result, 'all_inds'):
                mask = (s_iou > self.iou_threshold)
                all_inds = torch.where(mask > 0)[0]
                setattr(odm_sample_result, "all_inds", all_inds)
            else:
                all_inds = odm_sample_result.all_inds # 使用所有的 pos sample
            
            if all_inds.any() > 0:
                ## bf.230412 label 没有经过 image_to_level 处理
                sr_label = odm_sample_result.labels[all_inds]
                # sr_id = odm_sample_result.ids[all_inds] 
                sr_iou = s_iou[all_inds]
                ## bsf.c feature are processed in batch
                # all_feat = self.encoder(odm_batch_feat)
                feat = odm_batch_feat[all_inds]
                feature_list.append(feat)
                label_list.append(sr_label[all_inds])
                ious_list.append(sr_iou[all_inds])
        cls_feats = torch.cat(feature_list)
        labels = torch.cat(label_list)
        ious = torch.cat(ious_list)
        return cls_feats, labels, ious
    def _collect_proposal_features_id(self, odm_cls_feats: "list[torch.Tensor]", outs: "ConS2ANetHeadForwardResult"):
        odm_sample_results: "SamplingResult"    = outs['odm_sample_results'] 
        odm_cls_scores: "list[torch.Tensor]"    = outs['odm_cls_scores'] 
        if self.use_mlp_cls:
            m_cls_feats = torch.cat(odm_cls_feats, dim=1)
            odm_cls_scores = torch.cat(odm_cls_scores, dim=1)

        batch_size = len(odm_sample_results)

        feature_list = []; label_list = []; ious_list = []; id_list = []; score_list = []
            
        for bid in range(batch_size):
            
            odm_batch_feat = m_cls_feats[bid]
            odm_sample_result = odm_sample_results[bid]
            odm_cls_score = odm_cls_scores[bid]
  
            s_iou = odm_sample_result.max_overlaps
            if not hasattr(odm_sample_result, 'all_inds'):
                mask = (s_iou > self.iou_threshold)
                bg_mask = (s_iou < self.bg_iou_threshold) 
                ms = mask.sum() * 3
                if bg_mask.sum() > ms:
                    ind = torch.where(bg_mask > 0)[0]
                    bg_mask[ind[ms:]] = 0
                all_inds = torch.where(mask > 0)[0]
                setattr(odm_sample_result, "all_inds", all_inds)
            else:
                all_inds = odm_sample_result.all_inds # 使用所有的 pos sample
            
            if all_inds.any() > 0:
                ## bf.230412 label 没有经过 image_to_level 处理
                sr_label = odm_sample_result.labels[all_inds]
                sr_score = odm_cls_score[all_inds]
                # sr_id = odm_sample_result.ids[all_inds]
                # sr_label = odm_sample_result.ids[all_inds] 
                sr_iou = s_iou[all_inds]
                # all_feat = self.encoder(odm_batch_feat)  # bsf.c 在收集完 feature 后再 encoder
                feat = odm_batch_feat[all_inds]
                feature_list.append(feat)
                label_list.append(sr_label)
                ious_list.append(sr_iou)
                # id_list.append(sr_id)
                score_list.append(sr_score.sigmoid())
        cls_feats = torch.cat(feature_list)
        labels = torch.cat(label_list)
        ious = torch.cat(ious_list)
        # ids = torch.cat(id_list)
        scores = torch.cat(score_list)
        return cls_feats, labels, ious, scores