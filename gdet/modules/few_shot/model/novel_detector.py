import logging
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset
from mmdet.core.bbox.transforms import bbox2result
from mmrotate.models.detectors.s2anet import S2ANet

class NovelDetector(S2ANet):
    def set_datasets(self, datasets):
        super().set_datasets(datasets)
        offset = 0 if self.bbox_head.use_sigmoid_cls else 1
        if isinstance(self._datasets, ConcatDataset):
            bc = self._datasets.datasets[0].base_classes
        else:
            bc = self._datasets.base_classes

        self.bbox_head.base_cls_out_channels = len(bc) + offset
        self.base_mask = [ x for x in bc]
        
        self.len_base_mask = len(self.base_mask)
        self.data_mask_count = len(self._datasets.CLASSES) + offset
    
    def get_base_mapping(self, device):
        base_mapping = torch.zeros(self.data_mask_count, device=device).to(torch.bool)
        
        base_mapping[self.base_mask] = True
        if not self.bbox_head.use_sigmoid_cls:
            base_mapping[-1] = True # 
        return base_mapping
    
    def forward_for_visual(self, outs, img: "torch.Tensor", img_metas, gt_bboxes, gt_labels):
        ### visual ###
        contain_novel_bidx = -1
        for i in range(len(gt_labels)):
            if (gt_labels[i] >= len(self.base_mask)).any():
                contain_novel_bidx = i
                break
        device = gt_bboxes[0].device
        if contain_novel_bidx > -1: # show 
            if self.last_iter % 50 == 0:
                
                base_mapping = self.get_base_mapping(device)
                # 使用 base 和 novel 分支共同预测
                self.bbox_head.base_mapping = base_mapping
                bbox_list = self.bbox_head.get_bboxes(outs, img_metas)
                
                bbox_results = [
                    bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                    for det_bboxes, det_labels in bbox_list
                ]
                ti = img[contain_novel_bidx]
                ti_meta = img_metas[contain_novel_bidx]
                ti_pred = bbox_results[contain_novel_bidx]
                ti_gt = gt_labels[contain_novel_bidx]
                pred_ins = sum((p.shape[0] for p in ti_pred))
                self.show_train_result(ti, ti_meta, ti_pred, dataset = self.datasets, vis="train", extra=f"(p {pred_ins} / g {len(ti_gt)} )")
                del bbox_results
            self.last_iter += 1

