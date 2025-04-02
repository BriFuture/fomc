# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Non-max Suppression example.

This script does non-max suppression used in models like SSD
"""

import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import nms as tv_nms
from torchvision.ops import boxes as box_ops
_NMS_TILE_SIZE = 256


def _bbox_overlap(boxes, gt_boxes):
    """Find Bounding box overlap.

    Args:
      boxes: first set of bounding boxes
      gt_boxes: second set of boxes to compute IOU

    Returns:
      iou: Intersection over union matrix of all input bounding boxes
    """
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = torch.split(
        ary=boxes, indices_or_sections=4, axis=2
    )
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = torch.split(
        ary=gt_boxes, indices_or_sections=4, axis=2
    )

    # Calculates the intersection area.
    i_xmin = torch.maximum(bb_x_min, torch.transpose(gt_x_min, [0, 2, 1]))
    i_xmax = torch.minimum(bb_x_max, torch.transpose(gt_x_max, [0, 2, 1]))
    i_ymin = torch.maximum(bb_y_min, torch.transpose(gt_y_min, [0, 2, 1]))
    i_ymax = torch.minimum(bb_y_max, torch.transpose(gt_y_max, [0, 2, 1]))
    i_area = torch.maximum((i_xmax - i_xmin), 0) * torch.maximum((i_ymax - i_ymin), 0)

    # Calculates the union area.
    bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
    gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
    # Adds a small epsilon to avoid divide-by-zero.
    u_area = bb_area + torch.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

    # Calculates IoU.
    iou = i_area / u_area

    return iou

def non_max_suppression_padded(scores: "torch.Tensor", boxes: torch.Tensor, max_output_size: int, iou_threshold: float):
    batch_size = boxes.shape[0]
    num_boxes = boxes.shape[1] ## 2000
    
    pad = int(np.ceil(float(num_boxes) / _NMS_TILE_SIZE)) * _NMS_TILE_SIZE - num_boxes  ## 48
    boxes = torch.nn.functional.pad(boxes, (0, 0, 0, pad), "constant", 0)  ## T[2, 2048, 4]
    scores = torch.nn.functional.pad(scores, (0, pad), "constant", 0)  ## T[2, 2048]
    num_boxes += pad  ## 2048

    output_size = torch.zeros(batch_size, dtype=torch.int32)
    selected_boxes = []

    for idx in range(num_boxes // _NMS_TILE_SIZE):
        start_idx = idx * _NMS_TILE_SIZE
        end_idx = min((idx + 1) * _NMS_TILE_SIZE, num_boxes)
        
        box_tile = boxes[:, start_idx:end_idx, :]
        
        for j in range(idx):
            suppressing_tile_start_idx = j * _NMS_TILE_SIZE
            suppressing_tile_end_idx = (j + 1) * _NMS_TILE_SIZE
            suppressing_tile = boxes[:, suppressing_tile_start_idx:suppressing_tile_end_idx, :]
            
            iou = _bbox_overlap(box_tile, suppressing_tile)
            box_tile = box_tile * (iou < 0.5).unsqueeze(2).float()
        
        iou = _bbox_overlap(box_tile, box_tile)
        iou_changed = True
        
        while iou_changed:
            suppressing_boxes = (torch.sum(iou, dim=2) == 0).float()
            suppressed_boxes = (torch.sum(iou, dim=1) > 0).float()
            new_iou = iou * (1 - suppressed_boxes.unsqueeze(2))
            iou_changed = not torch.equal(new_iou, iou)
            iou = new_iou
        
        selected_boxes.append(suppressing_boxes.unsqueeze(1).repeat(1, _NMS_TILE_SIZE, 1))
        if torch.min(output_size) >= max_output_size:
            break
    
    selected_boxes = torch.cat(selected_boxes, dim=1)
    selected_indices = torch.topk(selected_boxes.view(-1), max_output_size, largest=True, sorted=False).indices
    selected_boxes = boxes.view(-1, 4)[selected_indices].view(batch_size, max_output_size, 4)
    selected_scores = scores.view(-1)[selected_indices].view(batch_size, max_output_size)
    
    return selected_scores, selected_boxes

def batched_nms(scores: "torch.Tensor", boxes: "torch.Tensor", max_output_size: "int", iou_threshold: "float", ):
    """used for batch_nms  from torch vision
    """
    nms_boxes  = []
    nms_scores = []
    for box, score in zip(boxes, scores):
        keep: "torch.Tensor" = tv_nms(box, score, iou_threshold)
        keep = keep[:max_output_size]
        box = box[keep]
        score = score[keep]
        nms_boxes.append(box)
        nms_scores.append(score)

    nms_boxes = torch.stack(nms_boxes, dim=0)    ## T[2, 1000, 4]
    nms_scores = torch.stack(nms_scores, dim=0)  ## T[2, 1000, ]
    return nms_scores, nms_boxes

def nms_padded(scores: "torch.Tensor", boxes: "torch.Tensor", 
            max_output_size: "int", iou_threshold: "float", ):
    """
    scores: T[B, N, ]
    boxes: T[B, N, 4]
    """
    """Non-maximum suppression with padding."""
    # keep = nms.non_max_suppression_padded(scores, boxes, max_output_size, iou_threshold)
    nms_boxes  = []
    nms_scores = []
    for box, score in zip(boxes, scores):
        keep: "torch.Tensor" = tv_nms(box, score, iou_threshold)
        keep = keep[:max_output_size]
        box = box[keep]
        score = score[keep]
        count = box.shape[0]
        if count < max_output_size:
            z_box = torch.zeros((max_output_size - count, box.shape[1])).to(device=box.device)
            z_score = torch.zeros((max_output_size - count,)).to(device=box.device)
            box = torch.cat([box, z_box], dim=0)
            score = torch.cat([score, z_score], dim=0)
        nms_boxes.append(box)
        nms_scores.append(score)

    nms_boxes = torch.stack(nms_boxes, dim=0)    ## T[2, 1000, 4]
    nms_scores = torch.stack(nms_scores, dim=0)  ## T[2, 1000, ]
    return nms_scores, nms_boxes


def nms_padded_with_idx(scores: "torch.Tensor", boxes: "torch.Tensor", 
    max_output_size: "int", iou_threshold: "float", 
    score_threshold=0.1, skip_bg=True
):
    """
    scores: T[B, N, ]
    boxes: T[B, N, 4]
    """
    """Non-maximum suppression with padding."""
    # keep = nms.non_max_suppression_padded(scores, boxes, max_output_size, iou_threshold)
    sval, sidx = scores.max(dim=-1)
    nms_boxes  = []
    nms_scores = []
    nms_idxes = []

    for box, score, idx in zip(boxes, sval, sidx):
        keep: "torch.Tensor" = tv_nms(box, score, iou_threshold)
        keep = keep[:max_output_size]
        box   = box[keep]
        score = score[keep]
        idx   = idx[keep]
        if score_threshold > 0:
            keep = score > score_threshold
            box   = box[keep]
            score = score[keep]
            idx   = idx[keep]
        count = box.shape[0]
        if count < max_output_size:
            z_box = torch.zeros((max_output_size - count, box.shape[1])).to(device=box.device)
            z_score = torch.zeros((max_output_size - count,)).to(device=box.device)
            box = torch.cat([box, z_box], dim=0)
            score = torch.cat([score, z_score], dim=0)
            idx = torch.cat([idx, z_score], dim=0)
        nms_boxes.append(box)
        nms_scores.append(score)
        nms_idxes.append(idx)

    nms_boxes = torch.stack(nms_boxes, dim=0)    ## T[2, 1000, 4]
    nms_scores = torch.stack(nms_scores, dim=0)  ## T[2, 1000, ]
    nms_idxes = torch.stack(nms_idxes, dim=0)  ## T[2, 1000, ]
    return nms_scores, nms_boxes, nms_idxes

def fg_nms_padded_with_idx(scores: "torch.Tensor", boxes: "torch.Tensor", 
    max_output_size: "int", iou_threshold: "float", 
    score_threshold=0.1
):
    """
    scores: T[N, ]
    boxes: T[N, 4]
    """
    """Non-maximum suppression with padding."""
    # keep = nms.non_max_suppression_padded(scores, boxes, max_output_size, iou_threshold)
    sval, sidx = scores.max(dim=-1)
    keep: "torch.Tensor" = tv_nms(boxes, sval, iou_threshold)
    keep = keep[:max_output_size]
    box   = boxes[keep]
    score = sval[keep]
    idx   = sidx[keep]
    if score_threshold > 0:
        keep = score > score_threshold
        box   = box[keep]
        score = score[keep]
        idx   = idx[keep]
    
    nms_boxes = box    ## T[2, 1000, 4]
    nms_scores = score  ## T[2, 1000, ]
    nms_idxes = idx  ## T[2, 1000, ]
    return nms_scores, nms_boxes, nms_idxes
