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

"""Generates detections from network predictions.

This is a JAX reimplementation of:
https://github.com/tensorflow/tpu/blob/master/models/official/detection/ops/postprocess_ops.py
"""

import functools
from typing import Dict, Tuple


import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

from torchvision.ops import nms
from gdet.utils import box_utils

BBOX_XFORM_CLIP = np.log(1000.0 / 16.0)


def batch_gather(x: "torch.Tensor", idx):
    """Performs a batched gather of the data.

    Args:
        x: A [batch, num_in, ...] tensor of data to gather from.
        idx: A [batch, num_out] tensor of dtype int32 or int64 specifying which
            elements to gather. Every value is expected to be in the range of [0, num_in].

    Returns:
        A [batch, num_out, ...] tensor of gathered data.
    """
    batch_size = x.shape[0]
    gathered = torch.stack([x[i][idx[i]] for i in range(batch_size)], dim=0)
    return gathered


def nms_padded(score: "torch.Tensor", box: "torch.Tensor", 
            max_output_size: "int", iou_threshold: "int", ):
    """Non-maximum suppression with padding. in single batch
    scores: T[N, ]
    box:  T[N, 4]
    """
    
    keep: "torch.Tensor" = nms(box, score, iou_threshold)
    keep = keep[:max_output_size]
    box = box[keep]
    score = score[keep]
    count = box.shape[0]
    if count < max_output_size:
        z_box = torch.zeros((max_output_size - count, box.shape[1])).to(device=box.device)
        z_score = torch.zeros((max_output_size - count,)).to(device=box.device)
        box = torch.cat([box, z_box], dim=0)
        score = torch.cat([score, z_score], dim=0)

    return score, box

def generate_detections(
    class_outputs: "torch.Tensor",
    box_outputs: "torch.Tensor",
    pre_nms_num_detections=5000,
    post_nms_num_detections=100,
    nms_threshold=0.3,
    score_threshold=0.05,
    class_box_regression=True,
    base_class_indicator=None,
):
    """Generates the detections given anchor boxes and predictions.

    Args:
      class_outputs: An array with shape [batch, num_boxes, num_classes] of
        class logits for each box.
      box_outputs: An array with shape [batch, num_boxes, num_classes, 4] of
        predicted boxes in [ymin, xmin, ymax, xmax] order. Also accept
        num_classes = 1 for class agnostic box outputs.
      pre_nms_num_detections: An integer that specifies the number of candidates
        before NMS.
      post_nms_num_detections: An integer that specifies the number of candidates
        after NMS.
      nms_threshold: A float number to specify the IOU threshold of NMS.
      score_threshold: A float representing the threshold for deciding when to
        remove boxes based on score.
      class_box_regression: Whether to use class-specific box regression or not.
        Default True is to assume box_outputs are class-specific.

    Returns:
      A tuple of arrays corresponding to
        (box coordinates, object categories for each boxes, and box scores).
    """
    batch_size, _, num_classes = class_outputs.shape

    final_boxes = []
    final_scores = []
    final_classes = []
    all_valid = []
    for b in range(batch_size):
        nmsed_boxes = []
        nmsed_scores = []
        nmsed_classes = []
        # Skips the background class.
        for i in range(1, num_classes):
            box_idx  = i if class_box_regression else 0
            boxes_i  = box_outputs[b, :, box_idx]
            scores_i = class_outputs[b, :, i]
            # Filter by threshold.
            if base_class_indicator is None:
                above_threshold = scores_i > score_threshold
            else:
                # above_threshold = scores_i > score_threshold
                ###  lower novel class threshold
                if base_class_indicator[i] == True:  ### base
                    above_threshold = scores_i > score_threshold
                else: ### novel
                    above_threshold = scores_i > 0.001

            scores_i = torch.where(above_threshold, scores_i, torch.full_like(scores_i, -1))
            # Obtains pre_nms_num_boxes before running NMS.
            top_k_count = min(pre_nms_num_detections, scores_i.shape[-1])
            scores_i, indices = torch.topk(
                scores_i, k=top_k_count
            )
            boxes_i = boxes_i[indices]

            nmsed_scores_i, nmsed_boxes_i = nms_padded(
                scores_i,
                boxes_i,
                max_output_size=post_nms_num_detections,
                iou_threshold=nms_threshold,
            )

            nmsed_classes_i = torch.ones([post_nms_num_detections]) * i
            nmsed_classes_i = nmsed_classes_i.to(boxes_i.device)
            nmsed_boxes.append(nmsed_boxes_i)
            nmsed_scores.append(nmsed_scores_i)
            nmsed_classes.append(nmsed_classes_i)

        # Concats results from all classes and sort them.
        nmsed_boxes   = torch.concat(nmsed_boxes,   dim=0)
        nmsed_scores  = torch.concat(nmsed_scores,  dim=0)
        nmsed_classes = torch.concat(nmsed_classes, dim=0)
        nmsed_scores, indices = torch.topk(nmsed_scores, k=post_nms_num_detections)
        nmsed_boxes   = nmsed_boxes[indices]
        nmsed_classes = nmsed_classes[indices]
        valid_detections = torch.sum((nmsed_scores > 0.0).to(torch.int32))

        final_boxes.append(nmsed_boxes)
        final_scores.append(nmsed_scores)
        final_classes.append(nmsed_classes)
        all_valid.append(valid_detections)
    final_boxes   = torch.stack(final_boxes, dim=0)
    final_scores  = torch.stack(final_scores, dim=0)
    final_classes = torch.stack(final_classes, dim=0)
    all_valid     = torch.stack(all_valid, dim=0)
    return (final_boxes, final_scores, final_classes, all_valid, )


def process_and_generate_detections(
    box_outputs: "torch.Tensor",
    class_outputs: "torch.Tensor",
    anchor_boxes: "torch.Tensor",
    image_shape,
    pre_nms_num_detections=5000,
    post_nms_num_detections=100,
    nms_threshold=0.5,
    score_threshold=0.05,
    class_box_regression=False,
    class_is_logit=False,
    base_class_indicator = None,
):
    """Generate final detections.

    Args:
      box_outputs: An array of shape of [batch_size, K, num_classes * 4]
        representing the class-specific box coordinates relative to anchors.
      class_outputs: An array of shape of [batch_size, K, num_classes]
        representing the class logits before applying score activiation.
      anchor_boxes: An array of shape of [batch_size, K, 4] representing the
        corresponding anchor boxes w.r.t `box_outputs`.
      image_shape: An array of shape of [batch_size, 2] storing the image height
        and width w.r.t. the scaled image, i.e. the same image space as
        `box_outputs` and `anchor_boxes`.
      pre_nms_num_detections: An integer that specifies the number of candidates
        before NMS.
      post_nms_num_detections: An integer that specifies the number of candidates
        after NMS.
      nms_threshold: A float number to specify the IOU threshold of NMS.
      score_threshold: A float representing the threshold for deciding when to
        remove boxes based on score.
      class_box_regression: Whether to use class-specific box regression or not.
        Default True is to assume box_outputs are class-specific.
      class_is_logit: Whether the class outputs are logits.
      use_vmap: Whether to use a vmapped version of the generate_detections
        functions. This is very helpful to speed up the compile time when the
        number of categories is large.

    Returns:
      A dictionary with the following key-value pairs:
        nmsed_boxes: `float` array of shape [batch_size, max_total_size, 4]
          representing top detected boxes in [y1, x1, y2, x2].
        nmsed_scores: `float` array of shape [batch_size, max_total_size]
          representing sorted confidence scores for detected boxes. The values are
          between [0, 1].
        nmsed_classes: `int` array of shape [batch_size, max_total_size]
          representing classes for detected boxes.
        valid_detections: `int` array of shape [batch_size] only the top
          `valid_detections` boxes are valid detections.
    """

    if class_is_logit:
        class_outputs = F.softmax(class_outputs, dim=-1)
    batch, num_locations, num_classes = class_outputs.shape
    if class_box_regression:
        num_detections = num_locations * num_classes
        box_outputs = box_outputs.reshape(-1, num_detections, 4)
        anchor_boxes = torch.tile(
            anchor_boxes.unsqueeze(dim=2), [1, 1, num_classes, 1]
        )
        anchor_boxes = anchor_boxes.reshape(-1, num_detections, 4)

    dec_boxes = box_utils.decode_boxes(
        box_outputs, anchor_boxes, weights=[10.0, 10.0, 5.0, 5.0]
    ) ## T[1000, 4]
    dec_boxes = box_utils.clip_boxes(dec_boxes, image_shape)
    if class_box_regression:
        dec_boxes = dec_boxes.reshape(-1, num_locations, num_classes, 4)
    else:
        dec_boxes = dec_boxes.unsqueeze(2)

    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = generate_detections(
        class_outputs,
        dec_boxes,
        pre_nms_num_detections,
        post_nms_num_detections,
        nms_threshold,
        score_threshold,
        class_box_regression,
        base_class_indicator,
    )

    return {
        "num_detections": valid_detections,
        "detection_boxes": nmsed_boxes,
        "detection_classes": nmsed_classes,
        "detection_scores": nmsed_scores,
    }
