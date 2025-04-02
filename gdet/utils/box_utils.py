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

"""Utility functions for bounding box processing.

This is a reimplementation of the box_utils at:
https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/box_utils.py
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

_EPSILON = 1e-7
BBOX_XFORM_CLIP = np.log(1000.0 / 16.0)



def filter_boxes(boxes: "torch.Tensor", scores: "torch.Tensor", image_shape, min_size_threshold):
    """Filter and remove boxes that are too small or fall outside the image.

    Args:
      boxes: a tensor whose last dimension is 4 representing the
        coordinates of boxes in ymin, xmin, ymax, xmax order.
      scores: a tensor whose shape is the same as boxes.shape[:-1]
        representing the original scores of the boxes.
      image_shape: a tensor whose shape is the same as, or `broadcastable` to
        `boxes` except the last dimension, which is 2, representing
        [height, width] of the scaled image.
      min_size_threshold: a float representing the minimal box size in each
        side (w.r.t. the scaled image). Boxes whose sides are smaller than it will
        be filtered out.

    Returns:
      filtered_boxes: a tensor whose shape is the same as `boxes` but with
        the position of the filtered boxes are filled with 0.
      filtered_scores: a tensor whose shape is the same as 'scores' but with
        the positinon of the filtered boxes filled with 0.
    """
    assert boxes.shape[-1] == 4, \
            "boxes.shape[1] is {:d}, but must be 4.".format(boxes.shape[-1])

    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
        height, width = image_shape
    else:
        width = image_shape[0]
        height = image_shape[1]

    xmin = boxes[Ellipsis, 0]
    ymin = boxes[Ellipsis, 1]
    xmax = boxes[Ellipsis, 2]
    ymax = boxes[Ellipsis, 3]

    # computing height and center of boxes
    w = xmax - xmin + 1.0
    h = ymax - ymin + 1.0
    xc = xmin + 0.5 * w
    yc = ymin + 0.5 * h

    min_size = max(min_size_threshold, 1.0)

    # filtering boxes based on constraints
    filtered_size_mask = (h > min_size) & (w > min_size)
    filtered_center_mask = torch.logical_and(
        (yc > 0.0) & (yc < height), (xc > 0.0) & (xc < width)
    )
    filtered_mask = (filtered_size_mask & filtered_center_mask)

    filtered_scores = torch.where(filtered_mask, scores, torch.zeros_like(scores))
    filtered_boxes  = torch.where(filtered_mask.unsqueeze(-1).repeat(1, 1, 4), boxes, torch.zeros_like(boxes))

    return filtered_boxes, filtered_scores


def filter_boxes_by_scores(boxes: "torch.Tensor", scores: "torch.Tensor", min_score_threshold):
    """Filters and removes boxes whose scores are smaller than the threshold.

    Args:
      boxes: a tensor whose last dimension is 4 representing the
        coordinates of boxes in ymin, xmin, ymax, xmax order.
      scores: a tensor whose shape is the same as tf.shape(boxes)[:-1]
        representing the original scores of the boxes.
      min_score_threshold: a float representing the minimal box score threshold.
        Boxes whose score are smaller than it will be filtered out.

    Returns:
      filtered_boxes: a tensor whose shape is the same as `boxes` but with
        the position of the filtered boxes are filled with 0.
      filtered_scores: a tensor whose shape is the same as 'scores' but with
        the
    """
    assert boxes.shape[-1] == 4, \
            "boxes.shape[1] is {:d}, but must be 4.".format(boxes.shape[-1])

    filtered_mask: "torch.Tensor" = scores < min_score_threshold
    # filtered_scores = torch.where(filtered_mask, scores, torch.zeros_like(scores))
    # filtered_boxes = filtered_mask.unsqueeze(dim=-1).to(boxes.dtype) * boxes
    scores[filtered_mask] = 0
    boxes[filtered_mask] = 0

    return boxes, scores


def clip_boxes(boxes: "torch.Tensor", image_shape: "torch.Tensor"):
    """Clips boxes to image boundaries. It's called from roi_ops.py.

    Args:
      boxes: a tensor whose last dimension is 4 representing the coordinates
        of boxes in ymin, xmin, ymax, xmax order.
      image_shape: a list of two integers, a two-element vector or a tensor such
        that all but the last dimensions are `broadcastable` to `boxes`. The last
        dimension is 2, which represents [height, width].

    Returns:
      clipped_boxes: a tensor whose shape is the same as `boxes` representing the
        clipped boxes in xmin, ymin, xmax, ymax, order.

    Raises:
      ValueError: If the last dimension of boxes is not 4.
    """
    assert boxes.shape[-1] == 4, \
            "boxes.shape[1] is {:d}, but must be 4.".format(boxes.shape[-1])

    if isinstance(image_shape, Sequence) or isinstance(image_shape, tuple):
        height, width = image_shape
    else:
        width  = image_shape[0][0]
        height = image_shape[0][1]
    # Clamp x coordinates (xmin and xmax) to be within [0, 480]
    # Clamp y coordinates (ymin and ymax) to be within [0, 640]
    boxes[Ellipsis, [0, 2]] = torch.clamp(boxes[Ellipsis, [0, 2]], min=0, max=width - 1)
    boxes[Ellipsis, [1, 3]] = torch.clamp(boxes[Ellipsis, [1, 3]], min=0, max=height - 1)

    return boxes


def encode_boxes(boxes: "torch.Tensor", anchors: "torch.Tensor", weights=None):
    """Encode boxes to targets.

    Args:
      boxes: a tensor whose last dimension is 4 representing the coordinates
        of boxes in ymin, xmin, ymax, xmax order.
      anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
        representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
      weights: None or a list of four float numbers used to scale coordinates.

    Returns:
      encoded_boxes: a tensor whose shape is the same as `boxes` representing the
        encoded box targets.

    Raises:
      ValueError: If the last dimension of boxes is not 4.
    """
    assert boxes.shape[-1] == 4, \
            "boxes.shape[1] is {:d}, but must be 4.".format(boxes.shape[-1])

    boxes = boxes.to(anchors.dtype)
    xmin = boxes[Ellipsis, 0:1]
    ymin = boxes[Ellipsis, 1:2]
    xmax = boxes[Ellipsis, 2:3]
    ymax = boxes[Ellipsis, 3:4]
    box_w = xmax - xmin + 1.0
    box_h = ymax - ymin + 1.0
    box_xc = xmin + 0.5 * box_w
    box_yc = ymin + 0.5 * box_h

    anchor_xmin = anchors[Ellipsis, 0:1]
    anchor_ymin = anchors[Ellipsis, 1:2]
    anchor_xmax = anchors[Ellipsis, 2:3]
    anchor_ymax = anchors[Ellipsis, 3:4]
    anchor_w = anchor_xmax - anchor_xmin + 1.0
    anchor_h = anchor_ymax - anchor_ymin + 1.0
    anchor_xc = anchor_xmin + 0.5 * anchor_w
    anchor_yc = anchor_ymin + 0.5 * anchor_h

    encoded_dx = (box_xc - anchor_xc) / anchor_w
    encoded_dy = (box_yc - anchor_yc) / anchor_h
    encoded_dw = torch.log(box_w / anchor_w)
    encoded_dh = torch.log(box_h / anchor_h)
    if weights:
        encoded_dx = encoded_dx * weights[0]
        encoded_dy = encoded_dy * weights[1]
        encoded_dw = encoded_dw * weights[2]
        encoded_dh = encoded_dh * weights[3]

    enc_boxes = [encoded_dx, encoded_dy, encoded_dw, encoded_dh,]
    enc_boxes = torch.concatenate(enc_boxes, dim=-1)
    return enc_boxes


def decode_boxes(encoded_boxes: "torch.Tensor", anchors: "torch.Tensor", weights=None):
    """Decode boxes.

    Args:
      encoded_boxes: T[B, N, D] a tensor whose last dimension is 4 representing the
        coordinates of encoded boxes in xmin, xmax, ymin, ymax order.
      anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
        representing the coordinates of anchors in xmin, xmax, ymin, ymax order.
      weights: None or a list of four float numbers used to scale coordinates.

    Returns:
      encoded_boxes: a tensor whose shape is the same as `boxes` representing the
        decoded box targets.
    """
    dx = encoded_boxes[Ellipsis, 0:1]
    dy = encoded_boxes[Ellipsis, 1:2]
    dw = encoded_boxes[Ellipsis, 2:3]
    dh = encoded_boxes[Ellipsis, 3:4]
    if weights:
        dx = dx / weights[0]
        dy = dy / weights[1]
        dw = dw / weights[2]
        dh = dh / weights[3]
    dw = torch.clamp(dw, max=BBOX_XFORM_CLIP)
    dh = torch.clamp(dh, max=BBOX_XFORM_CLIP)

    anchor_xmin = anchors[Ellipsis, 0:1]
    anchor_ymin = anchors[Ellipsis, 1:2]
    anchor_xmax = anchors[Ellipsis, 2:3]
    anchor_ymax = anchors[Ellipsis, 3:4]

    anchor_w = anchor_xmax - anchor_xmin + 1.0
    anchor_h = anchor_ymax - anchor_ymin + 1.0
    anchor_xc = anchor_xmin + 0.5 * anchor_w
    anchor_yc = anchor_ymin + 0.5 * anchor_h

    dec_boxes_xc = dx * anchor_w + anchor_xc
    dec_boxes_yc = dy * anchor_h + anchor_yc
    dec_boxes_w = torch.exp(dw) * anchor_w ### original
    dec_boxes_h = torch.exp(dh) * anchor_h
    # dec_boxes_w = torch.clamp(torch.exp(dw) * anchor_w, min=1.000001) ### for statbility
    # dec_boxes_h = torch.clamp(torch.exp(dh) * anchor_h, min=1.000001)
    #assert (dec_boxes_w.min() > 0) and (dec_boxes_h.min() > 0)
    dec_boxes_xmin = dec_boxes_xc - 0.5 * dec_boxes_w
    dec_boxes_ymin = dec_boxes_yc - 0.5 * dec_boxes_h
    dec_boxes_xmax = dec_boxes_xc + 0.5 * dec_boxes_w
    dec_boxes_ymax = dec_boxes_yc + 0.5 * dec_boxes_h
    # dec_boxes_xmax = dec_boxes_xmin + dec_boxes_w - 1.0
    # dec_boxes_ymax = dec_boxes_ymin + dec_boxes_h - 1.0

    dec_boxes = [
        dec_boxes_xmin,
        dec_boxes_ymin,
        dec_boxes_xmax,
        dec_boxes_ymax,
    ]
    dec_boxes = torch.concatenate(dec_boxes, dim=-1,)  ## T[2, 19200, 4]
    return dec_boxes


def decode_boxes_lrtb(encoded_boxes_lrtb: "torch.Tensor", anchors: "torch.Tensor", weights=None):
    """Decode LRTB boxes.

    Args:
      encoded_boxes_lrtb: a tensor whose last dimension is 4 representing the
        coordinates of encoded boxes in left, right, top, bottom order.
      anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
        representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
      weights: None or a list of four float numbers used to scale coordinates.
    Returns:
      decoded_boxes_lrtb: a tensor whose shape is the same as `boxes` representing
        the decoded box targets in lrtb (=left,right,top,bottom) format. The box
        decoded box coordinates represent the left, right, top, and bottom
        distances from an anchor location to the four borders of the matched
        groundtruth bounding box.
    """
    if encoded_boxes_lrtb.shape[-1] != 4:
        raise ValueError(
            "encoded_boxes_lrtb.shape[-1] is {:d}, but must be 4.".format(
                encoded_boxes_lrtb.shape[-1]
            )
        )

    encoded_boxes_lrtb = encoded_boxes_lrtb.to(anchors.dtype)
    left = encoded_boxes_lrtb[Ellipsis, 0:1]
    right = encoded_boxes_lrtb[Ellipsis, 1:2]
    top = encoded_boxes_lrtb[Ellipsis, 2:3]
    bottom = encoded_boxes_lrtb[Ellipsis, 3:4]
    if weights:
        left = left / weights[0]
        right = right / weights[1]
        top = top / weights[2]
        bottom = bottom / weights[3]

    anchor_xmin = anchors[Ellipsis, 0:1]
    anchor_ymin = anchors[Ellipsis, 1:2]
    anchor_xmax = anchors[Ellipsis, 2:3]
    anchor_ymax = anchors[Ellipsis, 3:4]
    anchor_h = anchor_ymax - anchor_ymin
    anchor_w = anchor_xmax - anchor_xmin
    anchor_yc = anchor_ymin + 0.5 * anchor_h
    anchor_xc = anchor_xmin + 0.5 * anchor_w

    dec_boxes_ymin = anchor_yc - top * anchor_h
    dec_boxes_xmin = anchor_xc - left * anchor_w
    dec_boxes_ymax = anchor_yc + bottom * anchor_h
    dec_boxes_xmax = anchor_xc + right * anchor_w
    dec_box = [
            dec_boxes_xmin,
            dec_boxes_ymin,
            dec_boxes_xmax,
            dec_boxes_ymax,
        ]
    dec_boxes_lrtb = torch.concatenate(dec_box, dim=-1,)
    return dec_boxes_lrtb



def rescale_boxes(boxes: "torch.Tensor", image_size, scale=1.0):
    """Rescale boxes by scale.

    Args:
      boxes: Boxes of shape [batch, num_rois, 4] representing the
        coordinates of boxes in ymin, xmin, ymax, xmax order in last dimension.
      image_size: Image shape array of shape [batch, 2] representing
        [height, width] of the scaled image in the last dimension.
      scale: The scale to resize the boxes by.

    Returns:
      clipped_boxes: Boxes of shape [batch, num_rois, 4] representing the
        coordinates of boxes in ymin, xmin, ymax, xmax order in last dimension.
    """
    if scale == 1.0:
        return boxes

    box_width = scale * (boxes[:, :, 2] - boxes[:, :, 0])
    box_height = scale * (boxes[:, :, 3] - boxes[:, :, 1])
    box_x = 0.5 * (boxes[:, :, 2] + boxes[:, :, 0])
    box_y = 0.5 * (boxes[:, :, 3] + boxes[:, :, 1])
    new_boxes = [
        box_x - box_width * 0.5,
        box_y - box_height * 0.5,
        box_x + box_width * 0.5,
        box_y + box_height * 0.5,
    ]
    new_boxes = torch.concatenate([coords[:, :, None] for coords in new_boxes], -1)
    clipped_boxes = clip_boxes(new_boxes, image_size[:, None, :])
    return clipped_boxes


def xyxy_to_xywh(boxes: "np.ndarray"):
    """Converts boxes from ymin, xmin, ymax, xmax to xmin, ymin, width, height.

    Args:
      boxes: a numpy array whose last dimension is 4 representing the coordinates
        of boxes in ymin, xmin, ymax, xmax order.

    Returns:
      boxes: a numpy array whose shape is the same as `boxes` in new format.

    Raises:
      ValueError: If the last dimension of boxes is not 4.
    """
    assert boxes.shape[-1] == 4, \
            "boxes.shape[1] is {:d}, but must be 4.".format(boxes.shape[-1])

    boxes_xmin = boxes[Ellipsis, 0]
    boxes_ymin = boxes[Ellipsis, 1]
    boxes_width  = boxes[Ellipsis, 2] - boxes[Ellipsis, 0]
    boxes_height = boxes[Ellipsis, 3] - boxes[Ellipsis, 1]
    new_boxes = np.stack([boxes_xmin, boxes_ymin, boxes_width, boxes_height], axis=-1)

    return new_boxes
