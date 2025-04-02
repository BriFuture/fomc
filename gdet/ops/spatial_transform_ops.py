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

"""Functions to perform a spatial transformation for Tensor.

This is a reimplementation of:
https://github.com/tensorflow/tpu/blob/master/models/official/detection/ops/spatial_transform_ops.py
in Jax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bfcommon.fp16_utils import convert_fp32
_EPSILON = 1e-8


def nearest_upsampling(data, scale):
    """Nearest neighbor upsampling implementation.

    Args:
      data: An array with a shape of [batch, height_in, width_in, channels].
      scale: An integer multiple to scale resolution of input data.

    Returns:
      data_up: An array with a shape of
        [batch, height_in*scale, width_in*scale, channels]. Same dtype as input
        data.
    """
    batch_size, height, width, channels = data.shape

    # Instead of broadcasting with a 6-d tensor, we're using stacking here.
    output = torch.stack([data] * scale, axis=3)
    output = torch.stack([output] * scale, axis=2)
    return torch.reshape(output, [batch_size, height * scale, width * scale, channels])


def feature_bilinear_interpolation(features: "torch.Tensor", kernel_y: "torch.Tensor", kernel_x: "torch.Tensor"):
    """Feature bilinear interpolation.

    The RoIAlign feature f can be computed by bilinear interpolation
    of four neighboring feature points f0, f1, f2, and f3.

    f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
                          [f10, f11]]
    f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
    f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
    kernel_y = [hy, ly]
    kernel_x = [hx, lx]

    Args:
      features: The features are in shape of [batch_size, num_boxes, output_size *
        2, output_size * 2, num_filters].
      kernel_y: an array of size [batch_size, boxes, output_size, 2, 1].
      kernel_x: an array of size [batch_size, boxes, output_size, 2, 1].

    Returns:
      A 5-D array representing feature crop of shape
      [batch_size, num_boxes, output_size, output_size, num_filters].
    
    """
    batch, num_boxes, num_filters, output_size, _ = features.shape

    output_size = output_size // 2 
    double_output = output_size * 2
    kernel_x = kernel_x.reshape(batch, num_boxes, 1, double_output) ## T[31, 1, 14]
    kernel_y = kernel_y.reshape(batch, num_boxes, double_output, 1) ## T[31, 14, 1]
    # Use implicit broadcast to generate the interpolation kernel. The
    # multiplier `4` is for avg pooling.
    interpolation_kernel = kernel_x * kernel_y * 4 ## T[24, 14, 14]

    # Interpolate the gathered features with computed interpolation kernels.
    features = features * interpolation_kernel.unsqueeze(axis=2) ## T[24, 256, 14, 14]
    features = features.reshape(batch * num_boxes, num_filters, double_output, double_output)
    features = F.avg_pool2d(features, (2, 2), (2, 2)) ## T[512, 256, 7, 7]

    return features.reshape(batch, num_boxes, num_filters, output_size, output_size,) ## T[B, A, C, H, W]


def compute_grid_positions(boxes: "torch.Tensor", boundaries: "torch.Tensor", 
                output_size, sample_offset) -> "tuple[torch.Tensor]":
    """Compute the grid position w.r.t.

    the corresponding feature map.

    Args:
      boxes: a 3-D array of shape [batch_size, num_boxes, 4] encoding the
        information of each box w.r.t. the corresponding feature map.
        boxes[:, :, 0:2] are the grid position in (x, y) (float) of the top-left
        corner of each box. boxes[:, :, 2:4] are the box sizes in (h, w) (float)
          in terms of the number of pixels of the corresponding feature map size.
      boundaries: a 3-D array of shape [batch_size, num_boxes, 2] representing
        the boundary (in (y, x)) of the corresponding feature map for each box.
        Any resampled grid points that go beyond the bounary will be clipped.
      output_size: a scalar indicating the output crop size.
      sample_offset: a float number in [0, 1] indicates the subpixel sample offset
        from grid point.

    Returns:
      kernel_x: an array of size [batch_size, boxes, output_size, 2, 1].
      kernel_y: an array of size [batch_size, boxes, output_size, 2, 1].
      box_grid_x0x1: an array of size [batch_size, boxes, output_size, 2]
      box_grid_y0y1: an array of size [batch_size, boxes, output_size, 2]
    """

    batch, num_boxes, _ = boxes.shape

    box_grid_x = []
    box_grid_y = []
    for i in range(output_size):
        bx = boxes[:, :, 0] + (i + sample_offset) * boxes[:, :, 2] / output_size    
        box_grid_x.append(bx)
        
        by = boxes[:, :, 1] + (i + sample_offset) * boxes[:, :, 3] / output_size
        box_grid_y.append(by)
        
    box_grid_x = torch.stack(box_grid_x, axis=-1) ## T[2, 512, 7]
    box_grid_y = torch.stack(box_grid_y, axis=-1) ## T[2, 512, 7]
    ### map to feat index
    box_grid_x0 = torch.floor(box_grid_x)
    box_grid_x0 = torch.clamp(box_grid_x0, min=0)
    box_grid_y0 = torch.floor(box_grid_y)
    box_grid_y0 = torch.clamp(box_grid_y0, min=0)

    box_grid_x0 = torch.clamp(box_grid_x0,     max=boundaries[:, :, 0].unsqueeze(-1))
    box_grid_x1 = torch.clamp(box_grid_x0 + 1, max=boundaries[:, :, 0].unsqueeze(-1))
    box_grid_x0x1 = torch.stack([box_grid_x0, box_grid_x1], axis=-1)  ## T[B, C, 7, 2]

    box_grid_y0 = torch.clamp(box_grid_y0,     max=boundaries[:, :, 1].unsqueeze(-1))
    box_grid_y1 = torch.clamp(box_grid_y0 + 1, max=boundaries[:, :, 1].unsqueeze(-1))
    box_grid_y0y1 = torch.stack([box_grid_y0, box_grid_y1], axis=-1)  ## T[B, C, 7, 2]

    # The RoIAlign feature f can be computed by bilinear interpolation of four
    # neighboring feature points f0, f1, f2, and f3.
    # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
    #                       [f10, f11]]
    # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
    # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
    lx = box_grid_x - box_grid_x0
    ly = box_grid_y - box_grid_y0
    
    hx = 1.0 - lx
    hy = 1.0 - ly
    
    kernel_x = torch.stack([hx, lx], axis=-1)  ### T[B, C, 7, 2]
    kernel_y = torch.stack([hy, ly], axis=-1)

    return kernel_x, kernel_y, box_grid_x0x1, box_grid_y0y1,


def get_grid_one_hot(box_gridy0y1, box_gridx0x1, feature_height, feature_width):
    """Get grid_one_hot from indices and feature_size."""
    (batch_size, num_boxes, output_size, _) = box_gridx0x1.shape
    if batch_size is None:
        batch_size = torch.shape(box_gridx0x1)[0]

    y_indices = torch.reshape(
        box_gridy0y1, [batch_size, num_boxes, output_size, 2]
    ).to(torch.int32)
    x_indices = torch.reshape(
        box_gridx0x1, [batch_size, num_boxes, output_size, 2]
    ).to(torch.int32)

    # shape is [batch_size, num_boxes, output_size, 2, height]
    grid_y_one_hot = F.one_hot(y_indices, feature_height)
    # shape is [batch_size, num_boxes, output_size, 2, width]
    grid_x_one_hot = F.one_hot(x_indices, feature_width)

    return grid_y_one_hot, grid_x_one_hot

def scaled_assign_boxes_to_levels(box_tensor, level_ids):
    """
    Map each box in `box_lists` to a feature map level index and return the assignment
    vector.

    """
    # eps = 1e-7
    # box_sizes = torch.sqrt(torch.cat([boxes.area() for boxes in box_lists]))
    # # Eqn.(1) in FPN paper
    # level_assignments = torch.floor(
    #     canonical_level + torch.log2(box_sizes / canonical_box_size + eps)
    # )
    # # clamp level to (min, max), in case the box size is too large or too small
    # # for the available feature maps
    # level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    # return level_assignments.to(torch.int64) - min_level
    ori_box = box_tensor
    min_level = min(level_ids)
    max_level = max(level_ids)
    box_width  = box_tensor[:, :, 2] - box_tensor[:, :, 0]
    box_height = box_tensor[:, :, 3] - box_tensor[:, :, 1]
    areas = box_height * box_width
    # areas = torch.clamp(areas, min=0.0)
    areas_sqrt = torch.sqrt(areas)
    # Maps levels between [min_level, max_level].
    levels = torch.floor(torch.log(areas_sqrt / 224.0) / np.log(2.0)) + 4  # T[4, 512]
    levels = levels.clamp(min=min_level, max=max_level).to(torch.int32)
    scale_to_level = torch.pow(2.0, levels)
    # box_tensor = box_tensor / scale_to_level.unsqueeze(-1)
    box_width  = box_width  / scale_to_level
    box_height = box_height / scale_to_level
    box_v = [
        box_tensor[:, :, 0:2] / scale_to_level.unsqueeze(-1),
        torch.unsqueeze(box_width, -1),
        torch.unsqueeze(box_height, -1),
    ]
    new_box_tensor = torch.concatenate(box_v, dim=-1,)  ### XYWH
    # levels = torch.clamp(levels, min=min_level, max=max_level)
    levels = levels - min_level
    return new_box_tensor, levels

def convert_boxes_to_pooler_format(rois: "torch.Tensor"):
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
    (see description under Returns).
    """
    B, N, _ = rois.shape
    batch_indices = torch.arange(B).repeat_interleave(N).to(device=rois.device)

    # Combine batch indices with RoIs
    rois = rois.view(-1, 4)  # Flatten to [K, 4]
    batch_indices = batch_indices.unsqueeze(1)
    rois_with_batch = torch.cat([batch_indices, rois], dim=1)  # [K, 5]
    return rois_with_batch

from torchvision.ops import roi_align
def multilevel_crop_and_resize(features: "dict[str, torch.Tensor]", 
        box_tensor: "torch.Tensor", output_size=7,
        use_einsum_gather=False):
    """Crop and resize on multilevel feature pyramid.

    Generate the (output_size, output_size) set of pixels for each input box
    by first locating the box into the correct feature level, and then cropping
    and resizing it using the correspoding feature map of that level.

    Here is the step-by-step algorithm with use_einsum_gather=True:
    1. Compute sampling points and their four neighbors for each output points.
       Each box is mapped to [output_size, output_size] points.
       Each output point is averaged among #sampling_raitio^2 points.
       Each sampling point is computed using bilinear
       interpolation of its four neighboring points on the feature map.
    2. Gather output points separately for each level. Gather and computation of
       output points are done for the boxes mapped to this level only.
       2.1. Compute indices of four neighboring point of each sampling
            point for x and y separately of shape
            [batch_size, num_boxes, output_size, 2].
       2.2. Compute the interpolation kernel for axis x and y separately of
            shape [batch_size, num_boxes, output_size, 2, 1].
       2.3. The features are colleced into a
            [batch_size, num_boxes, output_size, output_size, num_filters]
            Tensor.
            Instead of a one-step algorithm, a two-step approach is used.
            That is, first, an intermediate output is stored with a shape of
            [batch_size, num_boxes, output_size, width, num_filters];
            second, the final output is produced with a shape of
            [batch_size, num_boxes, output_size, output_size, num_filters].

            Blinear interpolation is done during the two step gather:
            f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
                                  [f10, f11]]
            [[f00, f01],
             [f10, f11]] = jnp.einsum(jnp.einsum(features, y_one_hot), x_one_hot)
            where [hy, ly] and [hx, lx] are the bilinear interpolation kernel.

            Note:
              a. Use one_hot with einsum to replace gather;
              b. Bilinear interpolation and averaging of
                 multiple sampling points are fused into the one_hot vector.

    Args:
      features: A dictionary with key as pyramid level and value as features. The
        features are in shape of [batch_size, num_filters, height_l, width_l].
      boxes: A 3-D array of shape [batch_size, num_boxes, 4]. Each row represents
        a box with [y1, x1, y2, x2] in un-normalized coordinates.
      output_size: A scalar to indicate the output crop size.
      use_einsum_gather: use einsum to replace gather or not. Replacing einsum
        with gather can potentially improve performance.

    Returns:
      A 5-D array representing feature crop of shape
      [batch_size, num_boxes, output_size, output_size, num_filters].
    """
    level_ids = list(features.keys())  ### keys: 5,4,3,2,6
    min_level = min(level_ids)
    max_level = max(level_ids)
    batch_size, num_filters, max_feat_height, max_feat_width = features[min_level].shape
    device = box_tensor.device
    box_tensor, levels = scaled_assign_boxes_to_levels(box_tensor, level_ids)
    # Stack feature pyramid into a features_all of shape
    # [batch_size, levels, height, width, num_filters].
    
    features_all = []
    feat_heights = []
    feat_widths = []
    for level in range(min_level, max_level + 1):
        feat = features[level]
        shape = feat.shape  ## T[3, 256, 128/64, 128/64]
        feat_heights.append(shape[2])
        feat_widths.append(shape[3])
        # concatenate array of [batch_size, height_l * width_l, num_filters] for each levels.
        lvl_feat = feat.reshape(batch_size, num_filters, -1).permute(0, 2, 1) ## T[N, 16384/4096, 256,]
        features_all.append(lvl_feat)
        
    features_r2 = torch.concatenate(features_all, dim=1)  ## T[N, 21824, 256,]
    features_r2 = features_r2.reshape(-1, num_filters) ## T[109120 * N, 256]
    # Calculate height_l * width_l for each level.
    level_dim_sizes = [fw * fh for fw, fh in zip(feat_widths, feat_heights)]
    
    
    # level_dim_offsets is accumulated sum of level_dim_size.
    level_dim_offsets = np.cumsum([0] + level_dim_sizes[:-1])
    batch_dim_size = np.sum(level_dim_sizes)
    

    level_dim_offsets = torch.Tensor(level_dim_offsets).to(dtype=torch.int32, device=device)
    height_dim_sizes = torch.Tensor(feat_heights).to(dtype=torch.int32, device=device)
        
    level_strides = torch.pow(2.0, levels.to(torch.float32))
    # Maps levels to [0, max_level-min_level].
    boundary = [
        (max_feat_width  / level_strides) - 1,
        (max_feat_height / level_strides) - 1,
    ]
    boundary = torch.stack(boundary, dim=-1,) ### T[B, C, 2]
    # Compute grid positions.
    kernel_x, kernel_y, box_grid_x0x1, box_grid_y0y1 = compute_grid_positions(
        box_tensor, boundary, output_size, sample_offset=0.5
    )
    box_grid_x0x1 : "torch.Tensor"
    _, num_boxes, _ = box_tensor.shape
    double_output = output_size * 2
    x_indices = box_grid_x0x1.reshape(batch_size, num_boxes, double_output).to(torch.int32)
    y_indices = box_grid_y0y1.reshape(batch_size, num_boxes, double_output).to(torch.int32)
    batch_size_offset = (torch.arange(batch_size) * batch_dim_size).to(device)
    batch_size_offset = batch_size_offset.reshape(batch_size, 1, 1, 1)
    batch_size_offset = torch.tile(batch_size_offset,
        [1, num_boxes, double_output, double_output],
    )
    # Get level offset for each box. Each box belongs to one level.
    levels_offset = level_dim_offsets[levels].reshape(batch_size, num_boxes, 1, 1)
    levels_offset = torch.tile(levels_offset, [1, 1, double_output, double_output],) ## T[512, 14, 14]

    y_ind_off = y_indices * height_dim_sizes[levels].unsqueeze(-1)
    # y_ind_off = y_indices
    y_ind_off = y_ind_off.reshape(batch_size, num_boxes, double_output, 1)
    y_indices_offset = torch.tile(y_ind_off, [1, 1, 1, double_output,], ) ## T[512, 14, 14]
    
    # x_ind_off = x_indices * width_dim_sizes[levels].unsqueeze(-1)
    x_ind_off = x_indices
    x_ind_off = x_ind_off.reshape(batch_size, num_boxes, 1, double_output)
    x_indices_offset = torch.tile(x_ind_off, [1, 1, double_output, 1],)

    indices = batch_size_offset + levels_offset + y_indices_offset + x_indices_offset
    indices = indices.reshape(-1).to(torch.int32)
    ##
    feats_per_box = features_r2[indices]  ## T[N, 512, 14, 14, C]
    feats_per_box = feats_per_box.reshape(batch_size, num_boxes, double_output, double_output, num_filters,) ## T[19, 256, 14, 14, ]

    feats_per_box = feats_per_box.permute(0, 1, 4, 2, 3)  ## T[B, A, C, H, W]
    feats_per_box = feature_bilinear_interpolation(
        feats_per_box, kernel_y, kernel_x
    )  ## T[4, 256, 256, 7, 7, ]

    
    # feats_per_box = feats_per_box.reshape(-1, num_filters, output_size, output_size)
    return feats_per_box

def assign_boxes_to_levels(box_tensor, level_ids, canonical_box_size=224.0, canonical_level=4):
    """
    Map each box in `box_lists` to a feature map level index and return the assignment
    vector.

    """
    
    min_level = min(level_ids)
    max_level = max(level_ids)
    box_width  = box_tensor[:, :, 2] - box_tensor[:, :, 0]
    box_height = box_tensor[:, :, 3] - box_tensor[:, :, 1]
    areas = box_height * box_width
    # areas = torch.clamp(areas, min=0.0)
    areas_sqrt = torch.sqrt(areas)
    # Maps levels between [min_level, max_level].
    levels = torch.floor(torch.log(areas_sqrt / canonical_box_size) / np.log(2.0)) + canonical_level  # T[4, 512]
    levels = levels.clamp(min=min_level, max=max_level).to(torch.int32)

    # levels = torch.clamp(levels, min=min_level, max=max_level)
    levels = levels - min_level
    return levels

def roi_box_pooler(features: "dict[str, torch.Tensor]", box_tensor: "torch.Tensor", output_size=7,):
    """torchvision implementation of roi box pooler
    """
    level_ids = list(features.keys())  ### keys: 5,4,3,2,6
    min_level = min(level_ids)
    max_level = max(level_ids)
    batch_size, num_filters, max_feat_height, max_feat_width = features[min_level].shape
    device = box_tensor.device
    levels = assign_boxes_to_levels(box_tensor, level_ids)

    pooler_fmt_boxes: "torch.Tensor" = convert_boxes_to_pooler_format(box_tensor)
    levels = levels.reshape(-1)
    # box_features = []
    total_num_boxes = pooler_fmt_boxes.shape[0]
    box_outputs = torch.zeros(
        (total_num_boxes, num_filters, output_size, output_size), dtype=box_tensor.dtype, device=device
    ) # T[512, 256, 14, 14]
    for lvl in range(0, max_level - min_level + 1):
        feat = features[lvl + min_level]
        inds = torch.nonzero(levels == lvl).squeeze(1)
        if inds.shape == (0,): 
            continue
        pooler_fmt_boxes_lvl = pooler_fmt_boxes[inds]
        sscale = 1 / pow(2, lvl + min_level)
        feat = convert_fp32(feat)
        box_feature = roi_align(feat, pooler_fmt_boxes_lvl, output_size=(output_size, output_size), spatial_scale=sscale, aligned=True)
        box_outputs[inds] = box_feature
    box_outputs = box_outputs.reshape(batch_size, -1, num_filters, output_size, output_size)
    return box_outputs
    

def multilevel_crop_and_resize_einsum(features: "dict[str, torch.Tensor]", 
        box_list: "list[torch.Tensor]", 
        output_size=7):
    """Crop and resize on multilevel feature pyramid.

    Generate the (output_size, output_size) set of pixels for each input box
    by first locating the box into the correct feature level, and then cropping
    and resizing it using the correspoding feature map of that level.

    Here is the step-by-step algorithm with use_einsum_gather=True:
    1. Compute sampling points and their four neighbors for each output points.
       Each box is mapped to [output_size, output_size] points.
       Each output point is averaged among #sampling_raitio^2 points.
       Each sampling point is computed using bilinear
       interpolation of its four neighboring points on the feature map.
    2. Gather output points separately for each level. Gather and computation of
       output points are done for the boxes mapped to this level only.
       2.1. Compute indices of four neighboring point of each sampling
            point for x and y separately of shape
            [batch_size, num_boxes, output_size, 2].
       2.2. Compute the interpolation kernel for axis x and y separately of
            shape [batch_size, num_boxes, output_size, 2, 1].
       2.3. The features are colleced into a
            [batch_size, num_boxes, output_size, output_size, num_filters]
            Tensor.
            Instead of a one-step algorithm, a two-step approach is used.
            That is, first, an intermediate output is stored with a shape of
            [batch_size, num_boxes, output_size, width, num_filters];
            second, the final output is produced with a shape of
            [batch_size, num_boxes, output_size, output_size, num_filters].

            Blinear interpolation is done during the two step gather:
            f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
                                  [f10, f11]]
            [[f00, f01],
             [f10, f11]] = jnp.einsum(jnp.einsum(features, y_one_hot), x_one_hot)
            where [hy, ly] and [hx, lx] are the bilinear interpolation kernel.

            Note:
              a. Use one_hot with einsum to replace gather;
              b. Bilinear interpolation and averaging of
                 multiple sampling points are fused into the one_hot vector.

    Args:
      features: A dictionary with key as pyramid level and value as features. The
        features are in shape of [batch_size, height_l, width_l, num_filters].
      boxes: A 3-D array of shape [batch_size, num_boxes, 4]. Each row represents
        a box with [y1, x1, y2, x2] in un-normalized coordinates.
      output_size: A scalar to indicate the output crop size.
      use_einsum_gather: use einsum to replace gather or not. Replacing einsum
        with gather can potentially improve performance.

    Returns:
      A 5-D array representing feature crop of shape
      [batch_size, num_boxes, output_size, output_size, num_filters].
    """
    levels = list(features.keys())  ### keys: 5,4,3,2,6
    min_level = min(levels)
    max_level = max(levels)
    batch_size, num_filters, max_feature_height, max_feature_width = features[min_level].shape

    
        
    # batch_size = len(boxes)
    new_box_list = []
    device = box_list[0].device
    for bi in range(batch_size):
        boxes = box_list[bi]  ## T[N, 4]
        num_boxes, _ = boxes.shape

        # Assigns boxes to the right level.
        box_height = boxes[:, 3] - boxes[:, 1]
        box_width = boxes[:, 2] - boxes[:, 0]
        
        # Maps levels between [min_level, max_level].
        # levels = level_list[bi]
        # scale_to_level = torch.pow(2.0, levels.float())

        areas_sqrt = torch.sqrt(box_height * box_width)
        levels = torch.floor(torch.log(areas_sqrt / 224.0) / np.log(2.0)) + 4  # T[4, 512]


        # Projects box location and sizes to corresponding feature levels.
        scale_to_level = torch.pow(2.0, levels)
        boxes = boxes / scale_to_level.unsqueeze(dim=1)
        box_width = box_width / scale_to_level
        box_height = box_height / scale_to_level
        box_v = [
                boxes[:, 0:2],
                torch.unsqueeze(box_width, -1),
                torch.unsqueeze(box_height, -1),
            ]
        boxes = torch.concatenate(box_v, dim=-1,)
        new_box_list.append(boxes)

                
    box_list = new_box_list


    def two_step_gather_per_level(features_level, mask):
        """Performs two-step gather using einsum for every level of features."""
        (_, feature_height, feature_width, _) = features_level.shape
        boundaries = torch.tile(
            torch.unsqueeze(
                torch.unsqueeze(torch.Tensor([feature_height, feature_width]), 0),
                0,
            ),
            [batch_size, num_boxes, 1],
        )
        boundaries = boundaries.to(boxes.dtype)
        kernel_y, kernel_x, box_gridy0y1, box_gridx0x1 = compute_grid_positions(
            boxes, boundaries, output_size, sample_offset=0.5
        )

        # shape is:
        # [batch_size, num_boxes, output_size, 2, spatial_size]
        box_grid_y_one_hot, box_grid_x_one_hot = get_grid_one_hot(
            box_gridy0y1, box_gridx0x1, feature_height, feature_width
        )

        # # shape is [batch_size, num_boxes, output_size, spatial_size]
        box_grid_y_weight = torch.sum(
            torch.multiply(box_grid_y_one_hot, kernel_y), axis=-2
        )
        box_grid_x_weight = torch.sum(
            torch.multiply(box_grid_x_one_hot, kernel_x), axis=-2
        )

        # shape is [batch_size, num_boxes, output_size, width, feature]
        y_outputs = torch.einsum(
            "bhwf,bnyh->bnywf",
            features_level,
            box_grid_y_weight.to(features_level.dtype),
        )

        # shape is [batch_size, num_boxes, output_size, output_size, feature]
        x_outputs = torch.einsum(
            "bnywf,bnxw->bnyxf",
            y_outputs,
            box_grid_x_weight.to(features_level.dtype),
        )

        outputs = torch.where(
            torch.equal(mask, torch.zeros_like(mask)),
            torch.zeros_like(x_outputs),
            x_outputs,
        )
        return outputs

    features_per_box = torch.zeros(
        [batch_size, num_boxes, output_size, output_size, num_filters],
        dtype=features[min_level].dtype,
    )
    for level in range(min_level, max_level + 1):
        level_equal = torch.equal(levels, level)
        mask = torch.tile(
            torch.reshape(level_equal, [batch_size, num_boxes, 1, 1, 1]),
            [1, 1, output_size, output_size, num_filters],
        )
        features_per_box = features_per_box + two_step_gather_per_level(features[level], mask)

    return features_per_box


def selective_crop_and_resize(
    features: "torch.Tensor",
    boxes: "torch.Tensor",
    box_levels,
    boundaries,
    output_size=7,
    sample_offset=0.5,
    use_einsum_gather=False,
):
    """Crop and resize boxes on a set of feature maps.

    Given multiple features maps indexed by different levels, and a set of boxes
    where each box is mapped to a certain level, it selectively crops and resizes
    boxes from the corresponding feature maps to generate the box features.

    We follow the ROIAlign technique (see https://arxiv.org/pdf/1703.06870.pdf,
    figure 3 for reference). Specifically, for each feature map, we select an
    (output_size, output_size) set of pixels corresponding to the box location,
    and then use bilinear interpolation to select the feature value for each
    pixel.

    For performance, we perform the gather and interpolation on all layers as a
    single operation. In this op the multi-level features are first stacked and
    gathered into [2*output_size, 2*output_size] feature points. Then bilinear
    interpolation is performed on the gathered feature points to generate
    [output_size, output_size] RoIAlign feature map.

    Here is the step-by-step algorithm:
      1. The multi-level features are gathered into a
         [batch_size, num_boxes, output_size*2, output_size*2, num_filters]
         Tensor. The array contains four neighboring feature points for each
         vertex in the output grid.
      2. Compute the interpolation kernel of shape
         [batch_size, num_boxes, output_size*2, output_size*2]. The last 2 axis
         can be seen as stacking 2x2 interpolation kernels for all vertices in the
         output grid.
      3. Element-wise multiply the gathered features and interpolation kernel.
         Then apply 2x2 average pooling to reduce spatial dimension to
         output_size.

    Args:
      features: a 5-D array of shape [batch_size, num_levels, max_height,
        max_width, num_filters] where cropping and resizing are based.
      boxes: a 3-D array of shape [batch_size, num_boxes, 4] encoding the
        information of each box w.r.t. the corresponding feature map.
        boxes[:, :, 0:2] are the grid position in (y, x) (float) of the top-left
        corner of each box. boxes[:, :, 2:4] are the box sizes in (h, w) (float)
          in terms of the number of pixels of the corresponding feature map size.
      box_levels: a 3-D array of shape [batch_size, num_boxes, 1] representing
        the 0-based corresponding feature level index of each box.
      boundaries: a 3-D array of shape [batch_size, num_boxes, 2] representing
        the boundary (in (y, x)) of the corresponding feature map for each box.
        Any resampled grid points that go beyond the bounary will be clipped.
      output_size: a scalar indicating the output crop size.
      sample_offset: a float number in [0, 1] indicates the subpixel sample offset
        from grid point.
      use_einsum_gather: use einsum to replace gather or not. Replacing einsum
        with gather can improve performance when feature size is not large, einsum
        is friendly with model partition as well. Gather's performance is better
        when feature size is very large and there are multiple box levels.

    Returns:
      features_per_box: a 5-D array of shape
        [batch_size, num_boxes, output_size, output_size, num_filters]
        representing the cropped features.
    """
    (batch_size, num_levels, max_feature_height, max_feature_width, num_filters) = (
        features.shape
    )
    _, num_boxes, _ = boxes.shape
    kernel_y, kernel_x, box_gridy0y1, box_gridx0x1 = compute_grid_positions(
        boxes, boundaries, output_size, sample_offset
    )
    x_indices = torch.reshape(
        box_gridx0x1, [batch_size, num_boxes, output_size * 2]
    ).to(torch.int32)
    y_indices = torch.reshape(
        box_gridy0y1, [batch_size, num_boxes, output_size * 2]
    ).to(torch.int32)

    if use_einsum_gather:
        # Blinear interpolation is done during the last two gathers:
        #        f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
        #                              [f10, f11]]
        #        [[f00, f01],
        #         [f10, f11]] = torch.einsum(torch.einsum(features, y_one_hot),
        #                                  x_one_hot)
        #       where [hy, ly] and [hx, lx] are the bilinear interpolation kernel.

        # shape is [batch_size, boxes, output_size, 2, 1]
        grid_y_one_hot, grid_x_one_hot = get_grid_one_hot(
            box_gridy0y1, box_gridx0x1, max_feature_height, max_feature_width
        )

        # shape is [batch_size, num_boxes, output_size, height]
        grid_y_weight: "torch.Tensor" = torch.sum(torch.multiply(grid_y_one_hot, kernel_y), axis=-2)
        # shape is [batch_size, num_boxes, output_size, width]
        grid_x_weight = torch.sum(torch.multiply(grid_x_one_hot, kernel_x), axis=-2)

        # Gather for y_axis.
        # shape is [batch_size, num_boxes, output_size, width, features]
        features_per_box = torch.einsum(
            "bmhwf,bmoh->bmowf", features, grid_y_weight.to(features.dtype)
        )
        # Gather for x_axis.
        # shape is [batch_size, num_boxes, output_size, output_size, features]
        features_per_box = torch.einsum(
            "bmhwf,bmow->bmhof", features_per_box, grid_x_weight.to(features.dtype)
        )
    else:
        height_dim_offset = max_feature_width
        level_dim_offset = max_feature_height * height_dim_offset
        batch_dim_offset = num_levels * level_dim_offset

        batch_size_offset = torch.tile(
            torch.reshape(
                torch.arange(batch_size) * batch_dim_offset, [batch_size, 1, 1, 1]
            ),
            [1, num_boxes, output_size * 2, output_size * 2],
        )
        box_levels_offset = torch.tile(
            torch.reshape(box_levels * level_dim_offset, [batch_size, num_boxes, 1, 1]),
            [1, 1, output_size * 2, output_size * 2],
        )
        y_indices_offset = torch.tile(
            torch.reshape(
                y_indices * height_dim_offset,
                [batch_size, num_boxes, output_size * 2, 1],
            ),
            [1, 1, 1, output_size * 2],
        )
        x_indices_offset = torch.tile(
            torch.reshape(x_indices, [batch_size, num_boxes, 1, output_size * 2]),
            [1, 1, output_size * 2, 1],
        )

        indices = torch.reshape(
            batch_size_offset + box_levels_offset + y_indices_offset + x_indices_offset,
            [-1],
        )

        features = torch.reshape(features, [-1, num_filters])
        features_per_box = torch.reshape(
            features[indices],
            [batch_size, num_boxes, output_size * 2, output_size * 2, num_filters],
        )
        features_per_box = feature_bilinear_interpolation(
            features_per_box, kernel_y, kernel_x
        )

    return features_per_box


def crop_mask_in_target_box(
    masks: "torch.Tensor", boxes, target_boxes, output_size, sample_offset=0.0, use_einsum=True
):
    """Crop masks in target boxes.

    Args:
      masks: An array with a shape of [batch_size, num_masks, height, width].
      boxes: a float array representing box cooridnates that tightly enclose
        masks with a shape of [batch_size, num_masks, 4] in un-normalized
        coordinates. A box is represented by [ymin, xmin, ymax, xmax].
      target_boxes: a float array representing target box cooridnates for masks
        with a shape of [batch_size, num_masks, 4] in un-normalized coordinates. A
        box is represented by [ymin, xmin, ymax, xmax].
      output_size: A scalar to indicate the output crop size. It currently only
        supports to output a square shape outputs.
      sample_offset: a float number in [0, 1] indicates the subpixel sample offset
        from grid point.
      use_einsum: Use einsum to replace gather in selective_crop_and_resize.

    Returns:
      A 4-D array representing feature crop of shape
      [batch_size, num_boxes, output_size, output_size].
    """
    batch_size, num_masks, height, width = masks.shape
    # Pad zeros on the boundary of masks.
    pad_value = torch.Tensor(0.0, dtype=masks.dtype)
    masks = F.pad(masks, pad_value, [(0, 0, 0), (0, 0, 0), (2, 2, 0), (2, 2, 0)])
    masks = torch.reshape(masks, [batch_size, num_masks, height + 4, width + 4, 1])

    # Projects target box locations and sizes to corresponding cropped
    # mask coordinates.
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = torch.split(boxes, 4, axis=2)
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = torch.split(target_boxes, 4, axis=2)
    y_transform = (bb_y_min - gt_y_min) * height / (gt_y_max - gt_y_min + _EPSILON) + 2
    x_transform = (bb_x_min - gt_x_min) * height / (gt_x_max - gt_x_min + _EPSILON) + 2
    h_transform = (bb_y_max - bb_y_min) * width  / (gt_y_max - gt_y_min + _EPSILON)
    w_transform = (bb_x_max - bb_x_min) * width  / (gt_x_max - gt_x_min + _EPSILON)

    boundaries = torch.concatenate(
        [
            torch.ones_like(y_transform) * ((height + 4) - 1),
            torch.ones_like(x_transform) * ((width + 4) - 1),
        ],
        axis=-1,
    ).to(masks.dtype)

    # Reshape tensors to have the right shape for selective_crop_and_resize.
    transformed_boxes = torch.concatenate(
        [y_transform, x_transform, h_transform, w_transform], -1
    )
    levels = torch.tile(
        torch.reshape(torch.arange(num_masks), [1, num_masks]), [batch_size, 1]
    )

    cropped_masks = selective_crop_and_resize(
        masks,
        transformed_boxes,
        levels,
        boundaries,
        output_size,
        sample_offset=sample_offset,
        use_einsum_gather=use_einsum,
    )
    cropped_masks = torch.squeeze(cropped_masks, axis=-1)

    return cropped_masks
