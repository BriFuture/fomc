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

"""Base losses in jax."""

import enum
from typing import Callable, Tuple, Union, Dict, Text, Any, Sequence

import torch
import torch.nn.functional as F
import numpy as np
from gdet.utils import safe_divide

Array = torch.Tensor
ArrayDict = Dict[Text, Array]
LossArray = Union[Array, ArrayDict]


@enum.unique
class LossReductionType(enum.IntEnum):
    """Reduction type for the loss as defined in TF."""

    MEAN = 0
    SUM = 1
    SUM_BY_NONZERO_WEIGHTS = 2
    NONE = 3
    RETURN_AS_IS = 4


EPSILON = 1e-7
_SIGMOID_EPSILON = 1e-20


def compute_weighted_loss(loss: "torch.Tensor", weights: "torch.Tensor", loss_reduction,):
    """Weights and reduces the loss.

    We convert to float32 before reducing following TF1 implementation.

    After weighting and reducing the losses, we convert the output back to the
    dtype of the input.

    Args:
      loss: an array of loss.
      weights: An array or scalar which must be broadcastable to logits and labels
        shape.
      dtype: loss output data type.
      loss_reduction: A loss reduction method as in the Tensorflow implementation.
        Currently supports SUM_BY_NONZERO_WEIGHTS, MEAN and SUM. Raises
        NotImplementedError if other values are provided.

    Returns:
      loss: a scalar of weighted and reduced loss.

    Raises:
      NotImplementedError: loss reduction type is undefined.
    """
    if loss_reduction == LossReductionType.RETURN_AS_IS:
        # Handle no loss reduction, by returning tensor as-is.
        return loss
    loss = loss.type(torch.float32)
    loss_weight = torch.broadcast_to(weights, loss.shape).type(torch.float32)
    loss = loss * loss_weight
    total_loss = torch.sum(loss)

    if loss_reduction == LossReductionType.SUM_BY_NONZERO_WEIGHTS:
        total_loss = safe_divide(total_loss, torch.sum(loss_weight != 0.0).float())
    elif loss_reduction == LossReductionType.MEAN:
        total_loss = safe_divide(total_loss, torch.sum(loss_weight).float())
    elif loss_reduction != LossReductionType.SUM:
        raise NotImplementedError(
            "LossReductionType not supported for this loss:" f"{loss_reduction}."
        )

    return total_loss



def sigmoid_cross_entropy(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    weights=1.0,
    loss_reduction=LossReductionType.SUM_BY_NONZERO_WEIGHTS,
    **kwargs,
):
    """Returns the sigmoid cross entropy loss.

    Implements:
    loss = label * (-1) * log(pred) + (1 — label) * (-1) * log(1 — pred).

    Please note: the default for TF is SUM_BY_NONZERO_WEIGHTS loss reduction.

    Args:
      logits: An array of shape of [batch, ..., num_classes].
      labels: An array of shape of [batch, ..., num_classes].
      weights: An array or scalar which must be broadcastable to logits and labels
        shape.
      loss_reduction: A loss reduction method as in the Tensorflow implementation.
        Currently supports SUM_BY_NONZERO_WEIGHTS, MEAN and SUM. Raises
        NotImplementedError if other values are provided.
      **kwargs: additional keyword arguments.

    Returns:
      A scalar loss
    """
    
    labels = labels.type(logits.dtype)
    logits = F.logsigmoid(logits)
    loss = -labels * logits - (1.0 - labels) * torch.log(
        torch.clamp(-torch.expm1(logits), min=_SIGMOID_EPSILON)
    )

    return compute_weighted_loss(loss, weights, loss_reduction)


def softmax_cross_entropy(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    label_smoothing=0.0,
    weights=1.0,
    loss_reduction=LossReductionType.SUM_BY_NONZERO_WEIGHTS,
    **kwargs,
):
    """Returns the softmax cross entropy loss.

    Please note: the default for TF is SUM_BY_NONZERO_WEIGHTS loss reduction.

    Args:
      logits: An array of shape of [batch, ..., num_classes].
      labels: An array of shape of [batch, ..., num_classes] of values betwwen [0,
        1]
      label_smoothing: how much label smoothing to apply, which smoothes out the
        label matrix. The new labels will be (1 - label_smoothing) * labels +
        label_smoothing / num_classes
      weights: A scalar or an array of shape [batch] for weighting the loss per
        example.
      loss_reduction: A loss reduction method as in the Tensorflow implementation.
        Currently supports SUM_BY_NONZERO_WEIGHTS, MEAN and SUM. Raises
        NotImplementedError if other values are provided.
      **kwargs: additional keyword arguments.

    Returns:
      A scalar loss.
    """
    del kwargs

    labels = labels.type(logits.dtype)
    if label_smoothing > 0:
        num_classes = labels.shape[-1]
        smooth_weight = label_smoothing / num_classes
        smooth_weight = np.array(smooth_weight, dtype=logits.dtype)
        labels = (1.0 - label_smoothing) * labels + smooth_weight

    logits = F.log_softmax(logits)
    loss = -labels * logits
    loss = np.sum(loss, axis=-1)

    return compute_weighted_loss(loss, weights, loss_reduction)


def weighted_softmax_cross_entropy(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    label_smoothing=0.0,
    weights=1.0,
    loss_reduction=LossReductionType.SUM_BY_NONZERO_WEIGHTS,
    background_weight=1.0,
    **kwargs,
):
    """Returns the softmax cross entropy loss with background loss adjustment.

    Please note: the default for TF is SUM_BY_NONZERO_WEIGHTS loss reduction.

    Args:
      logits: An array of shape of [batch, ..., num_classes].
      labels: An array of shape of [batch, ..., num_classes] of values betwwen [0,
        1]
      label_smoothing: how much label smoothing to apply, which smoothes out the
        label matrix. The new labels will be (1 - label_smoothing) * labels +
        label_smoothing / num_classes
      weights: A scalar or an array of shape [batch] for weighting the loss per
        example.
      loss_reduction: A loss reduction method as in the Tensorflow implementation.
        Currently supports SUM_BY_NONZERO_WEIGHTS, MEAN and SUM. Raises
        NotImplementedError if other values are provided.
      background_weight: A float to adjust the weights of background. Default
        1.0 is a no-op.
      **kwargs: additional keyword arguments.

    Returns:
      A scalar loss.
    """
    del kwargs

    labels = labels.type(logits.dtype)
    if label_smoothing > 0:
        num_classes = labels.shape[-1]
        smooth_weight = label_smoothing / num_classes
        smooth_weight = np.array(smooth_weight, dtype=logits.dtype)
        labels = (1.0 - label_smoothing) * labels + smooth_weight

    logits = F.log_softmax(logits)
    loss = -labels * logits

    # Apply background class weights
    class_weights = np.ones(loss.shape)
    class_weights[Ellipsis, :1] = background_weight  # Background is class 0.
    loss = loss * np.array(class_weights)

    loss = np.sum(loss, axis=-1)
    return compute_weighted_loss(loss, weights, loss_reduction)


def onehot_cross_entropy_loss(
    logits: "torch.Tensor", labels: "torch.Tensor", loss_reduction=LossReductionType.SUM_BY_NONZERO_WEIGHTS, **kwargs
):
    """Computes the cross entropy loss between logits and the actual labels.

    Converts the labels into one hot and calls softmax_cross_entropy function to
    compute the loss

    Args:
      logits: A float array representing the class prediction for each box with a
        shape of [batch_size, num_tokens, num_classes].
      labels: A float array representing int label for each token [batch_size,
        num_tokens]
      loss_reduction: A loss reduction method as in the Tensorflow implementation.
      **kwargs: additional keyword arguments.

    Returns:
      loss: A scalar representing total loss.
    """
    del kwargs
    vocab_size = logits.shape[-1]
    labels_one_hot = F.one_hot(labels.type(np.int32), vocab_size)
    weights = torch.where(labels > 0, 1, 0)
    return softmax_cross_entropy(
        logits, labels_one_hot, weights=weights, loss_reduction=loss_reduction
    )


def l1_loss(
    predictions: "torch.Tensor",
    labels: "torch.Tensor",
    weights=1.0,
    loss_reduction=LossReductionType.SUM_BY_NONZERO_WEIGHTS,
    **kwargs,
):
    """L1 loss.

    Args:
      predictions: an array of shape [batch, ..., d] containing model predictions.
      labels: an array of shape [batch, ..., d] containing ground truth.
      weights: A scalar or an array of shape [batch, ...] for weighting the loss
        per example.
      loss_reduction: a loss reduction method.
      **kwargs: additional keyword arguments.

    Returns:
      the L1 loss averaged over batch.
    """
    del kwargs  # Unused
    l1 = torch.sum(torch.abs(predictions - labels), axis=-1)
    loss = compute_weighted_loss(
        l1, weights=weights, loss_reduction=loss_reduction
    )
    return loss


def l2_loss(
    predictions: "torch.Tensor",
    labels: "torch.Tensor",
    weights=1.0,
    loss_reduction=LossReductionType.SUM_BY_NONZERO_WEIGHTS,
    **kwargs,
):
    """L2 loss.

    Args:
      predictions: An array of shape [batch, ..., d] containing model predictions.
      labels: An array of shape [batch, ..., d] containing ground truth.
      weights: A scalar or an array of shape [batch, ...] for weighting the loss
        per example.
      loss_reduction: A loss reduction method.
      **kwargs: additional keyword arguments.

    Returns:
      the L2 loss averaged over batch.
    """
    del kwargs  # Unused
    l2 = torch.sum(torch.square(predictions - labels), axis=-1)
    loss = compute_weighted_loss(
        l2, weights=weights, loss_reduction=loss_reduction,
    )
    return loss


def cosine_loss(
    predictions: "torch.Tensor",
    labels: "torch.Tensor",
    weights=1.0,
    loss_reduction=LossReductionType.SUM_BY_NONZERO_WEIGHTS,
    **kwargs,
):
    """Cosine loss.

    This loss computes the dot product between predictions and labels as loss.
    The value ranges from [0, 2.0] depending on the alignment of prediction and
    label vectors. This loss can be used when we want to optimize the alignment
    of the vectors directly.

    Args:
      predictions: An array of shape [batch, ..., d] containing model predictions.
        The predictions need to be normalized in the last dimension.
      labels: An array of shape [batch, ..., d] containing ground truth.
        The labels need to be normalized in the last dimension.
      weights: A scalar or an array of shape [batch, ...] for weighting the loss
        per example.
      loss_reduction: A loss reduction method.
      **kwargs: additional keyword arguments.

    Returns:
      The cosine loss averaged over batch.
    """
    del kwargs  # Unused
    cosine = 1.0 - torch.sum(predictions * labels, axis=-1)
    loss = compute_weighted_loss(
        cosine, weights=weights, loss_reduction=loss_reduction
    )
    return loss


def huber_loss(
    predictions: "torch.Tensor",
    labels: "torch.Tensor",
    weights=1.0,
    delta=1.0,
    loss_reduction=LossReductionType.SUM_BY_NONZERO_WEIGHTS,
    **kwargs,
):
    """Returns the Huber loss.

    Huber loss is computed as:
      0.5 * x^2                  if |x| <= d
      0.5 * d^2 + d * (|x| - d)  if |x| > d
    where x is the difference between labels and predictions.

    Args:
      predictions: An array of shape of [batch, num_channels].
      labels: An array of shape of [batch, num_channels].
      weights: A scalar or an array of shape [batch] for weighting the loss per
        example.
      delta: A range at which the function changes from quadratic to linear.
      loss_reduction: A loss reduction method as in the Tensorflow implementation.
        Currently supports SUM_BY_NONZERO_WEIGHTS, MEAN and SUM. Raises
        NotImplementedError if other values are provided.
      **kwargs: additional keyword arguments.

    Returns:
      A scalar loss.
    """
    del kwargs
    labels = labels.type(predictions.dtype)
    x = labels - predictions

    # Apply the formula above.
    loss = torch.where(
        torch.abs(x) <= delta,
        0.5 * torch.square(x),
        0.5 * delta * delta + delta * (torch.abs(x) - delta),
    )

    return compute_weighted_loss(loss, weights, loss_reduction)


def weight_decay_loss_wrapper(
    loss_fn,
    factor,
    exclude=(),
):
    """A wrapper to add weight decay to underlying loss function.

    Use this wrapper if the weight decay in the optimizer is not suitable. For
    example, if you need to exclude some parameters from decay loss.

    Args:
      loss_fn: The underlying loss function which accepts two args: outputs - A
        dictionary of outputs. labels - A dictionary of groundtruth labels.
      factor: A floating point to specify weight decay factor.
      exclude: A sequence of strings to use to filter out parameters.

    Returns:
      The wrapped loss function with weight decay added which accepts three args:
        outputs - A dictionary of outputs.
        labels - A dictionary of groundtruth labels.
        params - A frozen dictionary of parameters (pytree).
        **kwargs - Any additional arguments.
    """
    traversal = traverse_util.ModelParamTraversal(
        lambda path, _: all([e not in path for e in exclude])
    )

    def wrapped_loss(outputs, *args, params, **kwargs):
        losses = loss_fn(outputs, *args, **kwargs)
        weight_decay_params = list(traversal.iterate(params))
        weight_l2 = sum([torch.sum(x**2) for x in weight_decay_params])
        weight_penalty = factor * 0.5 * weight_l2

        if isinstance(losses, dict):
            if "model_loss" not in losses:
                raise ValueError(
                    "Losses must contain `model_loss` key as total model loss."
                )
            losses["pre_weight_penalty_model_loss"] = losses["model_loss"]
            losses["model_loss"] = losses["model_loss"] + weight_penalty
            losses["l2_regularization_loss"] = weight_penalty
        elif isinstance(losses, np.ndarray):
            losses = losses + weight_penalty
        else:
            raise ValueError("Encountered invalid loss type: ", type(losses))

        return losses

    return wrapped_loss
