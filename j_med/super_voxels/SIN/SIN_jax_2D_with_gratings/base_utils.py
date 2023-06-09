
from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import torchio
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import jax
import tensorflow as tf
import torch 
import einops
import torchio as tio
import optax
from flax.training import train_state  
from torch.utils.data import DataLoader
import jax.profiler
import ml_collections
from ml_collections import config_dict
from functools import partial
import toolz
import chex
from typing import (Any, Callable, Iterable, Optional, Tuple, Union)
from flax.linen.dtypes import canonicalize_dtype

from flax.linen.module import Module, compact, merge_param  # pylint: disable=g-multiple-import
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp


PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?

Axes = Union[int, Any]




def _canonicalize_axes(rank: int, axes: Axes) -> Tuple[int, ...]:
  """Returns a tuple of deduplicated, sorted, and positive axes."""
  if not isinstance(axes, Iterable):
    axes = (axes,)
  return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))


def _abs_sq(x):
  """Computes the elementwise square of the absolute value |x|^2."""
  if jnp.iscomplexobj(x):
    return lax.square(lax.real(x)) + lax.square(lax.imag(x))
  else:
    return lax.square(x)


def _compute_stats(x: Array, axes: Optional[Axes],
                   dtype: Optional[Dtype],
                   axis_name: Optional[str] = None,
                   axis_index_groups: Any = None,
                   use_mean: bool = True):
  """Computes mean and variance statistics.

  This implementation takes care of a few important details:
  - Computes in float32 precision for stability in half precision training.
  - mean and variance are computable in a single XLA fusion,
    by using Var = E[|x|^2] - |E[x]|^2 instead of Var = E[|x - E[x]|^2]).
  - Clips negative variances to zero which can happen due to
    roundoff errors. This avoids downstream NaNs.
  - Supports averaging across a parallel axis and subgroups of a parallel axis
    with a single `lax.pmean` call to avoid latency.

  Arguments:
    x: Input array.
    axes: The axes in ``x`` to compute mean and variance statistics for.
    dtype: Optional dtype specifying the minimal precision. Statistics
      are always at least float32 for stability (default: dtype of x).
    axis_name: Optional name for the pmapped axis to compute mean over.
    axis_index_groups: Optional axis indices.
    use_mean: If true, calculate the mean from the input and use it when
      computing the variance. If false, set the mean to zero and compute
      the variance without subtracting the mean.

  Returns:
    A pair ``(mean, var)``.
  """
  if dtype is None:
    dtype = jnp.result_type(x)
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  dtype = jnp.promote_types(dtype, jnp.float32)
  x = jnp.asarray(x, dtype)

  mean2 = jnp.mean(_abs_sq(x), axes)
  if use_mean:
    mean = jnp.mean(x, axes)
  else:
    mean = jnp.zeros(mean2.shape, dtype=dtype)

  if axis_name is not None:
    concatenated_mean = jnp.concatenate([mean, mean2])
    mean, mean2 = jnp.split(
        lax.pmean(
            concatenated_mean,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups), 2)
  # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
  # to floating point round-off errors.
  var = jnp.maximum(0., mean2 - _abs_sq(mean))
  return mean, var


def _normalize(mdl: Module, x: Array, mean: Array, var: Array,
               reduction_axes: Axes, feature_axes: Axes,
               dtype: Dtype, param_dtype: Dtype,
               epsilon: float,
               use_bias: bool, use_scale: bool,
               bias_init: Callable[[PRNGKey, Shape, Dtype], Array],
               scale_init: Callable[[PRNGKey, Shape, Dtype], Array]):
  """"Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

  Arguments:
    mdl: Module to apply the normalization in (normalization params will reside
      in this module).
    x: The input.
    mean: Mean to use for normalization.
    var: Variance to use for normalization.
    reduction_axes: The axes in ``x`` to reduce.
    feature_axes: Axes containing features. A separate bias and scale is learned
      for each specified feature.
    dtype: The dtype of the result (default: infer from input and params).
    param_dtype: The dtype of the parameters.
    epsilon: Normalization epsilon.
    use_bias: If true, add a bias term to the output.
    use_scale: If true, scale the output.
    bias_init: Initialization function for the bias term.
    scale_init: Initialization function for the scaling function.

  Returns:
    The normalized input.
  """
  reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
  feature_axes = _canonicalize_axes(x.ndim, feature_axes)
  stats_shape = list(x.shape)
  for axis in reduction_axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  feature_shape = [1] * x.ndim
  reduced_feature_shape = []
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
    reduced_feature_shape.append(x.shape[ax])
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  args = [x]
  if use_scale:
    scale = mdl.param('scale', scale_init, reduced_feature_shape,
                      param_dtype).reshape(feature_shape)
    mul *= scale
    args.append(scale)
  y *= mul
  if use_bias:
    bias = mdl.param('bias', bias_init, reduced_feature_shape,
                     param_dtype).reshape(feature_shape)
    y += bias
    args.append(bias)
  dtype = canonicalize_dtype(*args, dtype=dtype)
  return jnp.asarray(y, dtype)


class LayerNorm(Module):
  """Layer normalization (https://arxiv.org/abs/1607.06450).

  LayerNorm normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  i.e. applies a transformation that maintains the mean activation within
  each example close to 0 and the activation standard deviation close to 1.

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    reduction_axes: Axes for computing normalization statistics.
    feature_axes: Feature axes for learned bias and scaling.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
      This is only needed if the model is subdivided across devices, i.e. the
      array being normalized is sharded across devices within a pmap.
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over
      the examples on the first two and last two devices. See `jax.lax.psum`
      for more details.
  """
  epsilon: float = 1e-6
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
  reduction_axes: Axes = -1
  feature_axes: Axes = -1
  axis_name: Optional[str] = None
  axis_index_groups: Any = None

  @compact
  def __call__(self, x):
    """Applies layer normalization on the input.

    Args:
      x: the inputs

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    mean, var = _compute_stats(x, self.reduction_axes, self.dtype,
                               self.axis_name, self.axis_index_groups)

    return _normalize(
        self, x, mean, var, self.reduction_axes, self.feature_axes,
        self.dtype, self.param_dtype, self.epsilon,
        self.use_bias, self.use_scale,
        self.bias_init, self.scale_init)
