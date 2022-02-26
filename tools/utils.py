# Copyright 2021 Google Research. All Rights Reserved.
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
# ==============================================================================
"""Model utilities."""
import contextlib
import functools
import os
from absl import logging
import paddle as pd
import numpy as np
import math
def activation_fn(features, act_fn):
  """Customized non-linear activation type."""
  if act_fn in ('silu', 'swish'):
    return pd.nn.functional.swish(features)
  elif act_fn == 'silu_native':
    return features * pd.nn.functional.sigmoid(features)
  elif act_fn == 'hswish':
    return features * pd.nn.functional.relu6(features + 3) / 6
  elif act_fn == 'relu':
    return pd.nn.functional.relu(features)
  elif act_fn == 'relu6':
    return pd.nn.functional.relu6(features)
  elif act_fn == 'elu':
    return pd.nn.functional.elu(features)
  elif act_fn == 'leaky_relu':
    return pd.nn.functional.leaky_relu(features)
  elif act_fn == 'selu':
    return pd.nn.functional.selu(features)
  elif act_fn == 'mish':    
    return features * pd.tanh(pd.nn.functional.softplus(features))
  else:
    raise ValueError('Unsupported act_fn {}'.format(act_fn))

def get_act_fn(act_fn):
  if not act_fn:
    return pd.nn.functional.silu
  if isinstance(act_fn, str):
    return functools.partial(activation_fn, act_fn=act_fn)
  return act_fn

def build_optimizer(learning_rate,
                    optimizer_name='rmsprop',
                    decay=0.9,
                    epsilon=0.001,
                    momentum=0.9):
  """Build optimizer."""
  if optimizer_name == 'sgd':
    logging.info('Using SGD optimizer')
    optimizer = pd.optimizer.SGD(learning_rate=learning_rate)
  elif optimizer_name == 'momentum':
    logging.info('Using Momentum optimizer')
    optimizer = pd.optimizer.Momentum(learning_rate=learning_rate, momentum=momentum)
  elif optimizer_name == 'rmsprop':
    logging.info('Using RMSProp optimizer')
    optimizer = pd.optimizer.RMSProp(learning_rate=learning_rate, weight_decay=decay, momentum=momentum, epsilon=epsilon)
  elif optimizer_name == 'adam':
    logging.info('Using Adam optimizer')
    optimizer = pd.optimizer.Adam(learning_rate)
  else:
    logging.fatal('Unknown optimizer: %s', optimizer_name)

  return optimizer

def normalization(norm_type: str,
                  axis=-1,
                  epsilon=0.001,
                  momentum=0.99,
                  groups=8,
                  name=None):
  """Normalization after conv layers."""
  if norm_type == 'gn':
    return pd.nn.GroupNorm(num_groups=groups, num_channels=axis, epsilon=epsilon)
  else:
    return pd.nn.BatchNorm(num_channels=axis, momentum=momentum, epsilon=epsilon)

def drop_connect(inputs, is_training, survival_prob):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not is_training:
    return inputs

  # Compute tensor.
  batch_size = pd.shape(inputs)[0]
  random_tensor = survival_prob
  random_tensor += pd.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = pd.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = inputs / survival_prob * binary_tensor
  return output


class Pair(tuple):

  def __new__(cls, name, value):
    return super().__new__(cls, (name, value))

  def __init__(self, name, _):  # pylint: disable=super-init-not-called
    self.name = name

@contextlib.contextmanager
def float16_scope():
  """Scope class for float16."""

  def _custom_getter(getter, *args, **kwargs):
    """Returns a custom getter that methods must be called under."""
    cast_to_float16 = False
    requested_dtype = kwargs['dtype']
    if requested_dtype == pd.float16:
      kwargs['dtype'] = pd.float32
      cast_to_float16 = True
    var = getter(*args, **kwargs)
    if cast_to_float16:
      var = pd.cast(var, pd.float16)
    return var

def round_filters(filters, mconfig, skip=False):
  """Round number of filters based on depth multiplier."""
  multiplier = mconfig.width_coefficient
  divisor = mconfig.depth_divisor
  min_depth = mconfig.min_depth
  if skip or not multiplier:
    return filters
  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
    new_filters += divisor
  return int(new_filters)


def round_repeats(repeats, multiplier, skip=False):
  """Round number of filters based on depth multiplier."""
  if skip or not multiplier:
    return repeats
  return int(math.ceil(multiplier * repeats))

def cal_padding(img_size, stride, filter_size, dilation=1):
    """Calculate padding size."""
    if img_size % stride == 0:
        out_size = max(filter_size - stride, 0)
    else:
        out_size = max(filter_size - (img_size % stride), 0)
    return out_size // 2, out_size - out_size // 2

def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """定义卷积核初始化.
  调用paddle.normal生成符合正态分布的随机Tensor
  """
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape
  fan_out = int(kernel_height * kernel_width * out_filters)
  return pd.normal(
      shape=shape, mean=0.0, std=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
  """稠密dense层初始化
  调用paddle.uniform实现均匀分布
  """
  del partition_info
  init_range = 1.0 / np.sqrt(shape[1])
  return pd.uniform(shape, min=-init_range, max=init_range, dtype=dtype)