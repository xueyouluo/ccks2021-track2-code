# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, hvd=None, amp=False, accumulation_step=1,freeze_bert=False, head_lr_ratio=1.0):
  """Creates an optimizer training op.
  
  Args:
    loss: training loss
    init_lr: initial learning rate
    num_train_steps: total training steps
    num_warmup_steps: warmup steps
    hvd: whether use hvd for distribute training
    amp: whether use auto-mix-precision to speed up training
    accumulation_step: gradient accumulation steps
    freeze_bert: whether to freeze bert variables
    head_lr_ratio: bert and head should have different learning rate
  """
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,#if not use_swa else init_lr/2,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      head_lr_ratio=head_lr_ratio,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  if hvd is not None:
    from horovod.tensorflow.compression import Compression
    optimizer = hvd.DistributedOptimizer(optimizer, sparse_as_dense = True, compression=Compression.fp16 if amp else Compression.none)
  
  if amp:
    loss_scaler = tf.train.experimental.DynamicLossScale(initial_loss_scale=2**32, increment_period=1000, multiplier=2.0)
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer, loss_scaler)
    loss_scale_value = tf.identity(loss_scaler(), name="loss_scale")

  tvars = tf.trainable_variables()
  if freeze_bert:
    tvars = [var for var in tvars if 'bert' not in var.name]
  grads_and_vars = optimizer.compute_gradients(loss, tvars)

  if accumulation_step > 1:
    tf.logging.info('### Using Gradient Accumulation with {} ###'.format(accumulation_step))

    local_step = tf.get_variable(name="local_step", shape=[], dtype=tf.int32, trainable=False,
                                   initializer=tf.zeros_initializer)
    batch_finite = tf.get_variable(name="batch_finite", shape=[], dtype=tf.bool, trainable=False,
                                    initializer=tf.ones_initializer)
    accum_vars = [tf.get_variable(
        name=tvar.name.split(":")[0] + "/accum",
        shape=tvar.shape.as_list(),
        dtype=tf.float32,
        trainable=False,
        initializer=tf.zeros_initializer()) for tvar in tf.trainable_variables()]

    reset_step = tf.cast(tf.math.equal(local_step % accumulation_step, 0), dtype=tf.bool)
    local_step = tf.cond(reset_step, lambda:local_step.assign(tf.ones_like(local_step)), lambda:local_step.assign_add(1))

    grads_and_vars_and_accums = [(gv[0],gv[1],accum_vars[i]) for i, gv in enumerate(grads_and_vars) if gv[0] is not None]
    grads, tvars, accum_vars = list(zip(*grads_and_vars_and_accums))

    all_are_finite = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g in grads]) if amp else tf.constant(True, dtype=tf.bool)
    batch_finite = tf.cond(reset_step,
      lambda: batch_finite.assign(tf.math.logical_and(tf.constant(True, dtype=tf.bool), all_are_finite)),
      lambda: batch_finite.assign(tf.math.logical_and(batch_finite, all_are_finite)))

    # This is how the model was pre-trained.
    # ensure global norm is a finite number
    # to prevent clip_by_global_norm from having a hizzy fit.
    (clipped_grads, _) = tf.clip_by_global_norm(
          grads, clip_norm=1.0,
          use_norm=tf.cond(
              all_are_finite,
              lambda: tf.global_norm(grads),
              lambda: tf.constant(1.0)))

    accum_vars = tf.cond(reset_step,
            lambda: [accum_vars[i].assign(grad) for i, grad in enumerate(clipped_grads)],
            lambda: [accum_vars[i].assign_add(grad) for i, grad in enumerate(clipped_grads)])

    def update(accum_vars):
      return optimizer.apply_gradients(list(zip(accum_vars, tvars)))

    update_step = tf.identity(tf.cast(tf.math.equal(local_step % accumulation_step, 0), dtype=tf.bool), name="update_step")
    update_op = tf.cond(update_step,
                        lambda: update(accum_vars), lambda: tf.no_op())

    new_global_step = tf.cond(tf.math.logical_and(update_step, 
                                                  tf.cast(hvd.allreduce(tf.cast(batch_finite, tf.int32)), tf.bool) if hvd is not None else batch_finite),
                              lambda: global_step+1,
                              lambda: global_step)
    new_global_step = tf.identity(new_global_step, name='step_update')
    train_op = tf.group(update_op, [global_step.assign(new_global_step)])
  else:
    grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
    grads, tvars = list(zip(*grads_and_vars))
    all_are_finite = tf.reduce_all(
        [tf.reduce_all(tf.is_finite(g)) for g in grads]) if amp else tf.constant(True, dtype=tf.bool)

    # This is how the model was pre-trained.
    # ensure global norm is a finite number
    # to prevent clip_by_global_norm from having a hizzy fit.
    (clipped_grads, _) = tf.clip_by_global_norm(
        grads, clip_norm=1.0,
        use_norm=tf.cond(
            all_are_finite,
            lambda: tf.global_norm(grads),
            lambda: tf.constant(1.0)))

    train_op = optimizer.apply_gradients(
        list(zip(clipped_grads, tvars)))

    new_global_step = tf.cond(all_are_finite, lambda: global_step + 1, lambda: global_step)
    new_global_step = tf.identity(new_global_step, name='step_update')
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op, learning_rate

class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               head_lr_ratio=1.0,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.head_lr_ratio = head_lr_ratio

  def _apply_gradients(self, grads_and_vars, learning_rate):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))
      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self.weight_decay_rate > 0:
        if self._do_use_weight_decay(param_name):
          update += self.weight_decay_rate * param

      update_with_lr = learning_rate * update
      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])

    return assignments

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    if self.head_lr_ratio > 1.0:
      def is_backbone(n):
        return 'bert' in n
      assignments = []
      backbone_gvs = []
      head_gvs = []
      for grad,var in grads_and_vars:
        if is_backbone(var.name):
          backbone_gvs.append((grad,var))
        else:
          head_gvs.append((grad,var))
      assignments += self._apply_gradients(backbone_gvs,self.learning_rate)
      assignments += self._apply_gradients(head_gvs,self.learning_rate * self.head_lr_ratio)
    else:
      assignments = self._apply_gradients(grads_and_vars, self.learning_rate)
    return tf.group(*assignments,name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
