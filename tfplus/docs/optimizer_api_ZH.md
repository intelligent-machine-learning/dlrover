# Adaptive Optimizers with Sparse Group Lasso
<div id="top" align="center">

   | [English](./optimizer_api.md) | [中文](./optimizer_api_ZH.md) |

</div>

我们展示 TFPlus 带有 Sparse Group Lasso 功能优化器的使用方法，详细原理见 [Adaptive Optimizers with Sparse Group Lasso for Neural Networks in CTR Prediction](https://arxiv.org/abs/2107.14432)，ECML PKDD '21。

Lasso 和 Group Lasso 可以用来对模型进行稀疏化压缩，自动选择重要特征。我们提出基于 Sparse Group Lasso 的通用特征选择框架，对大部分常用优化器加入了 Sparse Group Lasso 功能，在蚂蚁的在线学习以及离线训练场景被广泛使用。目前开源的优化器包括 Group Adam 和 Sparse Group Ftrl (Group AdaGrad)，分别对应 Adam 和 AdaGrad。

## 使用示例
```python
import tfplus

l1 = 1.0E-5
l2 = 1.0E-5
l21 = 1.0E-5

# Group Adam
opt = tfplus.train.GroupAdamOptimizer(
    		       learning_rate=1e-4,
               initial_accumulator_value=0.0,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               l1_regularization_strength=l1,
               l2_regularization_strength=l2,
               l21_regularization_strength=l21,
               use_locking=False,
               name="GroupAdam",
               accum_name=None,
               linear_name=None,
               version=4)
"""Construct a new Group Adam optimizer.

    Args:
      learning_rate: A float value or a constant float `Tensor`.
      initial_accumulator_value: The starting value for accumulators.
        Only zero or positive values are allowed.
      beta1: A float value or a constant float tensor.
        The exponential decay rate for the 1st moment estimates.
      beta2: A float value or a constant float tensor.
        The exponential decay rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
      l1_regularization_strength: A float value, must be greater than or
        equal to zero.
      l2_regularization_strength: A float value, must be greater than or
        equal to zero.
      l21_regularization_strength: A float value, must be greater than or
        equal to zero.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "GroupAdam".
      accum_name: The suffix for the variable that keeps the gradient squared
        accumulator.  If not present, defaults to name.
      linear_name: The suffix for the variable that keeps the linear gradient
        accumulator.  If not present, defaults to name + "_1".
      version: the specific version of GroupAdam.

    Raises:
      ValueError: If one of the arguments is invalid.
    """

# Sparse Group FTRL
opt = tfplus.train.SparseGroupFtrlOptimizer(
               learning_rate=1e-2,
               learning_rate_power=-0.5,
               initial_accumulator_value=0.1,
               l1_regularization_strength=l1,
               l2_regularization_strength=l2,
               l21_regularization_strength=l3,
               use_locking=False,
               name="SparseGroupFtrl",
               accum_name=None,
               linear_name=None,
               l2_shrinkage_regularization_strength=0.0)
"""SparseGroupFtrlOptimizer inherits tf.compat.v1.train.FtrlOptimizer
    which implements ftrl + sparse group lasso for KvVariable.
    If var is tensorflow variable, the optimizer will
    equal to tf.compat.v1.train.FtrlOptimizer.

    Args:
      learning_rate: A float value or a constant float `Tensor`.
      learning_rate_power: Same to tf.compat.v1.train.FtrlOptimizer.
      initial_accumulator_value: The starting value for accumulators.
        Only zero or positive values are allowed.
      l1_regularization_strength: A float value, must be greater than or
        equal to zero.
      l2_regularization_strength: A float value, must be greater than or
        equal to zero.
      l21_regularization_strength: A float value, must be greater than or
        equal to zero.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "GroupAdam".
      accum_name: The suffix for the variable that keeps the gradient squared
        accumulator.  If not present, defaults to name.
      linear_name: The suffix for the variable that keeps the linear gradient
        accumulator.  If not present, defaults to name + "_1".
      l2_shrinkage_regularization_strength: Same to tf.compat.v1.train.FtrlOptimizer.
    Raises:
      ValueError: If one of the arguments is invalid.
    """
```