# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/tensor.py
# Multiprecision tensor with operations
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @date 2023-03-15

from .nntile_core import tensor as core_tensor
from .nntile_core.tensor import TensorTraits, Tensor_fp32, Tensor_fp64, Tensor_int64
from .nntile_core import TransOp, notrans, trans
from typing import Union, List

# Multiprecision tensor as a union type for all precisions
Tensor = Union[core_tensor.Tensor_fp32, core_tensor.Tensor_fp64]
# Optional tensor argument
TensorOrNone = Union[Tensor, None]
# Union of multiprecision tensor and float
TensorOrFloat = Union[Tensor, float]
TensorFloatOrInt = Union[Tensor, core_tensor.Tensor_int64]

# Struct meant for tensor, its gradient and a flag if gradient is required
class TensorMoments(object):
    value: TensorOrNone
    grad: TensorOrNone
    grad_required: bool

    def __init__(self, value: TensorOrNone, grad: TensorOrNone,
            grad_required: bool):
        self.value = value
        self.grad = grad
        self.grad_required = grad_required

    def __del__(self):
        self.unregister()

    def unregister(self):
        if self.value is not None:
            self.value.unregister()
        if self.grad is not None:
            self.grad.unregister()


# Wrapper for multiprecision gemm
def gemm_async(alpha: float, trans_A: TransOp, A: Tensor, trans_B: TransOp,
        B: Tensor, beta: float, C: Tensor, ndim: int) -> None:
    if type(A) is not type(B) or type(A) is not type(C):
        raise TypeError
    if type(A) is core_tensor.Tensor_fp32:
        core_tensor.gemm_async_fp32(alpha, trans_A, A, trans_B, B, beta, C,
                ndim)
    else:
        core_tensor.gemm_async_fp64(alpha, trans_A, A, trans_B, B, beta, C,
                ndim)

# Wrapper for multiprecision ReLU
def relu_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.relu_async_fp32(x)
    else:
        core_tensor.relu_async_fp64(x)

# Wrapper for multiprecision derivative of ReLU
def drelu_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.drelu_async_fp32(x)
    else:
        core_tensor.drelu_async_fp64(x)

# Wrapper for multiprecision sumnorm
def sumnorm_async(x: Tensor, sumnorm: Tensor, axis: int) -> None:
    if type(x) is not type(sumnorm):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.sumnorm_async_fp32(x, sumnorm, axis)
    else:
        core_tensor.sumnorm_async_fp64(x, sumnorm, axis)

# Wrapper for multiprecision softmax
def softmax_async(maxsumexp: Tensor, x: Tensor, axis: int) -> None:
    if type(maxsumexp) is not type(x):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.softmax_async_fp32(maxsumexp, x, axis)
    else:
        core_tensor.softmax_async_fp64(maxsumexp, x, axis)

# Wrapper for multiprecision scatter
def scatter_async(x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.scatter_async_fp32(x, y)
    else:
        core_tensor.scatter_async_fp64(x, y)

# Wrapper for multiprecision randn
def randn_async(x: Tensor, start: List[int], shape: List[int], seed: int,
        mean: float, dev: float) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.randn_async_fp32(x, start, shape, seed, mean, dev)
    else:
        core_tensor.randn_async_fp64(x, start, shape, seed, mean, dev)

# Wrapper for multiprecision prod
def prod_async(x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.prod_async_fp32(x, y)
    else:
        core_tensor.prod_async_fp64(x, y)

# Wrapper for multiprecision nrm2
def nrm2_async(x: Tensor, y: Tensor, tmp: Tensor) -> None:
    if type(x) is not type(y) or type(x) is not type(tmp):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.nrm2_async_fp32(x, y, tmp)
    else:
        core_tensor.nrm2_async_fp64(x, y, tmp)

# Wrapper for multiprecision normalize
def normalize_async(gb: Tensor, x: Tensor, y: Tensor, l: int, eps: float,
        axis: int) -> None:
    if type(x) is not type(y) or type(x) is not type(gb):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.normalize_async_fp32(gb, x, y, l, eps, axis)
    else:
        core_tensor.normalize_async_fp64(gb, x, y, l, eps, axis)

# Wrapper for multiprecision maxsumexp
def maxsumexp_async(x: Tensor, maxsumexp: Tensor, axis: int) -> None:
    if type(x) is not type(maxsumexp):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.maxsumexp_async_fp32(x, maxsumexp, axis)
    else:
        core_tensor.maxsumexp_async_fp64(x, maxsumexp, axis)

# Wrapper for multiprecision bias
def bias_async(bias: Tensor, x: Tensor, axis: int) -> None:
    if type(bias) is not type(x):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.bias_async_fp32(bias, x, axis)
    else:
        core_tensor.bias_async_fp64(bias, x, axis)

# Wrapper for multiprecision gather
def gather_async(x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.gather_async_fp32(x, y)
    else:
        core_tensor.gather_async_fp64(x, y)

# Wrapper for multiprecision copy_intersection
def copy_intersection_async(x: TensorFloatOrInt, x_offset: List[int], y: TensorFloatOrInt,
        y_offset: List[int]) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.copy_intersection_async_fp32(x, x_offset, y, y_offset)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.copy_intersection_async_fp64(x, x_offset, y, y_offset)
    elif type(x) is core_tensor.Tensor_int64:
        core_tensor.copy_intersection_async_int64(x, x_offset, y, y_offset)

# Wrapper for multiprecision copy
def copy_async(x: TensorFloatOrInt, y: TensorFloatOrInt) -> None:
    if type(x) is not type(y):
        print(type(x), type(y))
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.copy_async_fp32(x, y)
    elif type(x) is core_tensor.Tensor_fp64:
        core_tensor.copy_async_fp64(x, y)
    elif type(x) is core_tensor.Tensor_int64:
        core_tensor.copy_async_int64(x, y)

# Wrapper for multiprecision clear
def clear_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.clear_async_fp32(x)
    else:
        core_tensor.clear_async_fp64(x)

# Wrapper for multiprecision axpy
def axpy_async(alpha: TensorOrFloat, x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(alpha) is Tensor:
        if type(alpha) is not type(x):
            raise TypeError
        if type(x) is core_tensor.Tensor_fp32:
            core_tensor.axpy_async_fp32(alpha, x, y)
        else:
            core_tensor.axpy_async_fp64(alpha, x, y)
    else:
        if type(x) is core_tensor.Tensor_fp32:
            core_tensor.axpy_async_fp32(alpha, x, y)
        else:
            core_tensor.axpy_async_fp64(alpha, x, y)

# Wrapper for multiprecision square root
def sqrt_async(x: Tensor) -> None:
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.sqrt_async_fp32(x)
    else:
        core_tensor.sqrt_async_fp64(x)

# Wrapper for multiprecision elementwise maximum
def maximum_async(x: Tensor, y: Tensor) -> None:
    if type(x) is not type(y):
        raise TypeError
    if type(x) is core_tensor.Tensor_fp32:
        core_tensor.maximum_async_fp32(x, y)
    else:
        core_tensor.maximum_async_fp64(x, y)

# Wrapper for multiprecision axpy
def addcdiv_async(alpha: float, eps: float, nom: Tensor, denom: Tensor, src: Tensor) -> None:
    if type(nom) is not type(denom):
        raise TypeError
    if type(nom) is not type(src):
        raise TypeError
    
    if type(nom) is core_tensor.Tensor_fp32:
        core_tensor.addcdiv_async_fp32(alpha, eps, nom, denom, src)
    else:
        core_tensor.addcdiv_async_fp64(alpha, eps, nom, denom, src)

def logsumexp_async(maxsumexp: Tensor, logsumexp: Tensor) -> None:
    if type(maxsumexp) is not type(logsumexp):
        raise TypeError
    if type(maxsumexp) is core_tensor.Tensor_fp32:
        core_tensor.logsumexp_async_fp32(maxsumexp, logsumexp)
    else:
        core_tensor.logsumexp_async_fp64(maxsumexp, logsumexp)

def total_sum_accum_async(logsumexp: Tensor, src: Tensor, class_labels: Tensor_int64,
                          val: Tensor):
    if type(logsumexp) is core_tensor.Tensor_fp32:
        core_tensor.total_sum_accum_async_fp32(logsumexp, src, class_labels, val)
    else:
        core_tensor.total_sum_accum_async_fp64(logsumexp, src, class_labels, val)

def subtract_indexed_column_async(val: float, class_labels: Tensor_int64,
                          dst: Tensor):
    if type(dst) is core_tensor.Tensor_fp32:
        core_tensor.subtract_indexed_column_async_fp32(val, class_labels, dst)
    else:
        core_tensor.subtract_indexed_column_async_fp64(val, class_labels, dst)