#pragma once

#include <nntile/tensor/tensor.hh>
#include <nntile/constants.hh>

namespace nntile::tensor
{

void gemm_check(const TransOp &transA, const TensorTraits &A,
        const TransOp &transB, const TensorTraits &B, const TensorTraits &C,
        Index ndim, Index batch_ndim);

template<typename T>
void linear_relu_async(Scalar alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, Scalar beta,
        const Tensor<T> &C, Index ndim, Index batch_ndim, int redux=0);

} // namespace nntile::tensor