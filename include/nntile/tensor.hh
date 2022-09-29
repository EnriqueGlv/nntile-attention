/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor.hh
 * Header for Tensor<T> class with corresponding operations
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-29
 * */

#pragma once

// Get Tensor<T> class
#include <nntile/tensor/tensor.hh>
// MPI distributions
#include <nntile/tensor/distributions.hh>

// Tensor operations
#include <nntile/tensor/bias.hh>
#include <nntile/tensor/clear.hh>
#include <nntile/tensor/copy.hh>
#include <nntile/tensor/copy_intersection.hh>
#include <nntile/tensor/gather.hh>
#include <nntile/tensor/gelu.hh>
#include <nntile/tensor/gelutanh.hh>
#include <nntile/tensor/gemm.hh>
#include <nntile/tensor/normalize.hh>
#include <nntile/tensor/randn.hh>
#include <nntile/tensor/relu.hh>
#include <nntile/tensor/scatter.hh>
#include <nntile/tensor/sumnorm.hh>

namespace nntile
{
//! @namespace nntile::tensor
/*! This namespace holds high-level routines for Tensor<T>
 * */
namespace tensor
{

} // namespace tensor
} // namespace nntile

