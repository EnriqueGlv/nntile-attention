/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/pow.hh
 * Power operation low-level kernels
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/kernel/pow/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/pow/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::pow
/*! Low-level implementations of power operation
 * */
namespace nntile::kernel::pow
{

} // namespace nntile::kernel::pow
