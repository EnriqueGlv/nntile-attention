/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile.hh
 * Header for Tile<T> class with corresponding operations
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-29
 * */

#pragma once

// Get Tile<T> class
#include <nntile/tile/tile.hh>

// Tile<T> operations
#include <nntile/tile/bias.hh>
#include <nntile/tile/clear.hh>
#include <nntile/tile/copy.hh>
#include <nntile/tile/copy_intersection.hh>
#include <nntile/tile/gelu.hh>
#include <nntile/tile/gelutanh.hh>
#include <nntile/tile/gemm.hh>
#include <nntile/tile/normalize.hh>
#include <nntile/tile/randn.hh>
#include <nntile/tile/relu.hh>
#include <nntile/tile/sumnorm.hh>

namespace nntile
{
//! @namespace nntile::tile
/*! This namespace holds high-level routines for Tile<T>
 * */
namespace tile
{

} // namespace tile
} // namespace nntile

