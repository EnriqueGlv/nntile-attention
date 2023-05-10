/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/prod_slice.hh
 * Bias-like product operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-02
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{
namespace tile
{

// Tile<T> per-element multiplication of a tensor and a broadcasted slice
template<typename T>
void prod_slice_async(const Tile<T> &src, T alpha, const Tile<T> &dst,
        Index axis);

// Tile<T> per-element multiplication of a tensor and a broadcasted slice
template<typename T>
void prod_slice(const Tile<T> &src, T alpha, const Tile<T> &dst, Index axis);

} // namespace tile
} // namespace nntile

