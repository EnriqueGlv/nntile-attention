/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/prod_slice.hh
 * Bias-like product operation for Tile<T>
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Tile<T> per-element multiplication of a tensor and a broadcasted slice
template<typename T>
void prod_slice_async(const Tile<T> &src, T alpha, const Tile<T> &dst,
        Index axis);

// Tile<T> per-element multiplication of a tensor and a broadcasted slice
template<typename T>
void prod_slice(const Tile<T> &src, T alpha, const Tile<T> &dst, Index axis);

} // namespace nntile::tile

