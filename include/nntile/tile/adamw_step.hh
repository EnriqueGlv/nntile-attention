/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/adamw_step.hh
 * AdamW step for Tile<T>
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

// Asynchronous tile-wise AdamW step operation
template<typename T>
void adamw_step_async(Index num_iter, scal_t beta_1, scal_t beta_2, scal_t eps, scal_t lr, scal_t weight_decay,
                     const Tile<T> &grad, const Tile<T> &first_moment, const Tile<T> &second_moment,
                     const Tile<T> &p);

// Blocking version of tile-wise AdamW step operation
template<typename T>
void adamw_step(Index num_iter, scal_t beta_1, scal_t beta_2, scal_t eps, scal_t lr, scal_t weight_decay,
               const Tile<T> &grad, const Tile<T> &first_moment, const Tile<T> &second_moment,
               const Tile<T> &p);

} // namespace nntile::tile
