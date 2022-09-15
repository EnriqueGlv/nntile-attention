/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/normalize.cc
 * Normalize operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-15
 * */

#include "nntile/tensor/normalize.hh"
#include "nntile/starpu/normalize.hh"

namespace nntile
{
namespace tensor
{

//! Tile-wise average and deviation from sum and scaled sum of squares
template<typename T>
void normalize_async(const StarpuVariableHandle &gamma_beta,
        const Tensor<T> &src, const Tensor<T> &dst, Index l, T eps,
        Index axis)
{
    // Check inputs
    if(src.ndim != dst.ndim)
    {
        throw std::runtime_error("src.ndim != dst.ndim");
    }
    // Input shape dimension shall be at least 1
    if(src.ndim == 0)
    {
        throw std::runtime_error("src.ndim == 0");
    }
    // Check number of elements
    if(l <= 0)
    {
        throw std::runtime_error("l <= 0");
    }
    // Check regularization
    if(eps < 0)
    {
        throw std::runtime_error("eps < 0");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    // Check shapes
    if(src.shape[0] != 2)
    {
        throw std::runtime_error("src.shape[0] != 2");
    }
    if(src.basetile_shape[0] != 2)
    {
        throw std::runtime_error("src.basetile_shape[0] != 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != src.shape[i+1])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i+1]");
        }
        if(dst.basetile_shape[i] != src.basetile_shape[i+1])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src.basetile_shape[i+1]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
        if(dst.basetile_shape[i] != src.basetile_shape[i])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src.basetile_shape[i]");
        }
    }
    // Apply per-tile normalization asynchronously as needed
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        // Index of current source tile
        auto src_tile_index = src.grid.linear_to_index(i);
        // Source tile traits
        auto src_tile_traits = src.get_tile_traits(i);
        // Source tile handle
        auto src_tile_handle = src.get_tile_handle(i);
        // MPI rank and tag of the source tile
        int src_tile_rank = starpu_mpi_data_get_rank(src_tile_handle);
        // Set fixed indices of current destination tile
        std::vector<Index> dst_tile_index(dst.ndim);
        for(Index j = 0; j < axis; ++j)
        {
            dst_tile_index[j] = src_tile_index[j+1];
        }
        for(Index j = axis+1; j < dst.ndim; ++j)
        {
            dst_tile_index[j] = src_tile_index[j];
        }
        // Loop through all necessary destination tiles
        for(Index j = 0; j < dst.grid.shape[axis]; ++j)
        {
            // Set floating axis
            dst_tile_index[axis] = j;
            // Get linear offset from index
            Index dst_tile_offset = dst.grid.index_to_linear(dst_tile_index);
            // Get destination tile handle
            auto dst_tile_handle = dst.get_tile_handle(dst_tile_offset);
            // MPI rank of the destination tile
            int dst_tile_rank = starpu_mpi_data_get_rank(dst_tile_handle);
            // Transfer data
            if(mpi_rank == src_tile_rank or mpi_rank == dst_tile_rank)
            {
                ret = starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD,
                        src_tile_handle, dst_tile_rank, nullptr, nullptr);
                if(ret != 0)
                {
                    throw std::runtime_error("Error in starpu_mpi_get_data_on_"
                            "node_detached");
                }
            }
            // Execute on destination node
            if(mpi_rank == dst_tile_rank)
            {
                // Get destination tile traits
                auto dst_tile_traits = dst.get_tile_traits(dst_tile_offset);
                // Reshape inputs for simplicity: src -> (2,m,n), dst -> (m,k,n)
                // dst is a part of (m,l,n) tensor
                Index m, n, k;
                if(axis == 0)
                {
                    m = 1;
                    n = src_tile_traits.nelems / 2; // 2 elements per single n
                    k = dst_tile_traits.shape[0];
                }
                else if(axis == dst.ndim-1)
                {
                    m = src_tile_traits.nelems / 2; // 2 elements per single m
                    n = 1;
                    k = dst_tile_traits.shape[axis];
                }
                else
                {
                    m = dst_tile_traits.stride[axis];
                    n = dst_tile_traits.matrix_shape[axis+1][1];
                    k = dst_tile_traits.shape[axis];
                }
                // Insert corresponding task
                starpu::normalize::submit<T>(m, n, k, l, eps, gamma_beta,
                        src_tile_handle, dst_tile_handle);
            }
            // Flush cache for the output tile on every node
            starpu_mpi_cache_flush(MPI_COMM_WORLD, dst_tile_handle);
        }
    }
}

//! Tile-wise average and deviation from sum and scaled sum of squares
template<typename T>
void normalize(const StarpuVariableHandle &gamma_beta,
        const Tensor<T> &src, const Tensor<T> &dst, Index l, T eps,
        Index axis)
{
    normalize_async<T>(gamma_beta, src, dst, l, eps, axis);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void normalize(const StarpuVariableHandle &gamma_beta,
        const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst, Index l,
        fp32_t eps, Index axis);

template
void normalize(const StarpuVariableHandle &gamma_beta,
        const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst, Index l,
        fp64_t eps, Index axis);

} // namespace tensor
} // namespace nntile

