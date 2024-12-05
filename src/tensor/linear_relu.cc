#include "nntile/tensor/linear_relu.hh"
#include "nntile/starpu/linear_relu.hh"
#include "nntile/tensor/gemm.hh" // reuse gemm_check from nntile base gemm kernel
#include "nntile/starpu/gemm.hh"

namespace nntile::tensor{

//! Asynchronous version of tensor-wise gemm operation
/*! Matrix multiplication for tensors, which are virtually reshaped
 *
 * @param[in] alpha: Alpha multiplier
 * @param[in] transA: Transposition flag for the tensor A
 * @param[in] A: Input tensor A
 * @param[in] transB: Transposition flag for the tensor B
 * @param[in] B: Input tensor B
 * @param[in] beta: Beta multiplier
 * @param[inout] C: Output tensor C
 * @param[in] ndim: Number of dimensions used in gemm contraction
 * @param[in] batch_ndim: Number of last dimensions used for batching of gemms
 * @param[in] redux: Whether or not to use STARPU_REDUX
 * */
template<typename T>
void linear_relu_async(Scalar alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, Scalar beta,
        const Tensor<T> &C, Index ndim, Index batch_ndim, int redux, int act)
{
    // Check inputs (throw exception in case of an error)
    gemm_check(transA, A, transB, B, C, ndim, batch_ndim);
    // Sizes of A, B and C as simple matrices (grids of tiles) for gemm
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    constexpr Scalar one = 1;
    Index m = C.grid.matrix_shape[A.ndim-batch_ndim-ndim][0];
    Index batch = C.grid.matrix_shape[C.ndim-batch_ndim][1];
    Index n = C.grid.matrix_shape[A.ndim-batch_ndim-ndim][1] / batch;
    Index k;
    std::array<Index, 2> opA_stride, opB_stride;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            k = A.grid.matrix_shape[A.ndim-batch_ndim-ndim][1] / batch;
            opA_stride = {1, m};
            break;
        case TransOp::Trans:
            k = A.grid.matrix_shape[ndim][0];
            opA_stride = {k, 1};
            break;
    }
    switch(transB.value)
    {
        case TransOp::NoTrans:
            opB_stride = {1, k};
            break;
        case TransOp::Trans:
            opB_stride = {n, 1};
            break;
    }
    // All per-tile starpu gemm calls shall appear here
    for(Index b = 0; b < batch; ++b)
    {
        for(Index j = 0; j < n; ++j)
        {
            for(Index i = 0; i < m; ++i)
            {
                Index C_tile_offset = (b*n+j)*m + i;
                auto C_tile_handle = C.get_tile_handle(C_tile_offset);
                auto C_tile_traits = C.get_tile_traits(C_tile_offset);
                // int C_tile_rank = C_tile_handle.mpi_get_rank();
                Index tile_m = C_tile_traits.matrix_shape[
                    A.ndim-batch_ndim-ndim][0];
                Index tile_batch = C_tile_traits.matrix_shape[
                    C.ndim-batch_ndim][1];
                Index tile_n = C_tile_traits.matrix_shape[
                    A.ndim-batch_ndim-ndim][1] / tile_batch;
                // initialize C(i,j,b) = a*opA(i,0,b)*opB(0,j,b) + b*C(i,j,b)
                Index A_tile_offset = opA_stride[0]*i + b*m*k;
                Index B_tile_offset = opB_stride[1]*j + b*n*k;
                auto A_first_tile_handle = A.get_tile_handle(A_tile_offset);
                auto B_first_tile_handle = B.get_tile_handle(B_tile_offset);
                // int A_first_tile_rank = A_first_tile_handle.mpi_get_rank();
                // int B_first_tile_rank = B_first_tile_handle.mpi_get_rank();
                // Transfer first tile A on node with tile C
                // A_first_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
                // Transfer first tile B on node with tile C
                // B_first_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
                // Execute on node with tile C
                // if(mpi_rank == C_tile_rank)
                // {
                Index tile_k;
                auto A_first_tile_traits = A.get_tile_traits(A_tile_offset);
                switch(transA.value){
                    case TransOp::NoTrans:
                        tile_k = A_first_tile_traits.matrix_shape[
                            A.ndim-batch_ndim-ndim][1] / tile_batch;
                        break;
                        // This parameter was already checked
                        //case TransOp::Trans:
                    default:
                        tile_k = A_first_tile_traits.matrix_shape[ndim][0];
                        break;
                }
                if(k==1){
                    starpu::linRelu::submit<T>(transA, transB, tile_m,
                            tile_n,
                            tile_k, tile_batch, alpha, A_first_tile_handle,
                            B_first_tile_handle, beta, C_tile_handle, redux, act);
                } else {
                    starpu::gemm::submit<T>(transA, transB, tile_m,
                            tile_n,
                            tile_k, tile_batch, alpha, A_first_tile_handle,
                            B_first_tile_handle, beta, C_tile_handle, redux);
                }
                // }
                // all other l>0
                for(Index l = 1; l < k; ++l)
                {
                    // accumulate C(i,j,b) = a*opA(i,l,b)*opB(l,j,b) + C(i,j,b)
                    A_tile_offset += opA_stride[1];
                    B_tile_offset += opB_stride[0];
                    auto A_tile_handle = A.get_tile_handle(A_tile_offset);
                    auto B_tile_handle = B.get_tile_handle(B_tile_offset);
                    // int A_tile_rank = A_tile_handle.mpi_get_rank();
                    // int B_tile_rank = B_tile_handle.mpi_get_rank();
                    // Transfer tile A on node with tile C
                    // A_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
                    // Transfer tile B on node with tile C
                    // B_tile_handle.mpi_transfer(C_tile_rank, mpi_rank);
                    // Execute on node with tile C
                    // if(mpi_rank == C_tile_rank)
                    // {
                        Index tile_k;
                        auto A_tile_traits = A.get_tile_traits(A_tile_offset);
                        switch(transA.value)
                        {
                            case TransOp::NoTrans:
                                tile_k = A_tile_traits.matrix_shape[
                                    A.ndim-batch_ndim-ndim][1] / tile_batch;
                                break;
                                // This parameter was already checked
                                //case TransOp::Trans:
                            default:
                                tile_k = A_tile_traits.matrix_shape[ndim][0];
                                break;
                        }
                        // apply ReLU only on last GEMM
                        if(l == k-1){
                            starpu::linRelu::submit<T>(transA, transB, tile_m,
                                    tile_n,
                                    tile_k, tile_batch, alpha, A_tile_handle,
                                    B_tile_handle, one, C_tile_handle, redux, act);
                        } else {
                            starpu::gemm::submit<T>(transA, transB, tile_m,
                                    tile_n,
                                    tile_k, tile_batch, alpha, A_tile_handle,
                                    B_tile_handle, one, C_tile_handle, redux);
                        }

                    // }
                }

                // Flush cache for the output tile on every node
                C_tile_handle.mpi_flush();
            }
        }
    }
}

// Explicit instantiation
template
void linear_relu_async<fp32_t>(Scalar alpha, const TransOp &transA,
        const Tensor<fp32_t> &A,
        const TransOp &transB, const Tensor<fp32_t> &B, Scalar beta,
        const Tensor<fp32_t> &C, Index ndim, Index batch_ndim, int redux, int act);

}