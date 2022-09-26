/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/gemm.hh
 * GEMM operation for StarPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-26
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/constants.hh>
// This also includes all definitions
#include <nntile/starpu/config.hh>

namespace nntile
{
namespace starpu
{
namespace gemm
{

//! Structure for arguments
template<typename T>
struct args_t
{
    TransOp transA;
    TransOp transB;
    Index m;
    Index n;
    Index k;
    T alpha;
    T beta;
};

#ifdef NNTILE_USE_CBLAS
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CBLAS

#ifdef NNTILE_USE_CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CUDA

extern Codelet codelet_NN_fp32, codelet_NN_fp64,
       codelet_NT_fp32, codelet_NT_fp64,
       codelet_TN_fp32, codelet_TN_fp64,
       codelet_TT_fp32, codelet_TT_fp64;

template<typename T>
static
Codelet *codelet(TransOp transA, TransOp transB)
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
Codelet *codelet<fp32_t>(TransOp transA, TransOp transB)
{
    switch(transA.value)
    {
        case TransOp::NoTrans:
            switch(transB.value)
            {
                case TransOp::NoTrans:
                    return &codelet_NN_fp32;
                default:
                // This parameter was already checked in gemm_check_opA_opB
                //case TransOp::Trans:
                    return &codelet_NT_fp32;
            }
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            switch(transB.value)
            {
                case TransOp::NoTrans:
                    return &codelet_TN_fp32;
                // This parameter was already checked in gemm_check_opA_opB
                //case TransOp::Trans:
                default:
                    return &codelet_TT_fp32;
            }
    }
}

template<>
Codelet *codelet<fp64_t>(TransOp transA, TransOp transB)
{
    switch(transA.value)
    {
        case TransOp::NoTrans:
            switch(transB.value)
            {
                case TransOp::NoTrans:
                    return &codelet_NN_fp64;
                default:
                // This parameter was already checked in gemm_check_opA_opB
                //case TransOp::Trans:
                    return &codelet_NT_fp64;
            }
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            switch(transB.value)
            {
                case TransOp::NoTrans:
                    return &codelet_TN_fp64;
                // This parameter was already checked in gemm_check_opA_opB
                //case TransOp::Trans:
                default:
                    return &codelet_TT_fp64;
            }
    }
}

void init();

void restrict_where(uint32_t where);

void restore_where();

template<typename T>
void submit(const TransOp &transA, const TransOp &transB, Index m, Index n,
        Index k, T alpha, starpu_data_handle_t A, starpu_data_handle_t B,
        T beta, starpu_data_handle_t C);

} // namespace gemm
} // namespace starpu
} // namespace nntile

