#pragma once

#include <nntile/base_types.hh>
#include <nntile/constants.hh>
// This also includes all definitions
#include <nntile/starpu/config.hh>

#ifdef NNTILE_USE_CUDA
#    include <cublas_v2.h>
#    include <starpu_cublas_v2.h>
#    include <cublasLt.h>
//#    include <cuda_fp16.h>
#endif // NNTILE_USE_CUDA

namespace nntile::starpu::linRelu
{

//! Structure for arguments
struct args_t
{
    TransOp transA; // op(A)
    TransOp transB; // op(B)
    Index m; // Number of rows of op(A) and C
    Index n; // Number of columns of op(B) and C
    Index k; // Number of columns of op(A) and number of rows of op(B)
    Index batch; // Number of gemms in a batch
    Scalar alpha;
    Scalar beta;
    cublasLtEpilogue_t act;
    bool bias;
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

extern Codelet codelet_NN_fp32,
               codelet_NT_fp32,
               codelet_TN_fp32,
               codelet_TT_fp32;

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

void init();

void restrict_where(uint32_t where);

void restore_where();

template<typename T>
void submit(const TransOp &transA, const TransOp &transB, Index m, Index n,
        Index k, Index batch, Scalar alpha, Handle A, Handle B, Scalar beta,
        Handle C, int redux=0, int act=1, bool bias=false, Handle BH=nullptr);

} // namespace nntile::starpu::gemm
