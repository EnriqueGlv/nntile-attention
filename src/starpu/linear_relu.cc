/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gemm.cc
 * GEMM operation for StarPU buffers
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/linear_relu.hh"



// keep gemm kernel for cblas
#ifndef STARPU_SIMGRID
#   include "nntile/kernel/gemm.hh"
#endif

namespace nntile::starpu::linRelu
{

// using namespace nntile::kernel::gemm;

#ifdef NNTILE_USE_CBLAS
//! GEMM for contiguous matrices without padding through StarPU buffers
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    // Launch kernel
    const T *A = interfaces[0]->get_ptr<T>();
    const T *B = interfaces[1]->get_ptr<T>();
    T *C = interfaces[2]->get_ptr<T>();
    // It is OK to convert values as it was checked during task submission
    CBLAS_INT M=args->m, N=args->n, K=args->k, ldA, ldB, ldC=M;
    CBLAS_TRANSPOSE transA_, transB_;
    // Convert other values to CBLAS types
    switch(args->transA.value)
    {
        case TransOp::NoTrans:
            transA_ = CblasNoTrans;
            ldA = M;
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            transA_ = CblasTrans;
            ldA = K;
    }
    switch(args->transB.value)
    {
        case TransOp::NoTrans:
            transB_ = CblasNoTrans;
            ldB = K;
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            transB_ = CblasTrans;
            ldB = N;
    }
    // Call corresponding CBLAS routine
    Index A_offset = args->m * args->k, B_offset = args->n * args->k,
            C_offset = args->m * args->n;
    for(Index i = 0; i < args->batch; ++i)
    {
        nntile::kernel::gemm::cblas(transA_, transB_, M, N, K, args->alpha, A, ldA, B, ldB,
                args->beta, C, ldC);
        A += A_offset;
        B += B_offset;
        C += C_offset;
    }
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CBLAS

#ifdef NNTILE_USE_CUDA
//! GEMM for contiguous matrices without padding through StarPU buffers
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    // Launch kernel
    const T *A = interfaces[0]->get_ptr<T>();
    const T *B = interfaces[1]->get_ptr<T>();
    T *C = interfaces[2]->get_ptr<T>();
    // It is OK to convert values as it was checked during task submission
    int M=args->m, N=args->n, K=args->k, ldA, ldB, ldC=M;
    cublasOperation_t transA_, transB_;
    // Convert other values to CBLAS types
    switch(args->transA.value)
    {
        case TransOp::NoTrans:
            transA_ = CUBLAS_OP_N;
            ldA = M;
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            transA_ = CUBLAS_OP_T;
            ldA = K;
    }
    switch(args->transB.value)
    {
        case TransOp::NoTrans:
            transB_ = CUBLAS_OP_N;
            ldB = K;
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            transB_ = CUBLAS_OP_T;
            ldB = N;
    }
    // Get cuBLAS handle and CUDA stream
    // cublasHandle_t handle = starpu_cublas_get_local_handle();
    cublasLtHandle_t ltHandle = (cublasLtHandle_t)starpu_cublas_get_local_handle();
    cudaStream_t stream = starpu_cuda_get_local_stream();
    cublasSetStream((cublasHandle_t)ltHandle, stream);
    // alpha and beta parameters of GEMM operation are on CPU host
    cublasSetPointerMode((cublasHandle_t)ltHandle, CUBLAS_POINTER_MODE_HOST);

    // Call corresponding cuBLAS routine
    // Create GEMM descriptor
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // Set the epilogue to include ReLU
    cublasLtEpilogue_t epilogue = args->act;//CUBLASLT_EPILOGUE_RELU;
    cublasLtMatmulDescSetAttribute(matmulDesc,
                                    CUBLASLT_MATMUL_DESC_EPILOGUE,
                                    &epilogue,
                                    sizeof(epilogue));

    // set transppose ops
    cublasLtMatmulDescSetAttribute(matmulDesc, 
                                    CUBLASLT_MATMUL_DESC_TRANSA, 
                                    &transA_, 
                                    sizeof(transA_));
    cublasLtMatmulDescSetAttribute(matmulDesc, 
                                    CUBLASLT_MATMUL_DESC_TRANSB, 
                                    &transB_, 
                                    sizeof(transB_));

    // Create matrix layouts
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, (transA_ == CUBLAS_OP_N) ? M : K, (transA_ == CUBLAS_OP_N) ? K : M, ldA);
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, (transB_ == CUBLAS_OP_N) ? K : N, (transB_ == CUBLAS_OP_N) ? N : K, ldB);
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, M, N, ldC);

    if(args->batch == 1)
    {
        // cublas(handle, transA_, transB_, M, N, K, args->alpha, A, ldA, B, ldB,
        //         args->beta, C, M);
        // cublasSgemm(handle, transA_, transB_, M, N, K, &args->alpha, (const float *)A,
        //     ldA, (const float *)B, ldB, &args->beta, (float *)C, ldC);
        cublasLtMatmul(ltHandle,
                matmulDesc,
                &args->alpha,    // Alpha
                A, layoutA,      // Matrix A
                B, layoutB,      // Matrix B
                &args->beta,     // Beta
                C, layoutC,      // Matrix C
                C, layoutC,      // Output matrix
                nullptr,         // Algorithm (nullptr for default)
                nullptr,         // Workspace (nullptr for no extra memory)
                0,               // Workspace size
                stream);         // CUDA stream
    }
    else
    {
        Index A_offset = args->m * args->k, B_offset = args->n * args->k,
                C_offset = args->m * args->n;
        // cublas_batch(handle, transA_, transB_, M, N, K, args->alpha, A, ldA,
        //         A_offset, B, ldB, B_offset, args->beta, C, M, C_offset,
        //         args->batch);
        // cublasSgemmStridedBatched(handle, transA_, transB_, M, N, K, &args->alpha,
        //     (const float *)A, ldA, A_offset, (const float *)B, ldB, B_offset,
        //     &args->beta, (float *)C, ldC, C_offset, args->batch);
        for (int b = 0; b < args->batch; ++b) {
            const T *A_b = A + b * A_offset;
            const T *B_b = B + b * B_offset;
            T *C_b = C + b * C_offset;

            cublasLtMatmul(ltHandle,
                        matmulDesc,
                        &args->alpha,
                        A_b, layoutA,
                        B_b, layoutB,
                        &args->beta,
                        C_b, layoutC,
                        C_b, layoutC,
                        nullptr,
                        nullptr,
                        0,
                        stream);
        }
    }
#endif // STARPU_SIMGRID
}
#endif //NNTILE_USE_CUDA

//! Footprint for GEMM tasks that depends only on M, N, K and alpha
static
uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // In case alpha is zero, entire gemm is unnecessary so it is better to
    // give it a different footprint since gemm time will be totally different
    uint32_t hash = args->alpha == Scalar{0} ? -1 : 0;
    // Apply hash over parameters M, N and K. This way if we swap values of M,
    // N and K total size of buffers will remain the same, but the footprint
    // will be different
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    hash = starpu_hash_crc32c_be_n(&args->batch, sizeof(args->batch), hash);
    return hash;
}

Codelet codelet_NN_fp32,
        codelet_NT_fp32,
        codelet_TN_fp32,
        codelet_TT_fp32;

void init()
{
    codelet_NN_fp32.init("nntile_lin_relu_NN_fp32",
            footprint,
#ifdef NNTILE_USE_CBLAS
            {cpu<fp32_t>},
#else // NNTILE_USE_CBLAS
            {},
#endif // NNTILE_USE_CBLAS
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_NT_fp32.init("nntile_lin_relu_NT_fp32",
            footprint,
#ifdef NNTILE_USE_CBLAS
            {cpu<fp32_t>},
#else // NNTILE_USE_CBLAS
            {},
#endif // NNTILE_USE_CBLAS
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_TN_fp32.init("nntile_lin_relu_TN_fp32",
            footprint,
#ifdef NNTILE_USE_CBLAS
            {cpu<fp32_t>},
#else // NNTILE_USE_CBLAS
            {},
#endif // NNTILE_USE_CBLAS
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_TT_fp32.init("nntile_lin_relu_TT_fp32",
            footprint,
#ifdef NNTILE_USE_CBLAS
            {cpu<fp32_t>},
#else // NNTILE_USE_CBLAS
            {},
#endif // NNTILE_USE_CBLAS
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
}

void restrict_where(uint32_t where){
    codelet_NN_fp32.restrict_where(where);
    codelet_NT_fp32.restrict_where(where);
    codelet_TN_fp32.restrict_where(where);
    codelet_TT_fp32.restrict_where(where);
}

void restore_where(){
    codelet_NN_fp32.restore_where();
    codelet_NT_fp32.restore_where();
    codelet_TN_fp32.restore_where();
    codelet_TT_fp32.restore_where();
}

template<typename T>
void submit(const TransOp &transA, const TransOp &transB, Index m, Index n,
        Index k, Index batch, Scalar alpha, Handle A, Handle B, Scalar beta,
        Handle C, int redux, int act)
{
    // Check that matrix sizes fit proper types for underlying CBLAS
#ifdef NNTILE_USE_CBLAS
#ifndef STARPU_SIMGRID
    if(static_cast<CBLAS_INT>(m) != m)
    {
        throw std::runtime_error("GEMM size M does not fit CBLAS_INT");
    }
    if(static_cast<CBLAS_INT>(n) != n)
    {
        throw std::runtime_error("GEMM size N does not fit CBLAS_INT");
    }
    if(static_cast<CBLAS_INT>(k) != k)
    {
        throw std::runtime_error("GEMM size K does not fit CBLAS_INT");
    }
#endif // STARPU_SIMGRID
#endif // NNTILE_USE_CBLAS
    // Check that matrix sizes fit proper types for underlying CUBLAS
#ifdef NNTILE_USE_CUDA
#ifndef STARPU_SIMGRID
    if(static_cast<int>(m) != m)
    {
        throw std::runtime_error("GEMM size M does not fit int");
    }
    if(static_cast<int>(n) != n)
    {
        throw std::runtime_error("GEMM size N does not fit int");
    }
    if(static_cast<int>(k) != k)
    {
        throw std::runtime_error("GEMM size K does not fit int");
    }
#endif // STARPU_SIMGRID
#endif // NNTILE_USE_CUDA
    constexpr Scalar zero = 0, one = 1;
    enum starpu_data_access_mode C_mode;
    if(beta == zero)
    {
        C_mode = STARPU_W;
    }
    else if(beta == one)
    {
        if(redux != 0)
        {
            C_mode = STARPU_REDUX;
            //C_mode = Config::STARPU_RW_COMMUTE;
        }
        else
        {
            C_mode = Config::STARPU_RW_COMMUTE;
        }
    }
    else
    {
        C_mode = STARPU_RW;
    }
    // Codelet arguments
    auto args = new args_t
    {
        .transA = transA,
        .transB = transB,
        .m = m,
        .n = n,
        .k = k,
        .batch = batch,
        .alpha = alpha,
        .beta = beta,
        .act = static_cast<cublasLtEpilogue_t>(act)
    };
    double nflops = 2 * m * n * k * batch;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(transA, transB),
            STARPU_R, static_cast<starpu_data_handle_t>(A),
            STARPU_R, static_cast<starpu_data_handle_t>(B),
            C_mode, static_cast<starpu_data_handle_t>(C),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in lin_relu task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(const TransOp &transA, const TransOp &transB,
        Index m, Index n, Index k, Index batch, Scalar alpha, Handle A,
        Handle B, Scalar beta, Handle C, int redux, int act);

} // namespace nntile::starpu::gemm
