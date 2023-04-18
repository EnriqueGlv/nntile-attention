/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/pow.cc
 * Power operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
 * */

#include "nntile/starpu/pow.hh"
#include "nntile/kernel/pow.hh"
#include <cstdlib>

namespace nntile
{
namespace starpu
{
namespace pow
{

//! Inplace power operation of StarPU buffer on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t<T> *args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    // Launch kernel
    kernel::pow::cpu<T>(args->nelems, args->alpha, args->exp, data);
}

#ifdef NNTILE_USE_CUDA
// Inplace power operation of StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t<T> *args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::pow::cuda<T>(stream, args->nelems, args->alpha, args->exp, data);
}
#endif // NNTILE_USE_CUDA

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_pow_fp32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_pow_fp64",
            nullptr,
            {cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp64_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
}

void restrict_where(uint32_t where)
{
    codelet_fp32.restrict_where(where);
    codelet_fp64.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_fp64.restore_where();
}

template<typename T>
void submit(Index nelems, T alpha, T exp, Handle data)
//! Insert pow task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t<T> *args = (args_t<T> *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->exp = exp;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_RW, static_cast<starpu_data_handle_t>(data),
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in pow task submission");
    }
}

// Explicit instantiaion
template
void submit<fp32_t>(Index nelems, fp32_t alpha, fp32_t exp, Handle data);

template
void submit<fp64_t>(Index nelems, fp64_t alpha, fp64_t exp, Handle data);

} // namespace pow
} // namespace starpu
} // namespace nntile

