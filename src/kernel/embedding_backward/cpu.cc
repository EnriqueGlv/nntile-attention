/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/embedding_backward/cpu.cc
 * Backward of embeddings from vocabulary within buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-21
 * */

#include "nntile/kernel/embedding_backward/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace embedding_backward
{

template<typename T>
void cpu(Index m, Index n, Index k, Index k_start, Index k_size,
        const Index *index, const T *embed, T *vocab)
    noexcept
//! Accumulate gradients of embeddings into vocabulary
/*! Does the following operation:
 *      vocab[:, index[i, j]] += embed[i, k_start:k_start+k_size, j]
 *
 * @param[in] m: Size of the first mode of index and embed tensors
 * @param[in] n: Size of the last mode of index and embed tensors
 * @param[in] k: Size of the middle mode of embed tensor
 * @param[in] k_start: Offset of the middle mode of embed tensor
 * @param[in] k_size: Size of the first mode of vocab tensor
 * @param[in] index: Tokens (indices of embeddings)
 * @param[out] embed: Tensor of gradients of embeddings
 * @param[inout] vocab: Gradient of vocabulary. It is a contiguous matrix of
 *      shape (k_size, vocab_size) but vocab_size is not passed as a parameter.
 * */
{
    // Cycle over column of embed buffer
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over row of embed buffer
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Output slice of vocabulary
            T *vocab_slice = vocab + k_size*index[i2*m+i1];
            // Input slice of embedding
            const T *embed_slice = embed + (i2*k+k_start)*m + i1;
            // Cycle over slice of output vocab
            for(Index i0 = 0; i0 < k_size; ++i0)
            {
                vocab_slice[i0] += embed_slice[i0*m];
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        const Index *index, const fp32_t *embed, fp32_t *vocab)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        const Index *index, const fp64_t *embed, fp64_t *vocab)
    noexcept;

} // namespace embedding_backward
} // namespace kernel
} // namespace nntile

