/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/cpu/gelu.cc
 * GeLU operation on a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-04
 * */

#include "nntile/kernel/cpu/gelu.hh"
#include <vector>
#include <stdexcept>
#include <cmath>

using namespace nntile;
using namespace nntile::kernel::cpu;

// Templated validation
template<typename T>
void validate(Index nelems)
{
    // Init test input
    std::vector<T> dst(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = T(2*i+1-nelems) / T{10};
    }
    std::vector<T> dst2(dst);
    // Check low-level kernel
    gelu<T>(nelems, &dst[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        T x = dst2[i];
        T val_ref = 0.5*std::erf(x/std::sqrt(T(2))) + 0.5;
        val_ref *= x;
        if(dst[i] != val_ref)
        {
            throw std::runtime_error("Wrong value");
        }
    }
}

int main(int argc, char **argv)
{
    validate<fp32_t>(0);
    validate<fp32_t>(1);
    validate<fp32_t>(80000);
    validate<fp64_t>(0);
    validate<fp64_t>(1);
    validate<fp64_t>(80000);
    return 0;
}

