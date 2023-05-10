/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/hypot.cc
 * Hypot for 2 inputs
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
 * */

#include "nntile/kernel/hypot.hh"
#include "nntile/base_types.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::hypot;

// Templated validation
template<typename T>
void validate()
{
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // Init test input
    std::vector<T> x(1), y(1);
    x[0] = 0.3;
    y[0] = 0.4;
    // Check low-level kernel
    std::cout << "Run kernel::hypot::cpu<T>\n";
    cpu<T>(-2.0, &x[0], 0.0, &y[0]);
    TEST_ASSERT(std::abs(y[0]-0.6) <= 10*eps);
    std::cout << "OK: kernel::hypot::cpu<T>\n";
    std::cout << "Run kernel::hypot::cpu<T>\n";
    cpu<T>(0.0, &x[0], -2.0, &y[0]);
    TEST_ASSERT(std::abs(y[0]-1.2) <= 10*eps);
    std::cout << "OK: kernel::hypot::cpu<T>\n";
    x[0] = 0.9;
    std::cout << "Run kernel::hypot::cpu<T>\n";
    cpu<T>(-1.0, &x[0], 1.0, &y[0]);
    TEST_ASSERT(std::abs(y[0]-1.5) <= 10*eps);
    std::cout << "OK: kernel::hypot::cpu<T>\n";
}

int main(int argc, char **argv)
{
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

