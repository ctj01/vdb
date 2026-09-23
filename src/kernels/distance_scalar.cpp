/**
 * @file distance_scalar.cpp
 * @brief Scalar reference kernels — correct on every target, ground truth
 *        for the SIMD equivalence tests.
 */
#include "kernels/distance.hpp"

namespace vdb {

float L2SqScalar(const float* a, const float* b, std::size_t stride) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < stride; ++i) {
        const float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

float DotScalar(const float* a, const float* b, std::size_t stride) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < stride; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

}  // namespace vdb
