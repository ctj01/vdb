/**
 * @file distance.hpp
 * @brief Distance kernel contract + runtime dispatch.
 *
 * All kernels assume the padded-layout contract from types.hpp: both pointers
 * 64-byte aligned, stride a multiple of 16 floats, padding zeroed. Zeros in
 * the tail contribute nothing to L2 or dot, so kernels always run whole
 * 16-float blocks — no unaligned loads, no scalar tails.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace vdb {

/// stride is in floats and is a multiple of 16.
using KernelFn = float (*)(const float* a, const float* b, std::size_t stride);

struct Kernels {
    KernelFn l2sq;  ///< sum (a-b)^2
    KernelFn dot;   ///< sum a*b
    const char* name;  ///< "avx2" or "scalar" — for logs/bench.
};

/// Scalar reference implementations (ground truth for tests).
float L2SqScalar(const float* a, const float* b, std::size_t stride);
float DotScalar(const float* a, const float* b, std::size_t stride);

/// AVX2+FMA implementations. Only called through dispatch; calling them on a
/// CPU without AVX2 is an illegal instruction. On non-x86 builds they alias
/// the scalar versions.
float L2SqAvx2(const float* a, const float* b, std::size_t stride);
float DotAvx2(const float* a, const float* b, std::size_t stride);

/// CPU-dispatched kernel table, resolved once at startup.
const Kernels& GetKernels();

}  // namespace vdb
