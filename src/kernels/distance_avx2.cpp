/**
 * @file distance_avx2.cpp
 * @brief AVX2+FMA kernels. The ONLY translation unit built with AVX2 flags
 *        (see cmake/simd.cmake) — keeps AVX2 instructions out of the rest of
 *        the binary so runtime dispatch stays safe on older CPUs.
 *
 * Layout contract (types.hpp): 64-byte-aligned pointers, stride multiple of
 * 16 floats, zero padding. Hence aligned loads and no tails. Two independent
 * accumulator chains hide FMA latency (~4 cycles): with one chain every
 * fused op waits on the previous one.
 */
#include "kernels/distance.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

namespace vdb {

namespace {
inline float HorizontalSum(__m256 acc0, __m256 acc1) {
    const __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc),
                          _mm256_extractf128_ps(acc, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}
}  // namespace

float DotAvx2(const float* a, const float* b, std::size_t stride) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (std::size_t i = 0; i < stride; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8),
                               acc1);
    }
    return HorizontalSum(acc0, acc1);
}

float L2SqAvx2(const float* a, const float* b, std::size_t stride) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (std::size_t i = 0; i < stride; i += 16) {
        const __m256 d0 = _mm256_sub_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i));
        const __m256 d1 =
            _mm256_sub_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    }
    return HorizontalSum(acc0, acc1);
}

}  // namespace vdb

#else  // non-x86 (e.g. macOS arm64): alias the scalar versions so the
       // dispatch table always has valid targets. NEON is a future upgrade.

namespace vdb {
float DotAvx2(const float* a, const float* b, std::size_t stride) {
    return DotScalar(a, b, stride);
}
float L2SqAvx2(const float* a, const float* b, std::size_t stride) {
    return L2SqScalar(a, b, stride);
}
}  // namespace vdb

#endif
