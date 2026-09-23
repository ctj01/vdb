/**
 * @file dispatch.cpp
 * @brief Runtime CPU-feature detection, portable across GCC/Clang/MSVC.
 *
 * The reference project used __builtin_cpu_supports (GCC-only). Here the
 * MSVC path goes through __cpuidex + _xgetbv: AVX2 requires not only the
 * CPUID feature bits (AVX2 + FMA) but also OS support for saving YMM state
 * (OSXSAVE + XCR0 bits 1|2) — skipping that check crashes on rare
 * hypervisor/OS configurations even when CPUID says AVX2.
 */
#include "kernels/distance.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

namespace vdb {

namespace {

bool DetectAvx2Fma() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) return false;
    __cpuid(info, 1);
    const bool fma = (info[2] & (1 << 12)) != 0;
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    if (!fma || !osxsave) return false;
    // OS must save YMM registers across context switches.
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) return false;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;  // AVX2
#else
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
#else
    return false;  // non-x86: scalar (the "avx2" symbols alias scalar anyway)
#endif
}

Kernels MakeKernels() {
    if (DetectAvx2Fma()) {
        return Kernels{&L2SqAvx2, &DotAvx2, "avx2"};
    }
    return Kernels{&L2SqScalar, &DotScalar, "scalar"};
}

}  // namespace

const Kernels& GetKernels() {
    // Magic static: initialized exactly once, thread-safe, no locks after.
    static const Kernels k = MakeKernels();
    return k;
}

}  // namespace vdb
