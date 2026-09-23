/**
 * @file test_kernels.cpp
 * @brief SIMD vs scalar vs double-precision reference, under the padded
 *        layout contract (64B-aligned, stride multiple of 16, zero padding).
 *
 * Tolerance rationale (documented per the spec): float summation error grows
 * ~O(n * eps * magnitude) and FMA rounds once per multiply-add, so SIMD and
 * scalar legitimately differ in the last bits. Both are compared against a
 * double-precision reference with relative tolerance 1e-4 (generous for
 * dim <= 1536; real kernel bugs — dropped lanes, bad tails — show up as
 * errors orders of magnitude larger).
 */
#include <random>
#include <vector>

#include "common/aligned.hpp"
#include "harness.hpp"
#include "kernels/distance.hpp"
#include "vdb/types.hpp"

using namespace vdb;

namespace {

double DotRef(const float* a, const float* b, std::size_t n) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        acc += static_cast<double>(a[i]) * b[i];
    }
    return acc;
}

double L2Ref(const float* a, const float* b, std::size_t n) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        acc += d * d;
    }
    return acc;
}

}  // namespace

int main() {
    std::printf("kernels: %s\n", GetKernels().name);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Dims chosen to exercise padding: exact multiples of 16, one-off, tiny,
    // and realistic embedding sizes.
    const uint32_t dims[] = {1, 3, 7, 8, 15, 16, 17, 24, 31, 32, 100, 384, 768, 1000};

    for (const uint32_t dim : dims) {
        const uint32_t stride = PaddedStride(dim);
        CHECK(stride % 16 == 0);
        CHECK(stride >= dim);

        for (int rep = 0; rep < 20; ++rep) {
            // One buffer per vector: a second append on the same buffer may
            // reallocate and dangle the first pointer.
            AlignedFloatBuffer buf_a, buf_b;
            float* a = buf_a.append_zeroed(stride);
            float* b = buf_b.append_zeroed(stride);
            for (uint32_t i = 0; i < dim; ++i) a[i] = dist(rng);
            for (uint32_t i = 0; i < dim; ++i) b[i] = dist(rng);
            // Padding stays zero: kernels run full stride, results must match
            // the reference computed over dim only.
            const double dref = DotRef(a, b, dim);
            const double lref = L2Ref(a, b, dim);

            CHECK(ApproxEq(DotScalar(a, b, stride), dref, 1e-4, 1e-5));
            CHECK(ApproxEq(DotAvx2(a, b, stride), dref, 1e-4, 1e-5));
            CHECK(ApproxEq(L2SqScalar(a, b, stride), lref, 1e-4, 1e-5));
            CHECK(ApproxEq(L2SqAvx2(a, b, stride), lref, 1e-4, 1e-5));

            // Identities.
            CHECK(L2SqAvx2(a, a, stride) == 0.0f);
            CHECK(ApproxEq(DotAvx2(a, a, stride), DotRef(a, a, dim), 1e-4, 1e-6));
        }
    }
    return TestSummary("test_kernels");
}
