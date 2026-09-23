/**
 * @file harness.hpp
 * @brief Minimal zero-dependency test harness (CTest drives the executables;
 *        non-zero exit = FAIL). CHECK reports and continues.
 */
#pragma once

#include <cmath>
#include <cstdio>

inline int g_failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

/// For Result<T>: asserts failure with the expected code.
#define CHECK_ERR(expr, expected_code)                                       \
    do {                                                                     \
        auto r_ = (expr);                                                    \
        if (r_.ok() || r_.error().code != (expected_code)) {                 \
            std::printf("FAIL %s:%d: expected error %d from %s\n", __FILE__, \
                        __LINE__, static_cast<int>(expected_code), #expr);   \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

inline bool ApproxEq(double a, double b, double rel, double abs_floor) {
    const double diff = std::fabs(a - b);
    const double scale = std::fmax(std::fabs(a), std::fabs(b));
    return diff <= std::fmax(rel * scale, abs_floor);
}

inline int TestSummary(const char* name) {
    if (g_failures == 0) {
        std::printf("%s: all checks passed\n", name);
    } else {
        std::printf("%s: %d check(s) FAILED\n", name, g_failures);
    }
    return g_failures;
}
