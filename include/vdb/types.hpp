/**
 * @file types.hpp
 * @brief Core value types shared across the library.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace vdb {

/// Distance metric. Numeric values are part of the on-disk format — never
/// renumber.
enum class Metric : uint8_t {
    kL2 = 0,      ///< Squared Euclidean distance (smaller = closer).
    kCosine = 1,  ///< Cosine similarity; vectors are L2-normalized on insert.
    kDot = 2,     ///< Inner product (larger = closer).
};

/// HNSW tunables. M is the max degree on layers >= 1; layer 0 caps at 2*M.
struct HnswParams {
    uint32_t M = 16;
    uint32_t ef_construction = 200;
    uint64_t seed = 42;           ///< Level assignment is a pure f(seed, id).
    bool use_heuristic = true;    ///< Diversity heuristic vs naive M-closest.
    bool keep_pruned = true;      ///< Refill up to M with pruned candidates.
};

/// Sentinel for "no entry point" (empty index). Serialized as-is.
inline constexpr uint32_t kNoEntry = UINT32_MAX;

/// Every buffer the search path touches is aligned to this.
inline constexpr std::size_t kAlign = 64;

/// Vector rows are padded to a multiple of 16 floats (64 bytes), zeros in the
/// tail. With 64-byte-aligned bases, every row starts cache-line-aligned and
/// SIMD kernels need neither unaligned loads nor scalar tails.
inline constexpr uint32_t PaddedStride(uint32_t dim) {
    return (dim + 15u) & ~15u;
}

/// One search hit.
struct Neighbor {
    uint64_t external_id;
    uint32_t internal_id;
    float score;  ///< Metric-native: L2 -> squared distance, cosine/dot -> similarity.
};

}  // namespace vdb
