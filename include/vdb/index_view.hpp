/**
 * @file index_view.hpp
 * @brief Non-owning view over a frozen index — THE central abstraction.
 *
 * The search code takes only an IndexView. An index built in memory
 * (IndexBuilder::Freeze) and a segment opened with mmap (phase 2) expose the
 * exact same view over their buffers, so one search implementation serves
 * both. This is what makes "in-memory layout == on-disk format" real.
 *
 * Layout contract:
 *  - vectors: [count][stride] float32, stride = PaddedStride(dim), padding
 *    zeroed, base 64-byte aligned (so every row is 64-byte aligned).
 *  - level0:  per node, a fixed slot of (1 + 2*M) u32: {n, nbr[2*M]}.
 *  - upper layers: levels[count] (u8, number of layers above 0) + CSR:
 *    the row for (node, layer L>=1) is upper_row_start[node] + (L-1);
 *    its neighbors are upper_targets[upper_offsets[row] .. upper_offsets[row+1]).
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "vdb/types.hpp"

namespace vdb {

struct IndexView {
    uint32_t count = 0;
    uint32_t dim = 0;
    uint32_t stride = 0;  ///< In floats; multiple of 16.
    uint32_t M = 0;       ///< Layer-0 slot is (1 + 2*M) u32.
    Metric metric = Metric::kCosine;
    uint32_t entry = kNoEntry;
    int32_t max_level = -1;  ///< -1 while empty.

    const float* vectors = nullptr;
    const uint64_t* external_ids = nullptr;
    const uint32_t* level0 = nullptr;
    const uint8_t* levels = nullptr;
    const uint32_t* upper_row_start = nullptr;
    const uint32_t* upper_offsets = nullptr;  ///< [upper_rows + 1]
    const uint32_t* upper_targets = nullptr;

    uint32_t Level0Stride() const noexcept { return 1 + 2 * M; }
    const float* Vector(uint32_t id) const noexcept {
        return vectors + static_cast<std::size_t>(id) * stride;
    }
};

}  // namespace vdb
