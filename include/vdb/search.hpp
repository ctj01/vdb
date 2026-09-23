/**
 * @file search.hpp
 * @brief Query API over an IndexView (owned or memory-mapped alike).
 */
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "vdb/index_view.hpp"
#include "vdb/result.hpp"
#include "vdb/types.hpp"

namespace vdb {

/**
 * @brief Approximate top-k search.
 *
 * @param view Frozen index view. An empty view (count == 0) yields an empty
 *        result, not an error.
 * @param query Raw query of view.dim floats (normalized internally for
 *        cosine; zero-norm cosine queries are kInvalidArgument).
 * @param k Number of neighbors requested (clamped to view.count).
 * @param ef Beam width at layer 0 (clamped up to k). Recall/latency knob.
 * @return Neighbors best-first (metric-native score semantics; see types.hpp).
 * @note Thread-safe from any number of threads on a frozen view (lock-free;
 *       per-thread scratch).
 */
Result<std::vector<Neighbor>> Search(const IndexView& view,
                                     std::span<const float> query, uint32_t k,
                                     uint32_t ef = 100);

}  // namespace vdb
