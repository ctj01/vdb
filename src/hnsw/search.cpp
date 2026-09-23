/**
 * @file search.cpp
 * @brief Public query API: Search(IndexView, ...) — the single search path
 *        for in-memory and (phase 2) memory-mapped indexes.
 */
#include "vdb/search.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

#include "hnsw/search_core.hpp"
#include "kernels/distance.hpp"

namespace vdb {

namespace {

constexpr float kMinNorm = 1e-6f;

/// Graph concept adapter over the frozen view (level0 slots + upper CSR).
struct ViewGraph {
    const IndexView& v;
    const Kernels& k;

    uint32_t count() const { return v.count; }
    const float* Vector(uint32_t id) const { return v.Vector(id); }
    float Dist(const float* q, uint32_t id) const {
        return detail::MetricDist(v.metric, k, q, v.Vector(id), v.stride);
    }
    std::pair<const uint32_t*, uint32_t> Neighbors(uint32_t id, int layer) const {
        if (layer == 0) {
            const uint32_t* l =
                v.level0 + static_cast<std::size_t>(id) * v.Level0Stride();
            return {l + 1, l[0]};
        }
        const uint32_t r = v.upper_row_start[id] + static_cast<uint32_t>(layer - 1);
        const uint32_t begin = v.upper_offsets[r];
        return {v.upper_targets + begin, v.upper_offsets[r + 1] - begin};
    }
};

/// Per-thread 64B-aligned query buffer (padded copy of the caller's query).
/// thread_local => concurrent searches share nothing; no locks anywhere.
class QueryScratch {
public:
    ~QueryScratch() { Release(); }
    float* Prepare(uint32_t stride) {
        if (cap_ < stride) {
            Release();
            data_ = static_cast<float*>(
                ::operator new[](stride * sizeof(float), std::align_val_t{kAlign}));
            cap_ = stride;
        }
        return data_;
    }

private:
    void Release() {
        if (data_) ::operator delete[](data_, std::align_val_t{kAlign});
        data_ = nullptr;
        cap_ = 0;
    }
    float* data_ = nullptr;
    uint32_t cap_ = 0;
};

}  // namespace

Result<std::vector<Neighbor>> Search(const IndexView& view,
                                     std::span<const float> query, uint32_t k,
                                     uint32_t ef) {
    if (query.size() != view.dim) {
        return MakeError(ErrorCode::kInvalidArgument, "query dimension mismatch");
    }
    if (view.count == 0 || k == 0) {
        return std::vector<Neighbor>{};
    }
    if (view.entry == kNoEntry || view.entry >= view.count) {
        return MakeError(ErrorCode::kCorrupted, "invalid entry point");
    }
    k = std::min(k, view.count);
    ef = std::max(ef, k);

    thread_local QueryScratch scratch;
    float* q = scratch.Prepare(view.stride);
    std::memcpy(q, query.data(), view.dim * sizeof(float));
    std::memset(q + view.dim, 0, (view.stride - view.dim) * sizeof(float));

    if (view.metric == Metric::kCosine) {
        double n2 = 0.0;
        for (uint32_t i = 0; i < view.dim; ++i) {
            n2 += static_cast<double>(q[i]) * q[i];
        }
        const float norm = static_cast<float>(std::sqrt(n2));
        if (!(norm > kMinNorm)) {
            return MakeError(ErrorCode::kInvalidArgument,
                             "zero-norm query with cosine metric");
        }
        const float inv = 1.0f / norm;
        for (uint32_t i = 0; i < view.dim; ++i) q[i] *= inv;
    }

    const ViewGraph g{view, GetKernels()};

    uint32_t ep = view.entry;
    for (int layer = view.max_level; layer >= 1; --layer) {
        ep = detail::GreedyClosest(g, q, ep, layer);
    }

    thread_local detail::VisitedList visited;
    thread_local std::vector<detail::Candidate> found;
    detail::SearchLayer(g, q, ep, ef, 0, visited, found);

    std::vector<Neighbor> out;
    out.reserve(std::min<std::size_t>(k, found.size()));
    for (std::size_t i = 0; i < found.size() && i < k; ++i) {
        const auto& c = found[i];
        out.push_back({view.external_ids[c.id], c.id,
                       detail::DistToScore(view.metric, c.d)});
    }
    return out;
}

}  // namespace vdb
