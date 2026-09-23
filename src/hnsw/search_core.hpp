/**
 * @file search_core.hpp
 * @brief The HNSW search engine, templated over a Graph accessor.
 *
 * Both the build-time graph (mutable fixed-capacity slots) and the frozen
 * IndexView (level0 slots + CSR) satisfy the Graph concept:
 *
 *   uint32_t count() const;
 *   float Dist(const float* padded_query, uint32_t id) const;   // smaller = closer
 *   const float* Vector(uint32_t id) const;                     // for prefetch
 *   std::pair<const uint32_t*, uint32_t> Neighbors(uint32_t id, int layer) const;
 *
 * Ported semantics (validated in the reference project): two-heap beam with
 * the "closest pending worse than worst kept" stop condition, epoch-stamped
 * visited list, greedy ef=1 descent on upper layers, next-neighbor prefetch.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>  // _mm_prefetch: SSE baseline on x86-64
#define VDB_PREFETCH(p) _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0)
#else
#define VDB_PREFETCH(p) ((void)0)
#endif

#include "kernels/distance.hpp"
#include "vdb/types.hpp"

namespace vdb::detail {

struct Candidate {
    float d;
    uint32_t id;
};

struct CloserFirst {  // min-heap by distance: expansion frontier
    bool operator()(const Candidate& a, const Candidate& b) const { return a.d > b.d; }
};
struct FartherFirst {  // max-heap by distance: bounded result set
    bool operator()(const Candidate& a, const Candidate& b) const { return a.d < b.d; }
};

/// Epoch-stamped visited set: marking is a store, clearing is ++epoch.
class VisitedList {
public:
    void Reset(uint32_t n) {
        if (stamp_.size() < n) stamp_.resize(n, 0);
        if (++epoch_ == 0) {  // u32 wraparound: once per ~4B queries
            std::fill(stamp_.begin(), stamp_.end(), 0);
            epoch_ = 1;
        }
    }
    bool TestAndSet(uint32_t id) {
        if (stamp_[id] == epoch_) return true;
        stamp_[id] = epoch_;
        return false;
    }

private:
    std::vector<uint32_t> stamp_;
    uint32_t epoch_ = 0;
};

/// Internal distance (smaller = closer) for any metric over PADDED rows.
inline float MetricDist(Metric metric, const Kernels& k, const float* a,
                        const float* b, std::size_t stride) {
    switch (metric) {
        case Metric::kL2:
            return k.l2sq(a, b, stride);
        case Metric::kCosine:
            return 1.0f - k.dot(a, b, stride);  // rows are unit-norm
        case Metric::kDot:
        default:
            return -k.dot(a, b, stride);
    }
}

/// Maps the internal distance back to the metric-native user score.
inline float DistToScore(Metric metric, float d) {
    switch (metric) {
        case Metric::kL2:
            return d;  // squared euclidean, documented
        case Metric::kCosine:
            return 1.0f - d;
        case Metric::kDot:
        default:
            return -d;
    }
}

template <typename Graph>
uint32_t GreedyClosest(const Graph& g, const float* q, uint32_t ep, int layer) {
    float best = g.Dist(q, ep);
    bool improved = true;
    while (improved) {
        improved = false;
        const auto [ids, n] = g.Neighbors(ep, layer);
        for (uint32_t j = 0; j < n; ++j) {
            const float d = g.Dist(q, ids[j]);
            if (d < best) {
                best = d;
                ep = ids[j];
                improved = true;
            }
        }
    }
    return ep;
}

/// Beam search within one layer; fills `out` ascending by distance.
template <typename Graph>
void SearchLayer(const Graph& g, const float* q, uint32_t ep, uint32_t ef, int layer,
                 VisitedList& visited, std::vector<Candidate>& out) {
    visited.Reset(g.count());

    std::priority_queue<Candidate, std::vector<Candidate>, CloserFirst> frontier;
    std::priority_queue<Candidate, std::vector<Candidate>, FartherFirst> best;

    const float d0 = g.Dist(q, ep);
    frontier.push({d0, ep});
    best.push({d0, ep});
    visited.TestAndSet(ep);

    while (!frontier.empty()) {
        const Candidate c = frontier.top();
        if (c.d > best.top().d && best.size() >= ef) break;
        frontier.pop();

        const auto [ids, n] = g.Neighbors(c.id, layer);
        for (uint32_t j = 0; j < n; ++j) {
            if (j + 1 < n) VDB_PREFETCH(g.Vector(ids[j + 1]));
            const uint32_t nb = ids[j];
            if (visited.TestAndSet(nb)) continue;
            const float d = g.Dist(q, nb);
            if (best.size() < ef || d < best.top().d) {
                frontier.push({d, nb});
                best.push({d, nb});
                if (best.size() > ef) best.pop();
            }
        }
    }

    out.clear();
    out.reserve(best.size());
    while (!best.empty()) {
        out.push_back(best.top());
        best.pop();
    }
    std::reverse(out.begin(), out.end());
}

}  // namespace vdb::detail
