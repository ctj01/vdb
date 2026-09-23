/**
 * @file builder.cpp
 * @brief HNSW construction (ported semantics) + Freeze() compaction to CSR.
 */
#include "vdb/builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "common/aligned.hpp"
#include "hnsw/search_core.hpp"
#include "kernels/distance.hpp"

namespace vdb {

using detail::Candidate;
using detail::VisitedList;

namespace {

constexpr float kMinNorm = 1e-6f;
constexpr int kMaxLevel = 60;  // P(level > 60) is astronomically small for M >= 4

/// Level as a pure function of (seed, id): splitmix64 -> u in (0,1] ->
/// geometric with mL = 1/ln(M). No RNG state; reproducible builds.
int LevelFor(uint64_t seed, uint32_t id, double ml) {
    uint64_t z = seed + 0x9e3779b97f4a7c15ull * (static_cast<uint64_t>(id) + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    const double u = (static_cast<double>(z >> 11) + 1.0) * 0x1.0p-53;
    const int level = static_cast<int>(-std::log(u) * ml);
    return std::min(level, kMaxLevel);
}

}  // namespace

// ------------------------------------------------------------ FrozenIndex

struct FrozenIndex::Impl {
    AlignedFloatBuffer vectors;
    std::vector<uint64_t> external_ids;
    std::vector<uint8_t> levels;
    std::vector<uint32_t> level0;
    std::vector<uint32_t> upper_row_start;
    std::vector<uint32_t> upper_offsets;
    std::vector<uint32_t> upper_targets;
    IndexView view;
};

FrozenIndex::FrozenIndex() : impl_(std::make_unique<Impl>()) {}
FrozenIndex::FrozenIndex(FrozenIndex&&) noexcept = default;
FrozenIndex& FrozenIndex::operator=(FrozenIndex&&) noexcept = default;
FrozenIndex::~FrozenIndex() = default;

const IndexView& FrozenIndex::View() const { return impl_->view; }

// ------------------------------------------------------------ IndexBuilder

struct IndexBuilder::Impl {
    uint32_t dim = 0;
    uint32_t stride = 0;
    Metric metric = Metric::kCosine;
    HnswParams params;
    uint32_t mmax0 = 0;  // 2*M
    double ml = 0.0;

    uint32_t count = 0;
    AlignedFloatBuffer vectors;
    std::vector<uint64_t> external_ids;
    std::vector<uint8_t> levels;
    std::vector<uint32_t> level0;              // (1 + 2M) u32 per node
    std::vector<uint32_t> upper;               // per node: level * (1 + M) u32
    std::vector<std::size_t> upper_off;        // SIZE_MAX when level == 0

    uint32_t entry = kNoEntry;
    int max_level = -1;

    const Kernels& kernels = GetKernels();
    VisitedList visited;
    std::vector<Candidate> scratch;

    // -- accessors ---------------------------------------------------------

    const float* Vec(uint32_t id) const {
        return vectors.data() + static_cast<std::size_t>(id) * stride;
    }
    float DistQ(const float* q, uint32_t id) const {
        return detail::MetricDist(metric, kernels, q, Vec(id), stride);
    }
    float DistNodes(uint32_t a, uint32_t b) const {
        return detail::MetricDist(metric, kernels, Vec(a), Vec(b), stride);
    }

    uint32_t* Links(uint32_t id, int layer) {
        if (layer == 0) {
            return level0.data() + static_cast<std::size_t>(id) * (1 + mmax0);
        }
        return upper.data() + upper_off[id] +
               static_cast<std::size_t>(layer - 1) * (1 + params.M);
    }
    uint32_t Cap(int layer) const { return layer == 0 ? mmax0 : params.M; }

    // Graph concept adapter for the shared search core.
    struct Graph {
        const Impl* s;
        uint32_t count() const { return s->count; }
        const float* Vector(uint32_t id) const { return s->Vec(id); }
        float Dist(const float* q, uint32_t id) const { return s->DistQ(q, id); }
        std::pair<const uint32_t*, uint32_t> Neighbors(uint32_t id, int layer) const {
            const uint32_t* l = const_cast<Impl*>(s)->Links(id, layer);
            return {l + 1, l[0]};
        }
    };
    Graph AsGraph() const { return Graph{this}; }

    // -- construction ------------------------------------------------------

    /// Diversity heuristic (Algorithm 4) + keep-pruned refill. `candidates`
    /// must be sorted ascending by distance to the base node.
    std::vector<Candidate> SelectNeighbors(const std::vector<Candidate>& candidates,
                                           uint32_t m) const {
        if (!params.use_heuristic || candidates.size() <= m) {
            return {candidates.begin(),
                    candidates.begin() +
                        static_cast<std::ptrdiff_t>(
                            std::min<std::size_t>(m, candidates.size()))};
        }
        std::vector<Candidate> selected;
        std::vector<Candidate> pruned;
        selected.reserve(m);
        for (const Candidate& c : candidates) {
            if (selected.size() >= m) break;
            bool keep = true;
            for (const Candidate& s : selected) {
                if (DistNodes(c.id, s.id) < c.d) {  // c sits "behind" s
                    keep = false;
                    break;
                }
            }
            if (keep) {
                selected.push_back(c);
            } else if (params.keep_pruned) {
                pruned.push_back(c);
            }
        }
        for (std::size_t i = 0; i < pruned.size() && selected.size() < m; ++i) {
            selected.push_back(pruned[i]);
        }
        return selected;
    }

    /// Adds edge node->cand; on overflow re-prunes with the SAME heuristic
    /// (dropping "the farthest" silently destroys the diverse bridge edges).
    void AddLink(uint32_t node, uint32_t cand, int layer) {
        uint32_t* l = Links(node, layer);
        const uint32_t cap = Cap(layer);
        if (l[0] < cap) {
            l[++l[0]] = cand;
            return;
        }
        std::vector<Candidate> candidates;
        candidates.reserve(cap + 1);
        candidates.push_back({DistNodes(node, cand), cand});
        for (uint32_t j = 1; j <= l[0]; ++j) {
            candidates.push_back({DistNodes(node, l[j]), l[j]});
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.d < b.d; });
        const auto selected = SelectNeighbors(candidates, cap);
        l[0] = static_cast<uint32_t>(selected.size());
        for (std::size_t j = 0; j < selected.size(); ++j) {
            l[1 + j] = selected[j].id;
        }
    }

    void InsertNode(uint32_t id) {
        const int level = levels[id];
        const float* q = Vec(id);
        const Graph g = AsGraph();

        uint32_t ep = entry;
        const int top = max_level;
        for (int layer = top; layer > level; --layer) {
            ep = detail::GreedyClosest(g, q, ep, layer);
        }
        for (int layer = std::min(level, top); layer >= 0; --layer) {
            detail::SearchLayer(g, q, ep, params.ef_construction, layer, visited,
                                scratch);
            const auto neighbors = SelectNeighbors(scratch, params.M);

            uint32_t* l = Links(id, layer);
            l[0] = static_cast<uint32_t>(neighbors.size());
            for (std::size_t j = 0; j < neighbors.size(); ++j) {
                l[1 + j] = neighbors[j].id;
            }
            for (const Candidate& nb : neighbors) {
                AddLink(nb.id, id, layer);
            }
            ep = scratch.front().id;
        }
        if (level > max_level) {
            entry = id;
            max_level = level;
        }
    }
};

IndexBuilder::IndexBuilder() : impl_(std::make_unique<Impl>()) {}
IndexBuilder::IndexBuilder(IndexBuilder&&) noexcept = default;
IndexBuilder& IndexBuilder::operator=(IndexBuilder&&) noexcept = default;
IndexBuilder::~IndexBuilder() = default;

Result<IndexBuilder> IndexBuilder::Create(uint32_t dim, Metric metric,
                                          HnswParams params) {
    if (dim == 0) {
        return MakeError(ErrorCode::kInvalidArgument, "dim must be > 0");
    }
    if (params.M < 2 || params.M > 128) {
        return MakeError(ErrorCode::kInvalidArgument, "M must be in [2, 128]");
    }
    if (params.ef_construction < params.M) {
        return MakeError(ErrorCode::kInvalidArgument,
                         "ef_construction must be >= M");
    }
    IndexBuilder b;
    b.impl_->dim = dim;
    b.impl_->stride = PaddedStride(dim);
    b.impl_->metric = metric;
    b.impl_->params = params;
    b.impl_->mmax0 = 2 * params.M;
    b.impl_->ml = 1.0 / std::log(static_cast<double>(params.M));
    return b;
}

uint32_t IndexBuilder::Count() const { return impl_->count; }

Result<uint32_t> IndexBuilder::Add(uint64_t external_id, std::span<const float> vec) {
    Impl& s = *impl_;
    if (vec.size() != s.dim) {
        return MakeError(ErrorCode::kInvalidArgument, "dimension mismatch");
    }
    if (s.count == UINT32_MAX) {
        return MakeError(ErrorCode::kOutOfRange, "index full (u32 ids)");
    }

    // Validate BEFORE touching storage: Add either fully succeeds or leaves
    // the builder untouched.
    float inv_norm = 1.0f;
    if (s.metric == Metric::kCosine) {
        double n2 = 0.0;
        for (const float x : vec) n2 += static_cast<double>(x) * x;
        const float norm = static_cast<float>(std::sqrt(n2));
        if (!(norm > kMinNorm)) {
            return MakeError(ErrorCode::kInvalidArgument,
                             "zero or invalid norm for cosine metric");
        }
        inv_norm = 1.0f / norm;
    }

    const uint32_t id = s.count;
    const int level = LevelFor(s.params.seed, id, s.ml);

    float* row = s.vectors.append_zeroed(s.stride);
    for (std::size_t i = 0; i < vec.size(); ++i) {
        row[i] = vec[i] * inv_norm;
    }
    s.external_ids.push_back(external_id);
    s.levels.push_back(static_cast<uint8_t>(level));
    s.level0.resize(s.level0.size() + 1 + s.mmax0, 0);
    if (level >= 1) {
        s.upper_off.push_back(s.upper.size());
        s.upper.resize(s.upper.size() +
                           static_cast<std::size_t>(level) * (1 + s.params.M),
                       0);
    } else {
        s.upper_off.push_back(SIZE_MAX);
    }
    ++s.count;

    if (s.count == 1) {
        s.entry = id;
        s.max_level = level;
        return id;
    }
    s.InsertNode(id);
    return id;
}

Result<FrozenIndex> IndexBuilder::Freeze() && {
    Impl& s = *impl_;
    FrozenIndex out;
    FrozenIndex::Impl& f = *out.impl_;

    // Upper layers: mutable per-node blocks -> immutable CSR.
    f.upper_row_start.resize(s.count, 0);
    uint32_t rows = 0;
    for (uint32_t i = 0; i < s.count; ++i) {
        f.upper_row_start[i] = rows;
        rows += s.levels[i];
    }
    f.upper_offsets.resize(rows + 1, 0);
    uint32_t nnz = 0;
    uint32_t row = 0;
    for (uint32_t i = 0; i < s.count; ++i) {
        for (int layer = 1; layer <= s.levels[i]; ++layer, ++row) {
            f.upper_offsets[row] = nnz;
            nnz += s.Links(i, layer)[0];
        }
    }
    f.upper_offsets[rows] = nnz;
    f.upper_targets.resize(nnz);
    row = 0;
    for (uint32_t i = 0; i < s.count; ++i) {
        for (int layer = 1; layer <= s.levels[i]; ++layer, ++row) {
            const uint32_t* l = s.Links(i, layer);
            std::memcpy(f.upper_targets.data() + f.upper_offsets[row], l + 1,
                        l[0] * sizeof(uint32_t));
        }
    }

    // Vectors, ids, levels and layer 0 already have the frozen layout: move.
    f.vectors = std::move(s.vectors);
    f.external_ids = std::move(s.external_ids);
    f.levels = std::move(s.levels);
    f.level0 = std::move(s.level0);

    IndexView& v = f.view;
    v.count = s.count;
    v.dim = s.dim;
    v.stride = s.stride;
    v.M = s.params.M;
    v.metric = s.metric;
    v.entry = s.entry;
    v.max_level = s.max_level;
    v.vectors = f.vectors.data();
    v.external_ids = f.external_ids.data();
    v.level0 = f.level0.data();
    v.levels = f.levels.data();
    v.upper_row_start = f.upper_row_start.data();
    v.upper_offsets = f.upper_offsets.data();
    v.upper_targets = f.upper_targets.data();
    return out;
}

}  // namespace vdb
