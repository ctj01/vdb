/**
 * @file test_hnsw.cpp
 * @brief Recall vs exact brute force across dim/M/ef/metric combinations,
 *        on uniform and clustered data, plus edge cases and CSR integrity.
 *
 * The oracle computes exact scores in double precision over the RAW input
 * vectors — an independent path from the index's padded float storage.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "harness.hpp"
#include "vdb/builder.hpp"
#include "vdb/search.hpp"

using namespace vdb;

namespace {

using Vec = std::vector<float>;

Vec RandomVec(std::mt19937& rng, uint32_t dim) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Vec v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

std::vector<Vec> UniformCorpus(std::mt19937& rng, std::size_t n, uint32_t dim) {
    std::vector<Vec> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) out.push_back(RandomVec(rng, dim));
    return out;
}

std::vector<Vec> ClusteredCorpus(std::mt19937& rng, std::size_t n, uint32_t dim,
                                 int n_clusters) {
    std::normal_distribution<float> noise(0.0f, 0.15f);
    std::vector<Vec> centers;
    for (int c = 0; c < n_clusters; ++c) centers.push_back(RandomVec(rng, dim));
    std::vector<Vec> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        Vec v = centers[i % n_clusters];
        for (auto& x : v) x += noise(rng);
        out.push_back(std::move(v));
    }
    return out;
}

/// Exact metric-native score in double (bigger = better for cosine/dot,
/// smaller = better for L2 — hence the comparator flip below).
double ExactScore(Metric m, const Vec& q, const Vec& v) {
    double dot = 0.0, nq = 0.0, nv = 0.0, l2 = 0.0;
    for (std::size_t i = 0; i < q.size(); ++i) {
        dot += static_cast<double>(q[i]) * v[i];
        nq += static_cast<double>(q[i]) * q[i];
        nv += static_cast<double>(v[i]) * v[i];
        const double d = static_cast<double>(q[i]) - v[i];
        l2 += d * d;
    }
    switch (m) {
        case Metric::kL2: return l2;
        case Metric::kCosine: return dot / (std::sqrt(nq) * std::sqrt(nv));
        case Metric::kDot: default: return dot;
    }
}

std::vector<uint32_t> ExactTopK(Metric m, const Vec& q, const std::vector<Vec>& corpus,
                                std::size_t k) {
    std::vector<std::pair<double, uint32_t>> scored;
    scored.reserve(corpus.size());
    for (uint32_t i = 0; i < corpus.size(); ++i) {
        scored.push_back({ExactScore(m, q, corpus[i]), i});
    }
    const bool smaller_better = (m == Metric::kL2);
    std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(k),
                      scored.end(), [&](const auto& a, const auto& b) {
                          return smaller_better ? a.first < b.first
                                                : a.first > b.first;
                      });
    std::vector<uint32_t> ids;
    for (std::size_t i = 0; i < k; ++i) ids.push_back(scored[i].second);
    return ids;
}

double RecallAtK(const std::vector<uint32_t>& exact,
                 const std::vector<Neighbor>& approx) {
    std::size_t hits = 0;
    for (const uint32_t e : exact) {
        for (const auto& a : approx) {
            if (a.internal_id == e) {
                ++hits;
                break;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(exact.size());
}

FrozenIndex BuildIndex(const std::vector<Vec>& corpus, uint32_t dim, Metric metric,
                       uint32_t M) {
    HnswParams p;
    p.M = M;
    auto rb = IndexBuilder::Create(dim, metric, p);
    CHECK(rb.ok());
    IndexBuilder b = std::move(rb).value();
    for (uint32_t i = 0; i < corpus.size(); ++i) {
        auto ra = b.Add(1000 + i, corpus[i]);
        CHECK(ra.ok());
        CHECK(ra.value() == i);
    }
    auto rf = std::move(b).Freeze();
    CHECK(rf.ok());
    return std::move(rf).value();
}

struct Combo {
    uint32_t dim;
    uint32_t M;
    Metric metric;
    bool clustered;
    const char* label;
};

}  // namespace

int main() {
    constexpr std::size_t kN = 2000;
    constexpr std::size_t kK = 10;
    constexpr int kQueries = 30;
    std::mt19937 rng(123);

    // dim=24 deliberately not a multiple of 16 (stride padding path).
    const Combo combos[] = {
        {24, 8, Metric::kCosine, false, "d24/M8/cos/uniform"},
        {24, 8, Metric::kL2, true, "d24/M8/l2/clustered"},
        {100, 16, Metric::kCosine, true, "d100/M16/cos/clustered"},
        {100, 16, Metric::kL2, false, "d100/M16/l2/uniform"},
        {384, 32, Metric::kCosine, false, "d384/M32/cos/uniform"},
        {384, 16, Metric::kL2, true, "d384/M16/l2/clustered"},
        {100, 16, Metric::kDot, false, "d100/M16/dot/uniform"},
    };

    for (const Combo& c : combos) {
        const auto corpus = c.clustered ? ClusteredCorpus(rng, kN, c.dim, 24)
                                        : UniformCorpus(rng, kN, c.dim);
        const auto index = BuildIndex(corpus, c.dim, c.metric, c.M);
        CHECK(index.View().count == kN);

        double recall_50 = 0.0, recall_100 = 0.0;
        for (int qi = 0; qi < kQueries; ++qi) {
            const Vec q = c.clustered
                              ? [&] {
                                    Vec v = corpus[static_cast<std::size_t>(qi) * 7];
                                    std::normal_distribution<float> noise(0.0f, 0.1f);
                                    for (auto& x : v) x += noise(rng);
                                    return v;
                                }()
                              : RandomVec(rng, c.dim);
            const auto exact = ExactTopK(c.metric, q, corpus, kK);
            auto r50 = Search(index.View(), q, kK, 50);
            auto r100 = Search(index.View(), q, kK, 100);
            CHECK(r50.ok());
            CHECK(r100.ok());
            recall_50 += RecallAtK(exact, r50.value());
            recall_100 += RecallAtK(exact, r100.value());
            // Results sorted best-first; external ids map back.
            const auto& rr = r100.value();
            for (std::size_t j = 0; j < rr.size(); ++j) {
                CHECK(rr[j].external_id == 1000 + rr[j].internal_id);
            }
        }
        recall_50 /= kQueries;
        recall_100 /= kQueries;
        std::printf("%-26s recall@10 ef=50: %.3f  ef=100: %.3f\n", c.label,
                    recall_50, recall_100);
        // Spec threshold. Dot is not a proper metric (MIPS caveat, known
        // HNSW limitation) — still asserted, watch this one.
        CHECK(recall_100 >= 0.95);
    }

    // ---- edge cases --------------------------------------------------------

    {  // count == 0
        auto rb = IndexBuilder::Create(24, Metric::kCosine);
        CHECK(rb.ok());
        auto rf = std::move(rb.value()).Freeze();
        CHECK(rf.ok());
        const auto& view = rf.value().View();
        CHECK(view.count == 0);
        CHECK(view.entry == kNoEntry);
        Vec q(24, 0.5f);
        auto rs = Search(view, q, 5);
        CHECK(rs.ok());
        CHECK(rs.value().empty());
    }
    {  // count == 1
        auto b = std::move(IndexBuilder::Create(24, Metric::kL2).value());
        Vec v(24, 1.0f);
        CHECK(b.Add(7, v).ok());
        auto frozen = std::move(std::move(b).Freeze().value());
        auto rs = Search(frozen.View(), v, 10);
        CHECK(rs.ok());
        CHECK(rs.value().size() == 1);
        CHECK(rs.value()[0].external_id == 7);
        CHECK(rs.value()[0].score == 0.0f);  // exact self-match, L2 == 0
    }
    {  // duplicated vectors must not break construction or search
        std::mt19937 r2(7);
        auto b = std::move(IndexBuilder::Create(32, Metric::kCosine).value());
        const Vec dup = RandomVec(r2, 32);
        for (uint32_t i = 0; i < 50; ++i) CHECK(b.Add(i, dup).ok());
        for (uint32_t i = 50; i < 300; ++i) CHECK(b.Add(i, RandomVec(r2, 32)).ok());
        auto frozen = std::move(std::move(b).Freeze().value());
        auto rs = Search(frozen.View(), dup, 10, 100);
        CHECK(rs.ok());
        CHECK(rs.value().size() == 10);
        CHECK(ApproxEq(rs.value()[0].score, 1.0, 1e-4, 0.0));
        // No id may repeat in one result set.
        for (std::size_t i = 0; i < rs.value().size(); ++i) {
            for (std::size_t j = i + 1; j < rs.value().size(); ++j) {
                CHECK(rs.value()[i].internal_id != rs.value()[j].internal_id);
            }
        }
    }
    {  // deterministic build: same input -> bit-identical results
        std::mt19937 r3(99);
        const auto corpus = UniformCorpus(r3, 500, 24);
        const auto a = BuildIndex(corpus, 24, Metric::kCosine, 16);
        const auto b = BuildIndex(corpus, 24, Metric::kCosine, 16);
        const Vec q = corpus[13];
        const auto ra = Search(a.View(), q, 5).value();
        const auto rb2 = Search(b.View(), q, 5).value();
        CHECK(ra.size() == rb2.size());
        for (std::size_t i = 0; i < ra.size(); ++i) {
            CHECK(ra[i].internal_id == rb2[i].internal_id);
            CHECK(ra[i].score == rb2[i].score);
        }
    }
    {  // CSR integrity on a real build
        std::mt19937 r4(5);
        const auto corpus = UniformCorpus(r4, 1500, 32);
        const auto idx = BuildIndex(corpus, 32, Metric::kL2, 16);
        const IndexView& v = idx.View();
        uint32_t rows = 0;
        for (uint32_t i = 0; i < v.count; ++i) {
            CHECK(v.upper_row_start[i] == rows);
            rows += v.levels[i];
        }
        for (uint32_t r = 0; r < rows; ++r) {
            CHECK(v.upper_offsets[r] <= v.upper_offsets[r + 1]);
            for (uint32_t t = v.upper_offsets[r]; t < v.upper_offsets[r + 1]; ++t) {
                CHECK(v.upper_targets[t] < v.count);
            }
        }
        // Level-0 slots: counts within cap, targets in range.
        for (uint32_t i = 0; i < v.count; ++i) {
            const uint32_t* l = v.level0 + static_cast<std::size_t>(i) * v.Level0Stride();
            CHECK(l[0] <= 2 * v.M);
            for (uint32_t j = 1; j <= l[0]; ++j) CHECK(l[j] < v.count);
        }
    }

    // ---- validation --------------------------------------------------------

    CHECK_ERR(IndexBuilder::Create(0, Metric::kL2), ErrorCode::kInvalidArgument);
    {
        HnswParams bad;
        bad.M = 1;
        CHECK_ERR(IndexBuilder::Create(8, Metric::kL2, bad),
                  ErrorCode::kInvalidArgument);
    }
    {
        auto b = std::move(IndexBuilder::Create(24, Metric::kCosine).value());
        CHECK_ERR(b.Add(0, Vec(23, 1.0f)), ErrorCode::kInvalidArgument);
        CHECK_ERR(b.Add(0, Vec(24, 0.0f)), ErrorCode::kInvalidArgument);  // zero norm
        CHECK(b.Count() == 0);  // failed Adds left no partial state
        CHECK(b.Add(0, Vec(24, 1.0f)).ok());
        auto frozen = std::move(std::move(b).Freeze().value());
        CHECK_ERR(Search(frozen.View(), Vec(23, 1.0f), 3),
                  ErrorCode::kInvalidArgument);
        CHECK_ERR(Search(frozen.View(), Vec(24, 0.0f), 3),
                  ErrorCode::kInvalidArgument);  // zero-norm cosine query
    }

    return TestSummary("test_hnsw");
}
