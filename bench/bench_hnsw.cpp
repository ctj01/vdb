/**
 * @file bench_hnsw.cpp
 * @brief Phase-1 numbers: build rate, recall@10 vs efSearch, latency vs
 *        exact brute-force scan. Clustered corpus (the regime embeddings
 *        live in). Usage: vdb_bench [n] [dim]  (defaults 50000 768)
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include "kernels/distance.hpp"
#include "vdb/builder.hpp"
#include "vdb/search.hpp"

using namespace vdb;
using Clock = std::chrono::steady_clock;

namespace {

double SecondsSince(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

double RecallAtK(const std::vector<uint32_t>& exact,
                 const std::vector<Neighbor>& approx) {
    std::size_t hits = 0;
    for (const uint32_t e : exact) {
        for (const auto& a : approx) {
            if (a.internal_id == e) { ++hits; break; }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(exact.size());
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 50'000;
    const uint32_t dim = argc > 2
                             ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10))
                             : 768;
    constexpr uint32_t kK = 10;
    constexpr int kQueries = 100;

    std::printf("vdb bench: n=%zu dim=%u metric=cosine kernels=%s\n\n", n, dim,
                GetKernels().name);

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> cdist(-1.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.25f);
    constexpr int kClusters = 256;

    std::vector<std::vector<float>> centers(kClusters);
    for (auto& c : centers) {
        c.resize(dim);
        for (auto& x : c) x = cdist(rng);
    }
    auto make_point = [&](std::size_t cluster) {
        std::vector<float> v = centers[cluster % kClusters];
        for (auto& x : v) x += noise(rng);
        return v;
    };

    std::vector<std::vector<float>> corpus;
    corpus.reserve(n);
    for (std::size_t i = 0; i < n; ++i) corpus.push_back(make_point(i));
    std::vector<std::vector<float>> queries;
    for (int i = 0; i < kQueries; ++i) queries.push_back(make_point(i));

    // ---- build ------------------------------------------------------------
    auto builder = std::move(IndexBuilder::Create(dim, Metric::kCosine).value());
    auto t0 = Clock::now();
    for (uint32_t i = 0; i < n; ++i) {
        auto r = builder.Add(i, corpus[i]);
        if (!r.ok()) { std::printf("add failed\n"); return 1; }
        if ((i + 1) % 10'000 == 0) {
            std::printf("  build %u/%zu (%.1fs)\n", i + 1, n, SecondsSince(t0));
        }
    }
    const double build_s = SecondsSince(t0);
    t0 = Clock::now();
    auto frozen = std::move(std::move(builder).Freeze().value());
    const double freeze_s = SecondsSince(t0);
    std::printf("build: %.1fs (%.0f inserts/s), freeze: %.3fs\n\n", build_s,
                n / build_s, freeze_s);
    const IndexView& view = frozen.View();

    // ---- exact ground truth + brute-force latency --------------------------
    std::vector<std::vector<uint32_t>> exact(kQueries);
    // Brute force over the index's own padded storage (same kernels).
    const auto& k = GetKernels();
    t0 = Clock::now();
    for (int qi = 0; qi < kQueries; ++qi) {
        // Normalize query like the index does.
        std::vector<float> q(view.stride, 0.0f);
        double n2 = 0.0;
        for (uint32_t i = 0; i < dim; ++i) { q[i] = queries[qi][i]; n2 += q[i] * (double)q[i]; }
        const float inv = static_cast<float>(1.0 / std::sqrt(n2));
        for (uint32_t i = 0; i < dim; ++i) q[i] *= inv;

        std::vector<std::pair<float, uint32_t>> scored;
        scored.reserve(view.count);
        for (uint32_t i = 0; i < view.count; ++i) {
            scored.push_back({k.dot(q.data(), view.Vector(i), view.stride), i});
        }
        std::partial_sort(scored.begin(), scored.begin() + kK, scored.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        exact[qi].clear();
        for (uint32_t i = 0; i < kK; ++i) exact[qi].push_back(scored[i].second);
    }
    const double brute_us = SecondsSince(t0) * 1e6 / kQueries;
    std::printf("brute force (exact): %.0f us/query\n\n", brute_us);

    // ---- ef sweep -----------------------------------------------------------
    std::printf("ef     recall@10    us/query   speedup\n");
    for (const uint32_t ef : {10u, 20u, 50u, 100u, 200u, 400u}) {
        double recall = 0.0;
        t0 = Clock::now();
        for (int qi = 0; qi < kQueries; ++qi) {
            auto r = Search(view, queries[qi], kK, ef);
            recall += RecallAtK(exact[qi], r.value());
        }
        const double us = SecondsSince(t0) * 1e6 / kQueries;
        std::printf("%-6u %9.3f %10.0f %9.0fx\n", ef, recall / kQueries, us,
                    brute_us / us);
    }
    return 0;
}
