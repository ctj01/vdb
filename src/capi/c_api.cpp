/**
 * @file c_api.cpp
 * @brief C ABI implementation: thin translation between C handles/codes and
 *        the C++ Result-based API. The catch-all guards are belt-and-braces —
 *        the C++ layer does not throw by design, but std::bad_alloc and
 *        friends must still never cross the ABI.
 */
#include "vdb/vdb_c.h"

#include <optional>
#include <span>
#include <string>

#include "vdb/builder.hpp"
#include "vdb/search.hpp"
#include "vdb/segment.hpp"

namespace {

thread_local std::string tls_error;

int32_t Fail(const vdb::Error& e) {
    tls_error = e.message;
    return static_cast<int32_t>(e.code);
}

int32_t Fail(int32_t code, const char* msg) {
    tls_error = msg;
    return code;
}

}  // namespace

/// Opaque handle bodies. vdb_index owns EITHER a frozen in-memory index or
/// an open segment — both expose the same IndexView, so search code below
/// does not care which.
struct vdb_builder {
    vdb::IndexBuilder impl;
};
struct vdb_index {
    std::optional<vdb::FrozenIndex> frozen;
    std::optional<vdb::Segment> segment;
    const vdb::IndexView* view = nullptr;
};

extern "C" {

const char* vdb_last_error(void) { return tls_error.c_str(); }

int32_t vdb_builder_create(uint32_t dim, uint8_t metric, uint32_t M,
                           uint32_t ef_construction, uint64_t seed,
                           vdb_builder** out_builder) {
    if (!out_builder) return Fail(VDB_INVALID_ARGUMENT, "out_builder is null");
    *out_builder = nullptr;
    if (metric > 2) return Fail(VDB_INVALID_ARGUMENT, "bad metric");
    try {
        vdb::HnswParams p;
        if (M != 0) p.M = M;
        if (ef_construction != 0) p.ef_construction = ef_construction;
        p.seed = seed;
        auto r = vdb::IndexBuilder::Create(dim, static_cast<vdb::Metric>(metric), p);
        if (!r.ok()) return Fail(r.error());
        *out_builder = new vdb_builder{std::move(r).value()};
        return VDB_OK;
    } catch (const std::exception& e) {
        return Fail(VDB_INTERNAL, e.what());
    }
}

int32_t vdb_builder_add(vdb_builder* builder, uint64_t external_id,
                        const float* vec, uint32_t len) {
    if (!builder || !vec) return Fail(VDB_INVALID_ARGUMENT, "null argument");
    try {
        auto r = builder->impl.Add(external_id, std::span<const float>(vec, len));
        if (!r.ok()) return Fail(r.error());
        return VDB_OK;
    } catch (const std::exception& e) {
        return Fail(VDB_INTERNAL, e.what());
    }
}

int32_t vdb_builder_freeze(vdb_builder* builder, vdb_index** out_index) {
    if (!builder || !out_index) return Fail(VDB_INVALID_ARGUMENT, "null argument");
    *out_index = nullptr;
    try {
        auto r = std::move(builder->impl).Freeze();
        if (!r.ok()) return Fail(r.error());
        auto* idx = new vdb_index;
        idx->frozen.emplace(std::move(r).value());
        idx->view = &idx->frozen->View();
        *out_index = idx;
        delete builder;  // consumed on success (documented contract)
        return VDB_OK;
    } catch (const std::exception& e) {
        return Fail(VDB_INTERNAL, e.what());
    }
}

void vdb_builder_destroy(vdb_builder* builder) { delete builder; }

int32_t vdb_index_write(const vdb_index* index, const char* path) {
    if (!index || !path) return Fail(VDB_INVALID_ARGUMENT, "null argument");
    try {
        auto r = vdb::WriteSegment(path, *index->view);
        if (!r.ok()) return Fail(r.error());
        return VDB_OK;
    } catch (const std::exception& e) {
        return Fail(VDB_INTERNAL, e.what());
    }
}

int32_t vdb_index_open(const char* path, int32_t validate_crcs,
                       vdb_index** out_index) {
    if (!path || !out_index) return Fail(VDB_INVALID_ARGUMENT, "null argument");
    *out_index = nullptr;
    try {
        vdb::OpenOptions opts;
        opts.validate_section_crcs = validate_crcs != 0;
        auto r = vdb::Segment::Open(path, opts);
        if (!r.ok()) return Fail(r.error());
        auto* idx = new vdb_index;
        idx->segment.emplace(std::move(r).value());
        idx->view = &idx->segment->View();
        *out_index = idx;
        return VDB_OK;
    } catch (const std::exception& e) {
        return Fail(VDB_INTERNAL, e.what());
    }
}

uint32_t vdb_index_count(const vdb_index* index) {
    return index ? index->view->count : 0;
}

int32_t vdb_index_search(const vdb_index* index, const float* query, uint32_t len,
                         uint32_t k, uint32_t ef, uint64_t* out_external_ids,
                         float* out_scores, uint32_t* out_found) {
    if (!index || !query || !out_external_ids || !out_scores || !out_found) {
        return Fail(VDB_INVALID_ARGUMENT, "null argument");
    }
    *out_found = 0;
    try {
        auto r = vdb::Search(*index->view, std::span<const float>(query, len), k, ef);
        if (!r.ok()) return Fail(r.error());
        const auto& hits = r.value();
        for (std::size_t i = 0; i < hits.size(); ++i) {
            out_external_ids[i] = hits[i].external_id;
            out_scores[i] = hits[i].score;
        }
        *out_found = static_cast<uint32_t>(hits.size());
        return VDB_OK;
    } catch (const std::exception& e) {
        return Fail(VDB_INTERNAL, e.what());
    }
}

void vdb_index_destroy(vdb_index* index) { delete index; }

}  // extern "C"
