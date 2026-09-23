/**
 * @file vdb_c.h
 * @brief Minimal C ABI over vdb — the P/Invoke surface for .NET.
 *
 * Contract:
 *  - Every fallible function returns an int32 error code (0 = OK, matching
 *    vdb::ErrorCode). No exception ever crosses this boundary.
 *  - vdb_last_error() returns a thread-local message for the last failure
 *    on the calling thread (valid until the next failing call).
 *  - Handles are opaque; every *_destroy is safe on NULL.
 *
 * This is the phase-0 slice (index build/write/open/search). Metadata
 * columns, quantization and the rest arrive with the real NuGet binding.
 */
#ifndef VDB_C_H
#define VDB_C_H

#include <stdint.h>

#if defined(_WIN32)
#if defined(VDB_C_BUILD)
#define VDB_C_API __declspec(dllexport)
#else
#define VDB_C_API __declspec(dllimport)
#endif
#else
#define VDB_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vdb_builder vdb_builder;
typedef struct vdb_index vdb_index;

/* Error codes mirror vdb::ErrorCode. */
enum {
    VDB_OK = 0,
    VDB_INVALID_ARGUMENT = 1,
    VDB_OUT_OF_RANGE = 2,
    VDB_IO = 3,
    VDB_CORRUPTED = 4,
    VDB_UNSUPPORTED = 5,
    VDB_INTERNAL = 6,
};

/* Metrics mirror vdb::Metric. */
enum { VDB_METRIC_L2 = 0, VDB_METRIC_COSINE = 1, VDB_METRIC_DOT = 2 };

VDB_C_API const char* vdb_last_error(void);

VDB_C_API int32_t vdb_builder_create(uint32_t dim, uint8_t metric, uint32_t M,
                                     uint32_t ef_construction, uint64_t seed,
                                     vdb_builder** out_builder);
VDB_C_API int32_t vdb_builder_add(vdb_builder* builder, uint64_t external_id,
                                  const float* vec, uint32_t len);
/* On success the builder is CONSUMED (freed); do not use or destroy it.
 * On failure the builder stays valid and must still be destroyed. */
VDB_C_API int32_t vdb_builder_freeze(vdb_builder* builder, vdb_index** out_index);
VDB_C_API void vdb_builder_destroy(vdb_builder* builder);

VDB_C_API int32_t vdb_index_write(const vdb_index* index, const char* path);
VDB_C_API int32_t vdb_index_open(const char* path, int32_t validate_crcs,
                                 vdb_index** out_index);
VDB_C_API uint32_t vdb_index_count(const vdb_index* index);
/* out_external_ids/out_scores must hold k entries; *out_found receives the
 * actual number written (<= k). */
VDB_C_API int32_t vdb_index_search(const vdb_index* index, const float* query,
                                   uint32_t len, uint32_t k, uint32_t ef,
                                   uint64_t* out_external_ids, float* out_scores,
                                   uint32_t* out_found);
VDB_C_API void vdb_index_destroy(vdb_index* index);

#ifdef __cplusplus
}
#endif

#endif /* VDB_C_H */
