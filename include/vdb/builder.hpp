/**
 * @file builder.hpp
 * @brief In-memory HNSW construction: IndexBuilder -> Freeze() -> FrozenIndex.
 *
 * Build-time adjacency uses mutable fixed-capacity slots (reverse-edge
 * re-pruning rewrites lists constantly; a CSR cannot mutate). Freeze()
 * compacts the upper layers into the immutable CSR the IndexView (and the
 * phase-2 disk format) requires. Layer 0 already uses the frozen slot layout
 * during build, so freezing moves it without copying.
 *
 * Both types are pimpl'd: public headers stay dependency-free, which is also
 * the shape the future C ABI wants.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "vdb/index_view.hpp"
#include "vdb/result.hpp"
#include "vdb/types.hpp"

namespace vdb {

/// Owns the frozen buffers and exposes the IndexView over them. Move-only.
class FrozenIndex {
public:
    FrozenIndex(FrozenIndex&&) noexcept;
    FrozenIndex& operator=(FrozenIndex&&) noexcept;
    ~FrozenIndex();

    const IndexView& View() const;

private:
    friend class IndexBuilder;
    FrozenIndex();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class IndexBuilder {
public:
    /**
     * @brief Validates parameters and creates an empty builder.
     * @param dim Vector dimensionality (> 0; need not be a multiple of 16 —
     *        storage pads to PaddedStride(dim)).
     * @param metric For kCosine, vectors are L2-normalized on Add.
     */
    static Result<IndexBuilder> Create(uint32_t dim, Metric metric,
                                       HnswParams params = {});

    IndexBuilder(IndexBuilder&&) noexcept;
    IndexBuilder& operator=(IndexBuilder&&) noexcept;
    ~IndexBuilder();

    /**
     * @brief Copies one vector into the index (padded, normalized if cosine)
     *        and wires it into the graph. Single-threaded.
     * @return The dense internal id assigned (insertion order, 0-based).
     */
    Result<uint32_t> Add(uint64_t external_id, std::span<const float> vec);

    uint32_t Count() const;

    /**
     * @brief Compacts to the immutable layout and hands over ownership.
     *        Consumes the builder (call on an rvalue: std::move(b).Freeze()).
     */
    Result<FrozenIndex> Freeze() &&;

private:
    IndexBuilder();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vdb
