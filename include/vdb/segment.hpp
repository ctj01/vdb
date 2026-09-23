/**
 * @file segment.hpp
 * @brief On-disk segment (.vdb): atomic writer + zero-copy mmap reader.
 *
 * A segment is an immutable snapshot of a frozen index plus its metadata
 * columns. The reader validates structure and CRCs at open and then serves
 * searches DIRECTLY over the mapped memory — the same IndexView / Search()
 * path used for in-memory indexes; no data is copied.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vdb/index_view.hpp"
#include "vdb/meta.hpp"
#include "vdb/result.hpp"

namespace vdb {

/**
 * @brief Serializes a frozen view + metadata to `path`, atomically
 *        (tmp + fsync + rename; on POSIX also fsync of the directory).
 * @param meta Optional columns; each must have exactly view.count values.
 */
Result<void> WriteSegment(const std::string& path, const IndexView& view,
                          const std::vector<MetaColumn>& meta = {});

struct OpenOptions {
    /// When false, CRC validation of section payloads is skipped (the
    /// 128-byte header CRC and all structural bounds checks always run).
    /// Escape hatch for very large segments where a full CRC pass at open
    /// is too expensive.
    bool validate_section_crcs = true;
};

/// An open segment: owns the mapping, exposes the view + metadata. Move-only.
class Segment {
public:
    static Result<Segment> Open(const std::string& path, OpenOptions options = {});

    Segment(Segment&&) noexcept;
    Segment& operator=(Segment&&) noexcept;
    ~Segment();

    const IndexView& View() const;
    const std::vector<MetaColumnView>& Meta() const;

private:
    Segment();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vdb
