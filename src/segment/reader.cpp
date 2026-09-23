/**
 * @file reader.cpp
 * @brief Segment open: validate structure + CRCs, expose IndexView and
 *        metadata views DIRECTLY over the mapped memory (zero copies).
 *
 * A segment file is untrusted input: every offset/length is bounds-checked
 * before any pointer into the mapping is formed. Deep payload validation
 * (per-section CRCs, graph target bounds, string offset monotonicity) runs
 * when OpenOptions.validate_section_crcs is set (default) — skipping it is
 * the documented escape hatch for very large segments, and skips the deep
 * checks along with the CRCs.
 *
 * Aliasing note: pointers into the mapping are reinterpret_cast to typed
 * arrays (f32/u32/u64/i64). Offsets are validated to be suitably aligned
 * (sections at 64 bytes, interior arrays at their natural alignment), which
 * is the standard contract for mmap'd formats.
 */
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "io/file.hpp"
#include "segment/crc32c.hpp"
#include "segment/format.hpp"
#include "vdb/segment.hpp"

namespace vdb {

namespace {

Error Corrupted(const std::string& what) {
    return MakeError(ErrorCode::kCorrupted, "segment: " + what);
}

struct RawSection {
    uint64_t off = 0;
    uint64_t len = 0;
    uint32_t crc = 0;
    bool present = false;
};

}  // namespace

struct Segment::Impl {
    io::MappedFile map;
    IndexView view;
    std::vector<MetaColumnView> meta;
};

Segment::Segment() : impl_(std::make_unique<Impl>()) {}
Segment::Segment(Segment&&) noexcept = default;
Segment& Segment::operator=(Segment&&) noexcept = default;
Segment::~Segment() = default;

const IndexView& Segment::View() const { return impl_->view; }
const std::vector<MetaColumnView>& Segment::Meta() const { return impl_->meta; }

Result<Segment> Segment::Open(const std::string& path, OpenOptions options) {
    auto rmap = io::MappedFile::Open(path);
    if (!rmap.ok()) return rmap.error();
    io::MappedFile map = std::move(rmap).value();
    const uint8_t* base = map.data();
    const uint64_t size = map.size();

    // ---- envelope: magic, footer, header CRC, version ---------------------
    if (size < fmt::kHeaderSize + fmt::kFooterSize) {
        return Corrupted("file smaller than header + footer");
    }
    if (std::memcmp(base, fmt::kMagic, sizeof(fmt::kMagic)) != 0) {
        return Corrupted("bad magic");
    }
    if (std::memcmp(base + size - fmt::kFooterSize, fmt::kMagic,
                    sizeof(fmt::kMagic)) != 0) {
        return Corrupted("missing footer (truncated or incomplete write)");
    }
    {
        uint8_t header[fmt::kHeaderSize];
        std::memcpy(header, base, fmt::kHeaderSize);
        const uint32_t stored = fmt::GetU32(header + 52);
        std::memset(header + 52, 0, 4);
        if (Crc32c(header, fmt::kHeaderSize) != stored) {
            return Corrupted("header CRC mismatch");
        }
    }
    const uint16_t version_major = fmt::GetU16(base + 8);
    if (version_major > fmt::kVersionMajor) {
        return MakeError(ErrorCode::kUnsupported,
                         "segment version_major " + std::to_string(version_major) +
                             " > supported " + std::to_string(fmt::kVersionMajor));
    }

    // ---- header fields -----------------------------------------------------
    const uint32_t dim = fmt::GetU32(base + 12);
    const uint8_t metric_raw = base[16];
    const uint32_t M = fmt::GetU32(base + 20);
    const uint64_t count64 = fmt::GetU64(base + 24);
    const uint32_t entry = fmt::GetU32(base + 32);
    const int32_t max_level = fmt::GetI32(base + 36);
    const uint64_t table_off = fmt::GetU64(base + 40);
    const uint32_t section_count = fmt::GetU32(base + 48);

    if (dim == 0 || M < 2 || M > 128 || metric_raw > 2) {
        return Corrupted("invalid dim/M/metric");
    }
    if (count64 > UINT32_MAX - 1) return Corrupted("count exceeds u32 id space");
    const auto count = static_cast<uint32_t>(count64);
    if (count == 0) {
        if (entry != kNoEntry || max_level != -1) {
            return Corrupted("empty segment with entry/max_level set");
        }
    } else if (entry >= count || max_level < 0) {
        return Corrupted("invalid entry point / max_level");
    }

    // ---- section table -------------------------------------------------------
    if (table_off < fmt::kHeaderSize || section_count > 64 ||
        table_off + section_count * fmt::kSectionEntrySize >
            size - fmt::kFooterSize) {
        return Corrupted("section table out of bounds");
    }
    RawSection secs[6];  // indexed by SectionType (1..5)
    for (uint32_t i = 0; i < section_count; ++i) {
        const uint8_t* e = base + table_off + i * fmt::kSectionEntrySize;
        const uint32_t type = fmt::GetU32(e);
        const uint64_t off = fmt::GetU64(e + 4);
        const uint64_t len = fmt::GetU64(e + 12);
        const uint32_t crc = fmt::GetU32(e + 20);
        if (type == 0 || type > 5) continue;  // unknown: forward-compatible skip
        if (off % fmt::kAlign != 0 || off < fmt::kHeaderSize || off + len > table_off) {
            return Corrupted("section bounds/alignment");
        }
        secs[type] = {off, len, crc, true};
    }
    for (uint32_t t = 1; t <= 5; ++t) {
        if (!secs[t].present) {
            return Corrupted("missing section type " + std::to_string(t));
        }
    }

    const bool deep = options.validate_section_crcs;
    if (deep) {
        for (uint32_t t = 1; t <= 5; ++t) {
            if (Crc32c(base + secs[t].off, secs[t].len) != secs[t].crc) {
                return Corrupted("CRC mismatch in section " + std::to_string(t));
            }
        }
    }

    // ---- expected sizes + typed pointers ------------------------------------
    const uint32_t stride = PaddedStride(dim);
    const uint64_t l0_stride = 1 + 2ull * M;

    if (secs[fmt::kIds].len != count64 * 8) return Corrupted("ids section size");
    if (secs[fmt::kVectors].len != count64 * stride * 4) {
        return Corrupted("vectors section size");
    }
    if (secs[fmt::kGraphL0].len != count64 * l0_stride * 4) {
        return Corrupted("level0 section size");
    }

    const uint8_t* up = base + secs[fmt::kGraphUpper].off;
    if (secs[fmt::kGraphUpper].len < 8) return Corrupted("upper section too small");
    const uint32_t rows = fmt::GetU32(up);
    const uint32_t nnz = fmt::GetU32(up + 4);
    const uint64_t levels_off = 8;
    const uint64_t row_start_off = (levels_off + count64 + 3) & ~3ull;
    const uint64_t offsets_off = row_start_off + count64 * 4;
    const uint64_t targets_off = offsets_off + (rows + 1ull) * 4;
    if (secs[fmt::kGraphUpper].len != targets_off + nnz * 4ull) {
        return Corrupted("upper section size");
    }

    Segment out;
    Impl& s = *out.impl_;
    IndexView& v = s.view;
    v.count = count;
    v.dim = dim;
    v.stride = stride;
    v.M = M;
    v.metric = static_cast<Metric>(metric_raw);
    v.entry = entry;
    v.max_level = max_level;
    v.external_ids = reinterpret_cast<const uint64_t*>(base + secs[fmt::kIds].off);
    v.vectors = reinterpret_cast<const float*>(base + secs[fmt::kVectors].off);
    v.level0 = reinterpret_cast<const uint32_t*>(base + secs[fmt::kGraphL0].off);
    v.levels = up + levels_off;
    v.upper_row_start = reinterpret_cast<const uint32_t*>(up + row_start_off);
    v.upper_offsets = reinterpret_cast<const uint32_t*>(up + offsets_off);
    v.upper_targets = reinterpret_cast<const uint32_t*>(up + targets_off);

    // ---- structural validation of the graph (deep mode) ---------------------
    if (deep) {
        uint64_t expect_rows = 0;
        for (uint32_t i = 0; i < count; ++i) {
            if (v.upper_row_start[i] != expect_rows) {
                return Corrupted("upper row_start inconsistent");
            }
            expect_rows += v.levels[i];
        }
        if (expect_rows != rows) return Corrupted("rows != sum(levels)");
        if (v.upper_offsets[rows] != nnz) return Corrupted("offsets[rows] != nnz");
        for (uint32_t r = 0; r < rows; ++r) {
            if (v.upper_offsets[r] > v.upper_offsets[r + 1]) {
                return Corrupted("upper offsets not monotone");
            }
        }
        for (uint32_t t = 0; t < nnz; ++t) {
            if (v.upper_targets[t] >= count) return Corrupted("upper target out of range");
        }
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t* l = v.level0 + i * l0_stride;
            if (l[0] > 2 * M) return Corrupted("level0 degree over cap");
            for (uint32_t j = 1; j <= l[0]; ++j) {
                if (l[j] >= count) return Corrupted("level0 target out of range");
            }
        }
        if (count > 0) {
            uint32_t highest = 0;
            for (uint32_t i = 0; i < count; ++i) {
                highest = std::max<uint32_t>(highest, v.levels[i]);
            }
            if (static_cast<uint32_t>(max_level) != highest) {
                return Corrupted("max_level != max(levels)");
            }
        }
    }

    // ---- metadata ------------------------------------------------------------
    const uint8_t* mp = base + secs[fmt::kMeta].off;
    const uint64_t mlen = secs[fmt::kMeta].len;
    if (mlen < 4) return Corrupted("meta section too small");
    const uint32_t col_count = fmt::GetU32(mp);
    if (col_count > 4096 || 4 + col_count * fmt::kMetaDescSize > mlen) {
        return Corrupted("meta descriptor area out of bounds");
    }
    for (uint32_t c = 0; c < col_count; ++c) {
        const uint8_t* d = mp + 4 + c * fmt::kMetaDescSize;
        const uint32_t name_off = fmt::GetU32(d);
        const uint32_t name_len = fmt::GetU32(d + 4);
        const uint8_t type_raw = d[8];
        const uint64_t data_off = fmt::GetU64(d + 12);
        const uint64_t data_len = fmt::GetU64(d + 20);
        const uint64_t aux_off = fmt::GetU64(d + 28);
        const uint64_t aux_len = fmt::GetU64(d + 36);

        if (type_raw > 2) return Corrupted("meta column type");
        if (name_off + static_cast<uint64_t>(name_len) > mlen ||
            data_off + data_len > mlen || aux_off + aux_len > mlen) {
            return Corrupted("meta column bounds");
        }
        MetaColumnView col;
        col.name = {reinterpret_cast<const char*>(mp + name_off), name_len};
        col.type = static_cast<MetaType>(type_raw);
        col.count = count64;
        switch (col.type) {
            case MetaType::kInt64:
                if (data_len != count64 * 8 || data_off % 8 != 0) {
                    return Corrupted("meta i64 column size/alignment");
                }
                col.i64 = reinterpret_cast<const int64_t*>(mp + data_off);
                break;
            case MetaType::kBool:
                if (data_len != count64) return Corrupted("meta bool column size");
                col.bools = mp + data_off;
                break;
            case MetaType::kString:
                if (data_len != (count64 + 1) * 4 || data_off % 4 != 0) {
                    return Corrupted("meta string offsets size/alignment");
                }
                col.str_offsets = reinterpret_cast<const uint32_t*>(mp + data_off);
                col.str_blob = reinterpret_cast<const char*>(mp + aux_off);
                if (deep) {
                    for (uint64_t r = 0; r < count64; ++r) {
                        if (col.str_offsets[r] > col.str_offsets[r + 1]) {
                            return Corrupted("meta string offsets not monotone");
                        }
                    }
                    if (col.str_offsets[count64] != aux_len) {
                        return Corrupted("meta string blob size mismatch");
                    }
                }
                break;
        }
        s.meta.push_back(col);
    }

    s.map = std::move(map);
    // The view points into the previous `map` object's memory; moving
    // MappedFile transfers the mapping without changing addresses, so the
    // pointers stay valid.
    return out;
}

}  // namespace vdb
