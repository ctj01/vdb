/**
 * @file writer.cpp
 * @brief Segment serialization + atomic write.
 *
 * The segment is assembled in memory and written in one atomic operation
 * (tmp + fsync + rename). For the current target sizes (hundreds of MB) the
 * transient memory cost is acceptable; a streaming writer with the same
 * layout is a future optimization, not a format change.
 */
#include <bit>
#include <cstring>

#include "io/file.hpp"
#include "segment/crc32c.hpp"
#include "segment/format.hpp"
#include "vdb/segment.hpp"

namespace vdb {

namespace {

struct SectionEntry {
    uint32_t type;
    uint64_t off;
    uint64_t len;
    uint32_t crc;
};

/// Appends raw bytes (host is little-endian; enforced in WriteSegment).
void PutBytes(std::string& out, const void* p, std::size_t n) {
    out.append(static_cast<const char*>(p), n);
}

Result<std::string> BuildMetaSection(const IndexView& view,
                                     const std::vector<MetaColumn>& meta) {
    const uint64_t count = view.count;
    for (const MetaColumn& c : meta) {
        if (c.name.empty() || c.name.size() > 65535) {
            return MakeError(ErrorCode::kInvalidArgument, "bad meta column name");
        }
        const std::size_t n = c.type == MetaType::kInt64  ? c.i64.size()
                              : c.type == MetaType::kBool ? c.bools.size()
                                                          : c.strings.size();
        if (n != count) {
            return MakeError(ErrorCode::kInvalidArgument,
                             "meta column '" + c.name + "' size != count");
        }
    }

    std::string sec;
    fmt::PutU32(sec, static_cast<uint32_t>(meta.size()));
    const std::size_t desc_area = 4 + meta.size() * fmt::kMetaDescSize;

    // Names region right after the descriptors.
    std::string names;
    std::vector<std::pair<uint64_t, uint64_t>> name_pos;  // off, len (section-rel)
    for (const MetaColumn& c : meta) {
        name_pos.push_back({desc_area + names.size(), c.name.size()});
        names += c.name;
    }

    // Column payloads, each 8-aligned relative to the section start.
    std::string payload;  // starts at desc_area + names.size()
    const std::size_t payload_base = desc_area + names.size();
    auto align8 = [&] {
        while ((payload_base + payload.size()) % 8 != 0) payload.push_back('\0');
    };

    struct ColPos {
        uint64_t data_off, data_len, aux_off, aux_len;
    };
    std::vector<ColPos> pos;
    for (const MetaColumn& c : meta) {
        ColPos p{};
        align8();
        p.data_off = payload_base + payload.size();
        switch (c.type) {
            case MetaType::kInt64:
                p.data_len = c.i64.size() * 8;
                PutBytes(payload, c.i64.data(), p.data_len);
                break;
            case MetaType::kBool:
                p.data_len = c.bools.size();
                PutBytes(payload, c.bools.data(), p.data_len);
                break;
            case MetaType::kString: {
                p.data_len = (count + 1) * 4;
                uint32_t off = 0;
                for (const std::string& s : c.strings) {
                    fmt::PutU32(payload, off);
                    off += static_cast<uint32_t>(s.size());
                }
                fmt::PutU32(payload, off);
                p.aux_off = payload_base + payload.size();
                p.aux_len = off;
                for (const std::string& s : c.strings) payload += s;
                break;
            }
        }
        pos.push_back(p);
    }

    // Emit descriptors now that every offset is known.
    for (std::size_t i = 0; i < meta.size(); ++i) {
        fmt::PutU32(sec, static_cast<uint32_t>(name_pos[i].first));
        fmt::PutU32(sec, static_cast<uint32_t>(name_pos[i].second));
        sec.push_back(static_cast<char>(meta[i].type));
        sec.append(3, '\0');
        fmt::PutU64(sec, pos[i].data_off);
        fmt::PutU64(sec, pos[i].data_len);
        fmt::PutU64(sec, pos[i].aux_off);
        fmt::PutU64(sec, pos[i].aux_len);
    }
    sec += names;
    sec += payload;
    return sec;
}

}  // namespace

Result<void> WriteSegment(const std::string& path, const IndexView& view,
                          const std::vector<MetaColumn>& meta) {
    if constexpr (std::endian::native != std::endian::little) {
        return MakeError(ErrorCode::kUnsupported,
                         "big-endian hosts are not supported by the writer");
    }
    if (view.count > 0 && (view.entry == kNoEntry || view.entry >= view.count)) {
        return MakeError(ErrorCode::kInvalidArgument, "view has invalid entry point");
    }

    const uint64_t count = view.count;
    uint64_t rows = 0;
    for (uint64_t i = 0; i < count; ++i) rows += view.levels[i];
    const uint64_t nnz = rows > 0 ? view.upper_offsets[rows] : 0;

    std::string buf(fmt::kHeaderSize, '\0');  // header placeholder
    std::vector<SectionEntry> table;

    auto begin_section = [&](uint32_t type) -> uint64_t {
        fmt::AlignTo64(buf);
        table.push_back({type, buf.size(), 0, 0});
        return buf.size();
    };
    auto end_section = [&](uint64_t start) {
        table.back().len = buf.size() - start;
        table.back().crc = Crc32c(buf.data() + start, buf.size() - start);
    };

    {  // kIds
        const uint64_t s = begin_section(fmt::kIds);
        PutBytes(buf, view.external_ids, count * 8);
        end_section(s);
    }
    {  // kVectors
        const uint64_t s = begin_section(fmt::kVectors);
        PutBytes(buf, view.vectors, count * view.stride * 4);
        end_section(s);
    }
    {  // kGraphL0
        const uint64_t s = begin_section(fmt::kGraphL0);
        PutBytes(buf, view.level0, count * view.Level0Stride() * 4);
        end_section(s);
    }
    {  // kGraphUpper
        const uint64_t s = begin_section(fmt::kGraphUpper);
        fmt::PutU32(buf, static_cast<uint32_t>(rows));
        fmt::PutU32(buf, static_cast<uint32_t>(nnz));
        PutBytes(buf, view.levels, count);
        while ((buf.size() - s) % 4 != 0) buf.push_back('\0');
        PutBytes(buf, view.upper_row_start, count * 4);
        PutBytes(buf, view.upper_offsets, (rows + 1) * 4);
        PutBytes(buf, view.upper_targets, nnz * 4);
        end_section(s);
    }
    {  // kMeta
        auto sec = BuildMetaSection(view, meta);
        if (!sec.ok()) return sec.error();
        const uint64_t s = begin_section(fmt::kMeta);
        buf += sec.value();
        end_section(s);
    }

    fmt::AlignTo64(buf);
    const uint64_t table_off = buf.size();
    for (const SectionEntry& e : table) {
        fmt::PutU32(buf, e.type);
        fmt::PutU64(buf, e.off);
        fmt::PutU64(buf, e.len);
        fmt::PutU32(buf, e.crc);
    }
    buf.append(fmt::kMagic, sizeof(fmt::kMagic));  // footer

    // Header, now that section_table_off is known.
    std::string header;
    header.reserve(fmt::kHeaderSize);
    header.append(fmt::kMagic, sizeof(fmt::kMagic));
    fmt::PutU16(header, fmt::kVersionMajor);
    fmt::PutU16(header, fmt::kVersionMinor);
    fmt::PutU32(header, view.dim);
    header.push_back(static_cast<char>(view.metric));
    header.push_back('\0');  // quant: reserved
    fmt::PutU16(header, 0);
    fmt::PutU32(header, view.M);
    fmt::PutU64(header, count);
    fmt::PutU32(header, view.entry);
    fmt::PutI32(header, view.max_level);
    fmt::PutU64(header, table_off);
    fmt::PutU32(header, static_cast<uint32_t>(table.size()));
    fmt::PutU32(header, 0);  // header_crc placeholder
    header.resize(fmt::kHeaderSize, '\0');
    const uint32_t hcrc = Crc32c(header.data(), fmt::kHeaderSize);
    // Patch the CRC in place (offset 52).
    std::string crc_bytes;
    fmt::PutU32(crc_bytes, hcrc);
    header.replace(52, 4, crc_bytes);
    buf.replace(0, fmt::kHeaderSize, header);

    return io::AtomicWriteFile(path, buf);
}

}  // namespace vdb
