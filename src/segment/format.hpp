/**
 * @file format.hpp
 * @brief .vdb segment format constants + little-endian serialization helpers.
 *
 * Layout (all sections 64-byte aligned; little-endian throughout):
 *
 *   [ header: 128 bytes ]
 *   [ sections ... each starting at a 64-byte boundary ]
 *   [ section table: section_count * 24 bytes ]
 *   [ footer: magic repeated, 8 bytes ]
 *
 * Header (offsets in bytes):
 *    0  magic[8]            "VDBSEG\0\1"
 *    8  u16 version_major   (reader rejects if > kVersionMajor)
 *   10  u16 version_minor
 *   12  u32 dim
 *   16  u8  metric
 *   17  u8  quant           (reserved for int8 quantization; 0 for now)
 *   18  u16 pad
 *   20  u32 M
 *   24  u64 count
 *   32  u32 entry_point     (kNoEntry when empty)
 *   36  i32 max_level       (-1 when empty)
 *   40  u64 section_table_off
 *   48  u32 section_count
 *   52  u32 header_crc      (crc32c of the 128 header bytes with this field 0)
 *   56.. reserved, zeroed to 128
 *
 * Section table entry (24 bytes): u32 type, u64 off, u64 len, u32 crc32c.
 *
 * Section payloads:
 *   kIds       u64 external_id[count]
 *   kVectors   f32 [count][stride], stride = PaddedStride(dim)
 *   kGraphL0   u32 [count][1 + 2M]
 *   kGraphUpper u32 rows, u32 nnz, u8 levels[count], pad to 4,
 *              u32 row_start[count], u32 offsets[rows+1], u32 targets[nnz]
 *   kMeta      u32 col_count, then col_count descriptors of 44 bytes
 *              {u32 name_off, u32 name_len, u8 type, u8 pad[3],
 *               u64 data_off, u64 data_len, u64 aux_off, u64 aux_len},
 *              then payload (names, column data; offsets relative to the
 *              section start, data 8-byte aligned). For kString columns,
 *              data = (count+1) u32 offsets, aux = UTF-8 blob.
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

namespace vdb::fmt {

inline constexpr char kMagic[8] = {'V', 'D', 'B', 'S', 'E', 'G', '\0', '\1'};
inline constexpr uint16_t kVersionMajor = 1;
inline constexpr uint16_t kVersionMinor = 0;
inline constexpr std::size_t kHeaderSize = 128;
inline constexpr std::size_t kFooterSize = 8;
inline constexpr std::size_t kSectionEntrySize = 24;
inline constexpr std::size_t kMetaDescSize = 44;
inline constexpr std::size_t kAlign = 64;

enum SectionType : uint32_t {
    kIds = 1,
    kVectors = 2,
    kGraphL0 = 3,
    kGraphUpper = 4,
    kMeta = 5,
};
inline constexpr uint32_t kSectionCount = 5;

// ---- little-endian primitives (bytewise: correct on any host) --------------

inline void PutU16(std::string& out, uint16_t v) {
    out.push_back(static_cast<char>(v & 0xFF));
    out.push_back(static_cast<char>((v >> 8) & 0xFF));
}
inline void PutU32(std::string& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}
inline void PutU64(std::string& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}
inline void PutI32(std::string& out, int32_t v) {
    PutU32(out, static_cast<uint32_t>(v));
}

inline uint16_t GetU16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
inline uint32_t GetU32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
inline uint64_t GetU64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
inline int32_t GetI32(const uint8_t* p) { return static_cast<int32_t>(GetU32(p)); }

/// Pads `out` with zeros up to the next 64-byte boundary.
inline void AlignTo64(std::string& out) {
    while (out.size() % kAlign != 0) out.push_back('\0');
}

}  // namespace vdb::fmt
