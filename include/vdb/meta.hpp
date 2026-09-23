/**
 * @file meta.hpp
 * @brief Typed metadata columns: build-side (owned) and read-side (zero-copy
 *        views into the mapped segment). No metadata indexes yet.
 */
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vdb {

/// Column type tags are part of the on-disk format — never renumber.
enum class MetaType : uint8_t {
    kInt64 = 0,
    kBool = 1,
    kString = 2,  ///< Stored as (count+1) u32 offsets + UTF-8 blob.
};

/// Build-side column: exactly one payload vector is used, per `type`.
/// The writer validates length == segment count.
struct MetaColumn {
    std::string name;
    MetaType type = MetaType::kInt64;
    std::vector<int64_t> i64;
    std::vector<uint8_t> bools;          ///< 0 / 1
    std::vector<std::string> strings;    ///< packed to offsets + blob on write
};

/// Read-side column: pointers into the memory-mapped segment. Valid while
/// the owning Segment is alive.
struct MetaColumnView {
    std::string_view name;
    MetaType type = MetaType::kInt64;
    uint64_t count = 0;
    const int64_t* i64 = nullptr;
    const uint8_t* bools = nullptr;
    const uint32_t* str_offsets = nullptr;  ///< [count + 1]
    const char* str_blob = nullptr;

    std::string_view String(uint64_t row) const {
        return {str_blob + str_offsets[row], str_offsets[row + 1] - str_offsets[row]};
    }
};

}  // namespace vdb
