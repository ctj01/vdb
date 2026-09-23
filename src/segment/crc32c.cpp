/**
 * @file crc32c.cpp
 * @brief Table-driven CRC32C. Table built once at startup (magic static).
 */
#include "segment/crc32c.hpp"

#include <array>

namespace vdb {

namespace {

std::array<uint32_t, 256> BuildTable() {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 1u) ? (crc >> 1) ^ 0x82F63B78u : crc >> 1;
        }
        table[i] = crc;
    }
    return table;
}

}  // namespace

uint32_t Crc32c(const void* data, std::size_t len) {
    static const std::array<uint32_t, 256> table = BuildTable();
    const auto* p = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

}  // namespace vdb
