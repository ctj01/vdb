/**
 * @file crc32c.hpp
 * @brief CRC32C (Castagnoli), software table implementation.
 *
 * Own implementation per spec (no external deps). The polynomial is the
 * Castagnoli one (reflected 0x82F63B78) — the same the SSE4.2 `crc32`
 * instruction computes, so a hardware fast path can be added later behind a
 * flag without changing stored checksums.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace vdb {

/// One-shot CRC32C of a buffer (init/finalize handled internally).
uint32_t Crc32c(const void* data, std::size_t len);

}  // namespace vdb
