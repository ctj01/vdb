/**
 * @file file.hpp
 * @brief The small platform-encapsulation layer for segment I/O: atomic
 *        write (tmp + fsync + rename) and read-only memory mapping.
 *
 * Everything platform-specific about durability and mapping lives behind
 * these two entry points — nothing above this layer sees an #ifdef.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "vdb/result.hpp"

namespace vdb::io {

/**
 * @brief Writes bytes durably and atomically to `final_path`:
 *        write to `<final_path>.tmp` -> fsync -> rename.
 *
 * POSIX: fsync(file), rename(2), then fsync of the parent directory (the
 * rename itself is metadata that must reach disk).
 * Windows: FlushFileBuffers + MoveFileExW(MOVEFILE_REPLACE_EXISTING |
 * MOVEFILE_WRITE_THROUGH).
 *
 * Readers either see the previous file or the complete new one — never a
 * torn intermediate.
 */
Result<void> AtomicWriteFile(const std::string& final_path, std::string_view bytes);

/// Read-only memory-mapped file. Move-only RAII; unmaps on destruction.
class MappedFile {
public:
    static Result<MappedFile> Open(const std::string& path);

    MappedFile() = default;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const uint8_t* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

private:
    const uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#endif
    void Release() noexcept;
};

}  // namespace vdb::io
