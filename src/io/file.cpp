/**
 * @file file.cpp
 * @brief Platform implementations of atomic write and memory mapping.
 */
#include "io/file.hpp"

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX  // windows.h min/max macros break std::min below
#include <windows.h>

#include <algorithm>

namespace vdb::io {

namespace {

std::wstring Widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(),
                                      static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

}  // namespace

Result<void> AtomicWriteFile(const std::string& final_path, std::string_view bytes) {
    const std::string tmp_path = final_path + ".tmp";
    const std::wstring wtmp = Widen(tmp_path);
    const std::wstring wfinal = Widen(final_path);

    HANDLE h = CreateFileW(wtmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return MakeError(ErrorCode::kIo, "CreateFileW failed: " + tmp_path);
    }
    std::size_t written = 0;
    while (written < bytes.size()) {
        const DWORD chunk = static_cast<DWORD>(
            std::min<std::size_t>(bytes.size() - written, 1u << 30));
        DWORD out = 0;
        if (!WriteFile(h, bytes.data() + written, chunk, &out, nullptr) || out == 0) {
            CloseHandle(h);
            return MakeError(ErrorCode::kIo, "WriteFile failed: " + tmp_path);
        }
        written += out;
    }
    if (!FlushFileBuffers(h)) {
        CloseHandle(h);
        return MakeError(ErrorCode::kIo, "FlushFileBuffers failed: " + tmp_path);
    }
    CloseHandle(h);
    if (!MoveFileExW(wtmp.c_str(), wfinal.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        return MakeError(ErrorCode::kIo, "MoveFileExW failed: " + final_path);
    }
    return {};
}

Result<MappedFile> MappedFile::Open(const std::string& path) {
    const std::wstring wpath = Widen(path);
    // FILE_SHARE_DELETE: match POSIX semantics — a mapped segment can be
    // deleted/renamed underneath us (the mapping stays valid until closed).
    // Without it, compaction could never remove segments still being read.
    HANDLE file = CreateFileW(wpath.c_str(), GENERIC_READ,
                              FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return MakeError(ErrorCode::kIo, "cannot open: " + path);
    }
    LARGE_INTEGER size;
    if (!GetFileSizeEx(file, &size) || size.QuadPart <= 0) {
        CloseHandle(file);
        return MakeError(ErrorCode::kIo, "cannot stat / empty file: " + path);
    }
    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
        CloseHandle(file);
        return MakeError(ErrorCode::kIo, "CreateFileMapping failed: " + path);
    }
    const void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        CloseHandle(mapping);
        CloseHandle(file);
        return MakeError(ErrorCode::kIo, "MapViewOfFile failed: " + path);
    }
    MappedFile mf;
    mf.data_ = static_cast<const uint8_t*>(view);
    mf.size_ = static_cast<std::size_t>(size.QuadPart);
    mf.file_handle_ = file;
    mf.mapping_handle_ = mapping;
    return mf;
}

void MappedFile::Release() noexcept {
    if (data_) UnmapViewOfFile(data_);
    if (mapping_handle_) CloseHandle(mapping_handle_);
    if (file_handle_) CloseHandle(file_handle_);
    data_ = nullptr;
    size_ = 0;
    mapping_handle_ = nullptr;
    file_handle_ = nullptr;
}

}  // namespace vdb::io

#else  // POSIX (Linux, macOS)

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace vdb::io {

namespace {

std::string DirOf(const std::string& path) {
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

Result<void> IoError(const std::string& what) {
    return MakeError(ErrorCode::kIo, what + ": " + std::strerror(errno));
}

}  // namespace

Result<void> AtomicWriteFile(const std::string& final_path, std::string_view bytes) {
    const std::string tmp_path = final_path + ".tmp";
    const int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return IoError("open " + tmp_path);

    std::size_t written = 0;
    while (written < bytes.size()) {
        const ssize_t n = ::write(fd, bytes.data() + written, bytes.size() - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            ::close(fd);
            return IoError("write " + tmp_path);
        }
        written += static_cast<std::size_t>(n);
    }
    if (::fsync(fd) != 0) {
        ::close(fd);
        return IoError("fsync " + tmp_path);
    }
    if (::close(fd) != 0) return IoError("close " + tmp_path);

    if (::rename(tmp_path.c_str(), final_path.c_str()) != 0) {
        return IoError("rename " + final_path);
    }
    // The rename is directory metadata: without fsync-ing the directory a
    // crash can lose the rename even though the file data is durable.
    const int dfd = ::open(DirOf(final_path).c_str(), O_RDONLY | O_DIRECTORY);
    if (dfd >= 0) {
        ::fsync(dfd);
        ::close(dfd);
    }
    return {};
}

Result<MappedFile> MappedFile::Open(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return MakeError(ErrorCode::kIo,
                         "cannot open: " + path + ": " + std::strerror(errno));
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        return MakeError(ErrorCode::kIo, "cannot stat / empty file: " + path);
    }
    void* mem = ::mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ,
                       MAP_PRIVATE, fd, 0);
    ::close(fd);  // the mapping keeps its own reference
    if (mem == MAP_FAILED) {
        return MakeError(ErrorCode::kIo, "mmap failed: " + path);
    }
    MappedFile mf;
    mf.data_ = static_cast<const uint8_t*>(mem);
    mf.size_ = static_cast<std::size_t>(st.st_size);
    return mf;
}

void MappedFile::Release() noexcept {
    if (data_) {
        ::munmap(const_cast<uint8_t*>(data_), size_);
        data_ = nullptr;
        size_ = 0;
    }
}

}  // namespace vdb::io

#endif

namespace vdb::io {

MappedFile::MappedFile(MappedFile&& other) noexcept { *this = std::move(other); }

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        Release();
        data_ = other.data_;
        size_ = other.size_;
#if defined(_WIN32)
        file_handle_ = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        other.file_handle_ = nullptr;
        other.mapping_handle_ = nullptr;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

MappedFile::~MappedFile() { Release(); }

}  // namespace vdb::io
