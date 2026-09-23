/**
 * @file aligned.hpp
 * @brief 64-byte-aligned growable float buffer (the vector arena's backing).
 *
 * std::vector cannot guarantee 64-byte alignment portably without a custom
 * allocator; C++17 aligned operator new can, on every target compiler
 * (including MSVC). Growth is geometric doubling — the reference project's
 * "reserve exact per insert" O(n^2) incident is the cautionary tale here.
 */
#pragma once

#include <cstddef>
#include <cstring>
#include <new>

#include "vdb/types.hpp"

namespace vdb {

class AlignedFloatBuffer {
public:
    AlignedFloatBuffer() = default;
    ~AlignedFloatBuffer() { release(); }

    AlignedFloatBuffer(AlignedFloatBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_), cap_(other.cap_) {
        other.data_ = nullptr;
        other.size_ = other.cap_ = 0;
    }
    AlignedFloatBuffer& operator=(AlignedFloatBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            size_ = other.size_;
            cap_ = other.cap_;
            other.data_ = nullptr;
            other.size_ = other.cap_ = 0;
        }
        return *this;
    }
    AlignedFloatBuffer(const AlignedFloatBuffer&) = delete;
    AlignedFloatBuffer& operator=(const AlignedFloatBuffer&) = delete;

    const float* data() const noexcept { return data_; }
    float* data() noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

    /// Appends n floats of space (zero-initialized) and returns the pointer
    /// to the new region. Amortized O(1).
    float* append_zeroed(std::size_t n) {
        if (size_ + n > cap_) {
            grow(size_ + n);
        }
        float* out = data_ + size_;
        std::memset(out, 0, n * sizeof(float));
        size_ += n;
        return out;
    }

private:
    void grow(std::size_t needed) {
        std::size_t new_cap = cap_ == 0 ? 1024 : cap_;
        while (new_cap < needed) new_cap *= 2;
        auto* fresh = static_cast<float*>(
            ::operator new[](new_cap * sizeof(float), std::align_val_t{kAlign}));
        if (size_ > 0) std::memcpy(fresh, data_, size_ * sizeof(float));
        release();
        data_ = fresh;
        cap_ = new_cap;
    }

    void release() noexcept {
        if (data_) {
            ::operator delete[](data_, std::align_val_t{kAlign});
            data_ = nullptr;
        }
    }

    float* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t cap_ = 0;
};

}  // namespace vdb
