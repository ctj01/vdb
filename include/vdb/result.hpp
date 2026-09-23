/**
 * @file result.hpp
 * @brief Error handling without exceptions: Result<T> / Error.
 *
 * The public surface of this library will eventually sit behind a C ABI for
 * .NET consumption; exceptions must never cross that boundary. Every fallible
 * operation returns Result<T>. C++20 has no std::expected (C++23), and the
 * project takes no external dependencies, so this is a minimal hand-rolled
 * equivalent.
 */
#pragma once

#include <cassert>
#include <string>
#include <utility>
#include <variant>

namespace vdb {

enum class ErrorCode : int {
    kOk = 0,
    kInvalidArgument = 1,
    kOutOfRange = 2,
    kIo = 3,
    kCorrupted = 4,
    kUnsupported = 5,
    kInternal = 6,
};

struct Error {
    ErrorCode code = ErrorCode::kInternal;
    std::string message;
};

inline Error MakeError(ErrorCode code, std::string message) {
    return Error{code, std::move(message)};
}

/**
 * @brief Holds either a T or an Error. [[nodiscard]] so callers cannot
 *        silently drop failures.
 */
template <typename T>
class [[nodiscard]] Result {
public:
    Result(T value) : v_(std::move(value)) {}          // NOLINT(implicit)
    Result(Error error) : v_(std::move(error)) {}      // NOLINT(implicit)

    bool ok() const noexcept { return std::holds_alternative<T>(v_); }
    explicit operator bool() const noexcept { return ok(); }

    T& value() & { assert(ok()); return std::get<T>(v_); }
    const T& value() const& { assert(ok()); return std::get<T>(v_); }
    T&& value() && { assert(ok()); return std::get<T>(std::move(v_)); }

    const Error& error() const& { assert(!ok()); return std::get<Error>(v_); }

private:
    std::variant<T, Error> v_;
};

/// Result<void>: success carries nothing; failure carries the Error.
template <>
class [[nodiscard]] Result<void> {
public:
    Result() = default;
    Result(Error error) : err_(std::move(error)), failed_(true) {}  // NOLINT

    bool ok() const noexcept { return !failed_; }
    explicit operator bool() const noexcept { return ok(); }
    const Error& error() const& { assert(!ok()); return err_; }

private:
    Error err_;
    bool failed_ = false;
};

}  // namespace vdb
