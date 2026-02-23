/**
 * @file status.h
 * @brief Lightweight status and result types used for explicit error propagation in IDet.
 *
 * This header provides:
 * - @ref idet::Status — a compact error code plus an optional diagnostic message.
 * - @ref idet::Result — a minimal "value-or-error" container (similar to @c std::expected).
 *
 * The IDet public API is designed to avoid exceptions in most code paths (especially hot paths),
 * while still giving callers structured, actionable failure information.
 *
 * @ingroup idet_status
 */

/**
 * @defgroup idet_status Status and Result
 * @brief Error handling primitives used across the IDet public API.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

/**
 * @def IDET_STATUS_TRY
 * @brief Evaluate an expression returning ::idet::Status; on error return it from the caller.
 *
 * Intended for early-return style error propagation.
 *
 * Semantics:
 * - If @p expr returns a non-OK status, the macro returns that status from the current function.
 * - If @p expr returns OK but contains a non-empty message, the message is treated as a non-fatal
 *   diagnostic (warning) and copied into @p stat.message with a @c "warning: " prefix.
 *
 * @param stat A mutable ::idet::Status used to carry a non-fatal diagnostic message (optional).
 * @param expr An expression producing ::idet::Status.
 *
 * @note
 * If multiple calls produce warnings, the last warning overwrites @p stat.message.
 *
 * @code
 * ::idet::Status f() {
 *   ::idet::Status st = ::idet::Status::Ok();
 *   IDET_STATUS_TRY(st, step1());
 *   IDET_STATUS_TRY(st, step2());
 *   return st; // OK, possibly with st.message set to a warning
 * }
 * @endcode
 */

#ifndef IDET_STATUS_TRY
    #define IDET_STATUS_TRY(stat, expr)                                                                                \
        do {                                                                                                           \
            ::idet::Status _idet_st = (expr);                                                                          \
            if (!_idet_st.ok()) return _idet_st;                                                                       \
            if (!_idet_st.message.empty()) stat.message = std::string("warning: ") + _idet_st.message;                 \
        } while (0)
#endif

namespace idet {

/**
 * @ingroup idet_status
 * @brief Represents the outcome of an operation: success or a typed error.
 *
 * A @ref Status consists of:
 * - a compact machine-readable @ref Code, and
 * - an optional human-readable @ref message (UTF-8 recommended).
 *
 * Convention:
 * - Non-OK codes should carry an actionable message.
 * - Some APIs may also return @ref Code::Ok with a non-empty message to carry a non-fatal
 *   diagnostic (warning).
 *
 * @thread_safety
 * Safe to read concurrently when not mutated. Do not mutate the same instance from multiple
 * threads (e.g. due to the underlying @c std::string).
 */
struct Status final {
    /**
     * @brief Enumerates canonical error codes used by IDet.
     *
     * Keep this enum stable to preserve ABI and predictable error semantics.
     * The numeric values are part of the contract: do not reorder or renumber existing entries.
     * Values are stored as @c std::uint8_t to keep the type compact.
     */
    enum class Code : std::uint8_t {
        /** Operation completed successfully. */
        Ok = 0,
        /** Invalid input argument or precondition violation. */
        InvalidArgument = 1,
        /** Requested resource was not found (file, model, key, etc.). */
        NotFound = 2,
        /** Operation or configuration is not supported in the current build/runtime. */
        Unsupported = 3,
        /** Failed to decode or parse input data (e.g., image decode). */
        DecodeError = 4,
        /** Unspecified internal failure (unexpected state, external library error). */
        Internal = 5,
        /** Memory allocation failed or requested memory cannot be obtained. */
        OutOfMemory = 6,
    };

    /** @brief Machine-readable status code. */
    Code code = Code::Ok;

    /**
     * @brief Human-readable diagnostic message (may be empty for @ref Code::Ok).
     *
     * Messages should be actionable when possible (include function name/context).
     * Prefer short, stable phrasing (suitable for logs) over verbose prose.
     */
    std::string message{};

    /**
     * @brief Constructs an OK status.
     * @return Status with @ref Code::Ok and empty message.
     */
    static Status Ok() {
        return {Code::Ok, {}};
    }

    /**
     * @brief Constructs an invalid argument status.
     * @param msg Error details (will be moved).
     * @return Status with @ref Code::InvalidArgument.
     */
    static Status Invalid(std::string msg) {
        return {Code::InvalidArgument, std::move(msg)};
    }

    /**
     * @brief Constructs a not found status.
     * @param msg Error details (will be moved).
     * @return Status with @ref Code::NotFound.
     */
    static Status NotFound(std::string msg) {
        return {Code::NotFound, std::move(msg)};
    }

    /**
     * @brief Constructs an unsupported status.
     * @param msg Error details (will be moved).
     * @return Status with @ref Code::Unsupported.
     */
    static Status Unsupported(std::string msg) {
        return {Code::Unsupported, std::move(msg)};
    }

    /**
     * @brief Constructs a decode error status.
     * @param msg Error details (will be moved).
     * @return Status with @ref Code::DecodeError.
     */
    static Status DecodeError(std::string msg) {
        return {Code::DecodeError, std::move(msg)};
    }

    /**
     * @brief Constructs an internal error status.
     * @param msg Error details (will be moved).
     * @return Status with @ref Code::Internal.
     */
    static Status Internal(std::string msg) {
        return {Code::Internal, std::move(msg)};
    }

    /**
     * @brief Constructs an out-of-memory status.
     * @param msg Error details (will be moved).
     * @return Status with @ref Code::OutOfMemory.
     */
    static Status OutOfMemory(std::string msg) {
        return {Code::OutOfMemory, std::move(msg)};
    }

    /**
     * @brief Checks whether the status represents success.
     * @return True if @ref code is @ref Code::Ok, otherwise false.
     */
    bool ok() const noexcept {
        return code == Code::Ok;
    }
};

/**
 * @ingroup idet_status
 * @brief A minimal container holding either a value of type @c T or an error @ref Status.
 *
 * `Result<T>` is intended for APIs that may fail without relying on exceptions.
 * On success, it holds a value. On failure, it holds a non-OK status.
 *
 * Usage example:
 * @code
 * idet::Result<int> r = parse_something();
 * if (!r.ok()) {
 *   std::cerr << r.status().message << "\n";
 *   return;
 * }
 * int v = std::move(r).value();
 * @endcode
 *
 * @warning
 * The @ref value() accessors do not perform checks. Calling @ref value() when @ref ok()
 * is false dereferences an empty @c std::optional and results in undefined behavior.
 * Always check @ref ok() first.
 *
 * @tparam T Value type stored on success.
 */
template <class T> class Result final {
  public:
    /**
     * @brief Constructs a successful result holding a value.
     * @param v Value to store (will be moved).
     * @return Result containing the value and @ref Status::Ok().
     */
    static Result Ok(T v) {
        Result r;
        r.value_.emplace(std::move(v));
        r.status_ = Status::Ok();
        return r;
    }

    /**
     * @brief Constructs an error result holding a status.
     * @param s A non-OK status to store (will be moved).
     * @return Result containing no value and the provided status.
     *
     * @note
     * @p s must be non-OK. Passing an OK status would make @ref ok() return true while
     * no value is present.
     */
    static Result Err(Status s) {
        Result r;
        r.status_ = std::move(s);
        return r;
    }

    /**
     * @brief Checks whether the result represents success.
     * @return True if the stored status is OK, otherwise false.
     */
    bool ok() const noexcept {
        return status_.ok();
    }

    /**
     * @brief Returns the stored status (OK or error).
     * @return Const reference to the status.
     */
    const Status& status() const noexcept {
        return status_;
    }

    /**
     * @brief Returns a mutable lvalue reference to the stored value.
     * @return Reference to the contained value.
     *
     * @warning
     * Requires `ok() == true`. No checks are performed.
     */
    T& value() & {
        return *value_;
    }

    /**
     * @brief Returns a const lvalue reference to the stored value.
     * @return Const reference to the contained value.
     *
     * @warning
     * Requires `ok() == true`. No checks are performed.
     */
    const T& value() const& {
        return *value_;
    }

    /**
     * @brief Returns an rvalue reference to the stored value for move-out.
     * @return Rvalue reference to the contained value.
     *
     * @warning
     * Requires `ok() == true`. No checks are performed.
     */
    T&& value() && {
        return std::move(*value_);
    }

  private:
    /** @brief Private default constructor; use @ref Ok or @ref Err factories. */
    Result() = default;

    /**
     * @brief Optional storage for the value.
     *
     * Present only on success. Empty on error.
     */
    std::optional<T> value_;

    /**
     * @brief Status for this result.
     *
     * Convention: OK on success, non-OK on error.
     */
    Status status_{};
};

} // namespace idet
