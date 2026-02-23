#pragma once

/**
 * @file progress_bar.h
 * @brief Small RAII wrapper around the `indicators` progress bar for CLI tools
 *
 * This module provides a minimal, dependency-friendly interface for showing progress
 * in long-running operations (batch processing, benchmarking, dataset iteration, etc.)
 */

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#if defined(__has_include) && __has_include(<indicators/progress_bar.hpp>)
    #include <indicators/progress_bar.hpp>
#elif defined(__has_include) && __has_include(<progress_bar.hpp>)
    #include <progress_bar.hpp>
#else
    #error "Indicators 'progress_bar.hpp' header not found"
#endif

namespace bar {

/**
 * @brief Supported color set for progress bar foreground
 *
 * This enum intentionally mirrors a subset of `indicators::Color` while allowing the
 * project to keep its own stable type.
 */
enum class Color {
    green,
    red,
    yellow,
    blue,
    cyan,
    magenta,
    white,
};

/**
 * @brief Converts project color enum to `indicators::Color`
 *
 * @note The default return value is white as a safe fallback.
 */
inline constexpr indicators::Color to_indicators_color(Color c) noexcept {
    using I = indicators::Color;
    switch (c) {
    case Color::green:
        return I::green;
    case Color::red:
        return I::red;
    case Color::yellow:
        return I::yellow;
    case Color::blue:
        return I::blue;
    case Color::cyan:
        return I::cyan;
    case Color::magenta:
        return I::magenta;
    case Color::white:
        return I::white;
    }
    return I::white;
}

/**
 * @brief Convenience progress bar wrapper for terminal applications
 *
 * Typical lifecycle:
 *  1) Construct or call @ref setup with max progress + optional styling
 *  2) Repeatedly call @ref tick or @ref set as work advances
 *  3) Call @ref done to force completion when work finishes early
 *
 * @note Concurrent updates from multiple threads are not guaranteed to be safe.
 */
class ProgressBar {
  public:
    /**
     * @brief Constructs a default progress bar (no label, green, width=50)
     *
     * The bar is initialized immediately. Default max is 1 to avoid undefined/odd
     * behavior in some indicators implementations when MaxProgress is 0.
     */
    ProgressBar() : text_{}, color_{Color::green}, width_{50}, max_progress_{1}, current_{0} {}

    /**
     * @brief Constructs a progress bar with user-provided label and max progress
     *
     * @param text Prefix text shown before the bar (moved into the object)
     * @param max_progress Total number of steps for completion (clamped to >= 1)
     * @param color Foreground color (defaults to green)
     * @param width Bar width in characters (defaults to 50; clamped to >= 1)
     */
    ProgressBar(std::string text, std::size_t max_progress, Color color = Color::green, std::size_t width = 50)
        : text_{std::move(text)}, color_{color}, width_{(width == 0 ? 1u : width)},
          max_progress_{(max_progress == 0 ? 1u : max_progress)}, current_{0} {
        rebuild_bar();
    }

    ProgressBar(const ProgressBar&) = delete;
    ProgressBar& operator=(const ProgressBar&) = delete;

    ProgressBar(ProgressBar&&) noexcept = default;
    ProgressBar& operator=(ProgressBar&&) noexcept = default;

    /**
     * @brief Sets the current progress to an absolute value
     *
     * Values greater than @ref max are clamped.
     */
    void set(std::size_t value) {
        if (!bar_) return;

        if (value > max_progress_) value = max_progress_;
        current_ = value;
        bar_->set_progress(current_);
    }

    /**
     * @brief Increments progress by @p delta (default: 1)
     *
     * Saturating add (prevents size_t overflow).
     */
    void tick(std::size_t delta = 1) {
        if (!bar_) return;

        if (current_ >= max_progress_) {
            set(max_progress_);
            return;
        }

        const std::size_t remaining = max_progress_ - current_;
        const std::size_t next = (delta >= remaining) ? max_progress_ : (current_ + delta);
        set(next);
    }

    /**
     * @brief Marks the progress bar as complete
     *
     * If current progress is below max, it is set to max and the UI is updated.
     */
    void done() {
        if (!bar_) return;
        if (current_ < max_progress_) {
            current_ = max_progress_;
            bar_->set_progress(current_);
        }
    }

    /**
     * @brief Reconfigures the progress bar and resets progress to zero
     *
     * @param new_max_progress New total progress target (clamped to >= 1)
     * @param new_text New prefix text (always applied; can be empty to clear)
     * @param new_color New bar color
     * @param new_width Optional new bar width; 0 means "keep current" (width is clamped to >= 1)
     */
    void setup(std::size_t new_max_progress, std::string new_text = {}, Color new_color = Color::green,
               std::size_t new_width = 0) {
        text_ = std::move(new_text);
        color_ = new_color;

        if (new_width != 0) width_ = (new_width == 0 ? width_ : new_width);
        if (width_ == 0) width_ = 1;

        max_progress_ = (new_max_progress == 0 ? 1u : new_max_progress);
        current_ = 0;

        rebuild_bar();
    }

    /**
     * @brief Set prefix text without changing other options (can be empty to clear)
     */
    void set_text(std::string text) {
        text_ = std::move(text);
        rebuild_bar();
    }

    /**
     * @brief Clears prefix text
     */
    void clear_text() {
        text_.clear();
        rebuild_bar();
    }

    [[nodiscard]] std::size_t max() const noexcept {
        return max_progress_;
    }
    [[nodiscard]] std::size_t current() const noexcept {
        return current_;
    }

  private:
    void rebuild_bar() {
        // Keep current within bounds after any reconfigure.
        if (current_ > max_progress_) current_ = max_progress_;

        bar_ = std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{width_}, indicators::option::Start{"["}, indicators::option::Fill{"="},
            indicators::option::Lead{">"}, indicators::option::Remainder{" "}, indicators::option::End{"]"},
            indicators::option::ForegroundColor{to_indicators_color(color_)}, indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true}, indicators::option::PrefixText{text_},
            indicators::option::MaxProgress{max_progress_});

        // Ensure UI reflects current progress after rebuild
        bar_->set_progress(current_);
    }

    std::string text_;
    Color color_;
    std::size_t width_;
    std::unique_ptr<indicators::ProgressBar> bar_;
    std::size_t max_progress_;
    std::size_t current_;
};

} // namespace bar
