#pragma once
#include <cstddef>
#include <indicators/progress_bar.hpp>
#include <memory>
#include <string>
#include <utility>

namespace util {

enum class Color {
    green,
    red,
    yellow,
    blue,
    cyan,
    magenta,
    white,
};

inline indicators::Color to_indicators_color(Color c) {
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

class ProgressBar {
  public:
    ProgressBar() : text_{}, color_{Color::green}, width_{50}, max_progress_{0}, current_{0} {
        rebuild_bar();
    }

    ProgressBar(std::string text, std::size_t max_progress, Color color = Color::green, std::size_t width = 50)
        : text_{std::move(text)}, color_{color}, width_{width}, max_progress_{max_progress}, current_{0} {
        rebuild_bar();
    }

    ProgressBar(const ProgressBar&) = delete;
    ProgressBar& operator=(const ProgressBar&) = delete;
    ProgressBar(ProgressBar&&) noexcept = default;
    ProgressBar& operator=(ProgressBar&&) noexcept = default;

    void set(std::size_t value) {
        if (!bar_) return;
        if (value > max_progress_) value = max_progress_;
        current_ = value;
        bar_->set_progress(current_);
    }

    void tick(std::size_t delta = 1) {
        set(current_ + delta);
    }

    void done() {
        if (!bar_) return;
        if (current_ < max_progress_) {
            current_ = max_progress_;
            bar_->set_progress(current_);
        }
    }

    void setup(std::size_t new_max_progress, std::string new_text = {}, Color new_color = Color::green,
               std::size_t new_width = 0) {
        if (!new_text.empty()) text_ = std::move(new_text);

        color_ = new_color;
        if (new_width != 0) width_ = new_width;

        max_progress_ = new_max_progress;
        current_ = 0;

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
        using namespace indicators;
        bar_ = std::make_unique<indicators::ProgressBar>(
            option::BarWidth{width_}, option::Start{"["}, option::Fill{"="}, option::Lead{">"}, option::Remainder{" "},
            option::End{"]"}, option::ForegroundColor{to_indicators_color(color_)}, option::ShowElapsedTime{true},
            option::ShowRemainingTime{true}, option::PrefixText{text_}, option::MaxProgress{max_progress_});
    }

    std::string text_;
    Color color_;
    std::size_t width_;

    std::unique_ptr<indicators::ProgressBar> bar_;
    std::size_t max_progress_;
    std::size_t current_;
};

} // namespace util