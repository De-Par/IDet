#pragma once

#include <iomanip>
#include <ostream>
#include <string>

namespace printer {

struct Ansi {
    bool enable = true;
    const char* reset() const noexcept {
        return enable ? "\033[0m" : "";
    }
    const char* dim() const noexcept {
        return enable ? "\033[2m" : "";
    }
    const char* bold() const noexcept {
        return enable ? "\033[1m" : "";
    }
    const char* cyan() const noexcept {
        return enable ? "\033[36m" : "";
    }
    const char* green() const noexcept {
        return enable ? "\033[32m" : "";
    }
    const char* red() const noexcept {
        return enable ? "\033[31m" : "";
    }
    const char* yellow() const noexcept {
        return enable ? "\033[33m" : "";
    }
};

struct Printer {
    std::ostream& os;
    Ansi a{};
    int key_w = 22;

    void section(const std::string& title, int indent = 0) noexcept {
        indent_spaces(indent);
        os << a.bold() << title << ":" << a.reset() << "\n";
    }

    template <class T>
    void kv(const std::string& key, const T& value, int indent = 0, const char* value_color = nullptr) noexcept {
        auto oldf = os.flags();
        indent_spaces(indent);
        os << std::left << std::setw(key_w) << (key + ":");
        if (value_color) os << value_color;
        os << value;
        if (value_color) os << a.reset();
        os << "\n";
        os.flags(oldf);
    }

    void kv_bool(const std::string& key, bool v, int indent = 0) noexcept {
        auto oldf = os.flags();
        indent_spaces(indent);
        os << std::left << std::setw(key_w) << (key + ":");
        os << (v ? a.green() : a.red()) << (v ? "true" : "false") << a.reset() << "\n";
        os.flags(oldf);
    }

    void kv_path(const std::string& key, const std::string& path, int indent = 0) noexcept {
        auto oldf = os.flags();
        indent_spaces(indent);
        os << std::left << std::setw(key_w) << (key + ":");
        if (path.empty()) {
            os << a.dim() << "(empty)" << a.reset() << "\n";
        } else {
            os << a.cyan() << path << a.reset() << "\n";
        }
        os.flags(oldf);
    }

    void hint(const std::string& msg, int indent = 0) noexcept {
        indent_spaces(indent);
        os << a.dim() << msg << a.reset() << "\n";
    }

  private:
    void indent_spaces(int indent) noexcept {
        for (int i = 0; i < indent; ++i)
            os.put(' ');
    }
};

} // namespace printer
