#pragma once

#include <iostream>
#include <string_view>
#include <chrono>
#include <iomanip>
#include <source_location>
#include <mutex>

namespace seecpp::utility {

enum class LogLevel : uint8_t { Debug = 0, Info = 1, Warn = 2, Error = 3 };

class Logger {
public:
    // --- Configuration (call once at startup) ---
    static void setLevel(LogLevel min_level) {
        min_level_ = min_level;
    }
    static LogLevel level() { return min_level_; }

    // --- Public API ---
    static void debug(std::string_view msg,
        const std::source_location loc = std::source_location::current()) {
        log(LogLevel::Debug, msg, loc);
    }
    static void info(std::string_view msg,
        const std::source_location loc = std::source_location::current()) {
        log(LogLevel::Info, msg, loc);
    }
    static void warn(std::string_view msg,
        const std::source_location loc = std::source_location::current()) {
        log(LogLevel::Warn, msg, loc);
    }
    static void error(std::string_view msg,
        const std::source_location loc = std::source_location::current()) {
        log(LogLevel::Error, msg, loc);
    }

private:
    static void log(LogLevel level, std::string_view msg,
                    const std::source_location& loc)
    {
        if (level < min_level_) return;

        // Timestamp — thread-safe via localtime_r / localtime_s
        auto now     = std::chrono::system_clock::now();
        auto time_t_ = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf{};
#if defined(_WIN32)
        localtime_s(&tm_buf, &time_t_);
#else
        localtime_r(&time_t_, &tm_buf);
#endif

        const char* color;
        const char* label;
        std::ostream* out;

        switch (level) {
            case LogLevel::Debug:
                color = "\033[36m"; label = "[DEBUG]"; out = &std::cout; break;
            case LogLevel::Info:
                color = "\033[32m"; label = "[INFO] "; out = &std::cout; break;
            case LogLevel::Warn:
                color = "\033[33m"; label = "[WARN] "; out = &std::cerr; break;
            case LogLevel::Error:
                color = "\033[31m"; label = "[ERROR]"; out = &std::cerr; break;
            default:
                color = "\033[0m";  label = "[?????]"; out = &std::cerr; break;
        }

        // Mutex guard: log() may be called from parallel passes in the future.
        static std::mutex mtx;
        std::lock_guard<std::mutex> lock(mtx);

        *out << color
             << label << ' '
             << std::put_time(&tm_buf, "%H:%M:%S") << ' '
             << loc.file_name() << ':' << loc.line()
             << " — " << msg
             << "\033[0m" << '\n';  // '\n' not endl: avoid per-line flush

        // Only hard-flush on errors where immediacy matters.
        if (level == LogLevel::Error) out->flush();
    }

    static inline LogLevel min_level_ = LogLevel::Info;
};

} // namespace seecpp::utility