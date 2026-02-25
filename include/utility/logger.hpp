#pragma once
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

namespace seecpp::utility {

enum class LogLevel { INFO, WARN, ERR };

class Logger {
public:
    static void info(const std::string& msg) { log(LogLevel::INFO, msg); }
    static void warn(const std::string& msg) { log(LogLevel::WARN, msg); }
    static void error(const std::string& msg) { log(LogLevel::ERR, msg); }

private:
    static void log(LogLevel level, const std::string& msg) {
        // Simple timestamp for rigor
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        
        std::string label;
        std::string color = "\033[0m"; // Reset

        switch (level) {
            case LogLevel::INFO:  label = "[INFO]";  color = "\033[32m"; break; // Green
            case LogLevel::WARN:  label = "[WARN]";  color = "\033[33m"; break; // Yellow
            case LogLevel::ERR:   label = "[ERROR]"; color = "\033[31m"; break; // Red
        }

        std::cerr << color << label << " " 
                  << std::put_time(std::localtime(&now), "%H:%M:%S") 
                  << " - " << msg << "\033[0m" << std::endl;
    }
};

} // namespace seecpp::utility