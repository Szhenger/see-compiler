#ifndef SEECPP_UTILITY_RESULT_H_
#define SEECPP_UTILITY_RESULT_H_

#include <stdexcept>
#include <utility>
#include <variant>

namespace seecpp::utility {

/// @brief A C++20 compatible implementation of a success-or-error return type.
/// Mimics C++23's std::expected for codebases without exception handling.
template <typename T, typename E>
class Result {
 public:
  // Constructors implicitly convert from either T or E
  Result(T value) : data_(std::move(value)) {}
  Result(E error) : data_(std::move(error)) {}

  /// @brief Checks if the result contains a success value.
  [[nodiscard]] bool has_value() const noexcept {
    return std::holds_alternative<T>(data_);
  }
  explicit operator bool() const noexcept { return has_value(); }

  /// @brief Accesses the success value. 
  /// @warning Calling this when has_value() is false is undefined behavior.
  T& value() { return std::get<T>(data_); }
  const T& value() const { return std::get<T>(data_); }

  /// @brief Accesses the error value.
  E& error() { return std::get<E>(data_); }
  const E& error() const { return std::get<E>(data_); }

 private:
  std::variant<T, E> data_;
};

}  // namespace seecpp::utility

#endif  // SEECPP_UTILITY_RESULT_H_
