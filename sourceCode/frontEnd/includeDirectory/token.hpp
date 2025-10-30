#pragma once
#include <cstdint>
#include <cstddef>

namespace see {

// ---------------------- Targeted Architectures ----------------------

enum class Arch : std::uint8_t {
  X86_64 = 0,
  ARM64  = 1
};

// ---------------------- Primitive Tokens ----------------------

enum class Prim : std::uint8_t {
  // Void
  Void,
  Nullptr,      // C++: std::nullptr_t

  // Boolean
  Bool,         // C++: bool

  // Character Family
  Char,         // plain 'char' (implementation-defined signedness)
  SChar,        // signed char
  UChar,        // unsigned char
  Char8,        // char8_t (C++20)
  Char16,       // char16_t
  Char32,       // char32_t
  WChar,        // wchar_t (arm64: 4 bytes)

  // Integers
  Short,
  UShort,
  Int,
  UInt,
  Long,
  ULong,
  LongLong,
  ULongLong,
  Int128,       // GNU/Clang extension
  UInt128,      // GNU/Clang extension

  // Floating
  Float,
  Double,
  LongDouble,

  // Count
  _COUNT
};

// ---------------------- Type Metadata ----------------------

struct TypeInfo {
  Prim          kind;
  const char *  spelling;        // canonical name (e.g., "unsigned long long")

  // Classification
  bool          is_integer;
  bool          is_signed;       // for integers; plain 'char' reported as false (indeterminate)
  bool          is_floating;
  bool          is_character;

  // ABI sizes/alignments (bytes) for macOS x86_64 and arm64
  std::uint8_t  size_x86_64;
  std::uint8_t  align_x86_64;
  std::uint8_t  size_arm64;
  std::uint8_t  align_arm64;

  // Ranks for usual arithmetic conversions
  // Integers: ISO rank; Floats: float(1) < double(2) < long double(3)
  std::uint8_t  int_rank;
  std::uint8_t  float_rank;
};

// ---------------------- Public API (implemented in token.cpp) ----------------------

// Lookup immutable metadata row for a primitive; returns nullptr if k is invalid.
const TypeInfo *type_info(Prim k) noexcept;

// Size / Alignment by Target
std::uint8_t size_of(Prim k, Arch a) noexcept;
std::uint8_t align_of(Prim k, Arch a) noexcept;

// Classification Helpers
bool is_integer(Prim k)   noexcept;
bool is_floating(Prim k)  noexcept;
bool is_signed(Prim k)    noexcept;
bool is_character(Prim k) noexcept;

// Conversion Ranks
std::uint8_t integer_rank(Prim k) noexcept;
std::uint8_t float_rank(Prim k)   noexcept;

// Canonical Spelling
const char *to_string(Prim k) noexcept;

// Spelling Lookups
bool from_spelling(const char *s, Prim *out) noexcept;
bool from_flexible_spelling(const char *s, Prim *out) noexcept;

}
