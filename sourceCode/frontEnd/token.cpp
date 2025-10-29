#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cctype>

namespace see {

// ---------------------- Public Enums ----------------------

enum class Arch : uint8_t { X86_64 = 0, ARM64 = 1 };

enum class Prim : uint8_t {
  // Void
  Void,
  Nullptr,      // C++: std::nullptr_t

  // Boolean
  Bool,         // C++: bool

  // Character Family
  Char,         // plain 'char' (signedness implicitly-defined)
  SChar,        // signed char
  UChar,        // unsigned char
  Char8,        // char8_t (C++20)
  Char16,       // char16_t
  Char32,       // char32_t
  WChar,        // wchar_t (Darwin: 4 bytes)

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

// ---------------------- Type Metadata Table ----------------------

struct TypeInfo {
  Prim        kind;
  const char *spelling;       // canonical spelling used by the compiler

  // Classification Flags
  bool        is_integer;
  bool        is_signed;      // for integers; indeterminate for plain 'char' -> false
  bool        is_floating;
  bool        is_character;

  // ABI sizes/alignments (bytes) for macOS x86_64 and arm64 respectively
  uint8_t     size_x86_64;
  uint8_t     align_x86_64;
  uint8_t     size_arm64;
  uint8_t     align_arm64;

  // Ranks for usual arithmetic conversions
  uint8_t     int_rank;
  uint8_t     float_rank;
};

static constexpr TypeInfo TI(Prim k, const char* name,
                             bool is_int, bool is_signed, bool is_float, bool is_char,
                             uint8_t sx86, uint8_t ax86, uint8_t sarm, uint8_t aarm,
                             uint8_t irank, uint8_t frank) {
  return TypeInfo{k, name, is_int, is_signed, is_float, is_char,
                  sx86, ax86, sarm, aarm, irank, frank};
}

static constexpr TypeInfo kTypeTable[static_cast<size_t>(Prim::_COUNT)] = {
  // Void
  TI(Prim::Void,       "void",               false,false,false,false, 0, 0, 0, 0, 0, 0),
  TI(Prim::Nullptr,    "nullptr_t",          false,false,false,false, 8, 8, 8, 8, 0, 0),

  // Boolean
  TI(Prim::Bool,       "bool",               true, false,false,false, 1, 1, 1, 1, 1, 0),

  // Character Family
  TI(Prim::Char,       "char",               true, false,false,true,  1, 1, 1, 1, 2, 0),
  TI(Prim::SChar,      "signed char",        true, true, false,true,  1, 1, 1, 1, 2, 0),
  TI(Prim::UChar,      "unsigned char",      true, false,false,true,  1, 1, 1, 1, 2, 0),
  TI(Prim::Char8,      "char8_t",            true, false,false,true,  1, 1, 1, 1, 2, 0),
  TI(Prim::Char16,     "char16_t",           true, false,false,true,  2, 2, 2, 2, 3, 0),
  TI(Prim::Char32,     "char32_t",           true, false,false,true,  4, 4, 4, 4, 4, 0),
  TI(Prim::WChar,      "wchar_t",            true, false,false,true,  4, 4, 4, 4, 4, 0),

  // Integers
  TI(Prim::Short,      "short",              true, true, false,false, 2, 2, 2, 2, 3, 0),
  TI(Prim::UShort,     "unsigned short",     true, false,false,false, 2, 2, 2, 2, 3, 0),
  TI(Prim::Int,        "int",                true, true, false,false, 4, 4, 4, 4, 4, 0),
  TI(Prim::UInt,       "unsigned int",       true, false,false,false, 4, 4, 4, 4, 4, 0),
  TI(Prim::Long,       "long",               true, true, false,false, 8, 8, 8, 8, 5, 0),
  TI(Prim::ULong,      "unsigned long",      true, false,false,false, 8, 8, 8, 8, 5, 0),
  TI(Prim::LongLong,   "long long",          true, true, false,false, 8, 8, 8, 8, 6, 0),
  TI(Prim::ULongLong,  "unsigned long long", true, false,false,false, 8, 8, 8, 8, 6, 0),
  TI(Prim::Int128,     "__int128",           true, true, false,false,16,16,16,16, 7, 0),
  TI(Prim::UInt128,    "unsigned __int128",  true, false,false,false,16,16,16,16, 7, 0),

  // Floating
  TI(Prim::Float,      "float",              false,false,true, false, 4, 4, 4, 4, 0, 1),
  TI(Prim::Double,     "double",             false,false,true, false, 8, 8, 8, 8, 0, 2),
  TI(Prim::LongDouble, "long double",        false,false,true, false,16,16, 8, 8, 0, 3), // x86_64: 16B; arm64: 8B
};

// ---------------------- Public API ----------------------

static inline const TypeInfo* type_info(Prim k) {
  const auto idx = static_cast<size_t>(k);
  return (idx < static_cast<size_t>(Prim::_COUNT)) ? &kTypeTable[idx] : nullptr;
}

static inline uint8_t size_of(Prim k, Arch a) {
  const TypeInfo *t = type_info(k);
  if (!t) return 0;
  return (a == Arch::X86_64) ? t->size_x86_64 : t->size_arm64;
}

static inline uint8_t align_of(Prim k, Arch a) {
  const TypeInfo *t = type_info(k);
  if (!t) return 0;
  return (a == Arch::X86_64) ? t->align_x86_64 : t->align_arm64;
}

static inline bool is_integer(Prim k)   { const TypeInfo *t = type_info(k); return t ? t->is_integer  : false; }
static inline bool is_floating(Prim k)  { const TypeInfo *t = type_info(k); return t ? t->is_floating : false; }
static inline bool is_signed(Prim k)    { const TypeInfo *t = type_info(k); return t ? t->is_signed   : false; }
static inline bool is_character(Prim k) { const TypeInfo *t = type_info(k); return t ? t->is_character: false; }

static inline uint8_t integer_rank(Prim k) { const TypeInfo *t = type_info(k); return t ? t->int_rank   : 0; }
static inline uint8_t float_rank(Prim k)   { const TypeInfo *t = type_info(k); return t ? t->float_rank : 0; }

static inline const char* to_string(Prim k) {
  const TypeInfo *t = type_info(k);
  return t ? t->spelling : "<invalid>";
}

// ---------------------- Spelling Lookups ----------------------

static inline bool from_spelling(const char *s, Prim *out) {
  if (!s || !out) return false;
  for (size_t i = 0; i < static_cast<size_t>(Prim::_COUNT); i++) {
    if (std::strcmp(s, kTypeTable[i].spelling) == 0) {
      *out = kTypeTable[i].kind;
      return true;
    }
  }
  return false;
}

static inline void normalize(const char *in, char *out_buf, size_t out_cap) {
  if (!in || !out_buf || out_cap == 0) return;
  size_t o = 0;
  bool last_space = true;

  for (size_t i = 0; in[i] != '\0'; ++i) {
    unsigned char c = static_cast<unsigned char>(in[i]);
    char d = (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : static_cast<char>(c);

    if ((d == '_') || (d >= 'a' && d <= 'z') || (d >= '0' && d <= '9')) {
      if (o + 1 < out_cap) out_buf[o++] = d;
      last_space = false;
    } else {
      if (!last_space) {
        if (o + 1 < out_cap) out_buf[o++] = ' ';
        last_space = true;
      }
    }
  }
  if (o > 0 && out_buf[o - 1] == ' ') --o;
  if (o < out_cap) out_buf[o] = '\0';
}

static inline bool from_flexible_spelling(const char *s, Prim *out) {
  if (!s || !out) return false;

  const size_t n = std::strlen(s);

  char buf[(size_t)(2 * n + 4)];
  normalize(s, buf, sizeof(buf));

  if (from_spelling(buf, out)) return true;

  struct Alias { const char *key; Prim val; };
  static const Alias aliases[] = {
    {"bool",                  Prim::Bool},
    {"nullptr",               Prim::Nullptr},
    {"unsigned",              Prim::UInt},
    {"signed",                Prim::Int},
    {"short int",             Prim::Short},
    {"signed short",          Prim::Short},
    {"signed short int",      Prim::Short},
    {"unsigned short",        Prim::UShort},
    {"unsigned short int",    Prim::UShort},
    {"long int",              Prim::Long},
    {"signed long",           Prim::Long},
    {"signed long int",       Prim::Long},
    {"unsigned long",         Prim::ULong},
    {"unsigned long int",     Prim::ULong},
    {"long long int",         Prim::LongLong},
    {"signed long long",      Prim::LongLong},
    {"signed long long int",  Prim::LongLong},
    {"unsigned long long",    Prim::ULongLong},
    {"unsigned long long int",Prim::ULongLong},
    {"signed char",           Prim::SChar},
    {"unsigned char",         Prim::UChar},
    {"wchar",                 Prim::WChar},
    {"wchar_t",               Prim::WChar},
    {"char8",                 Prim::Char8},
    {"char16",                Prim::Char16},
    {"char32",                Prim::Char32},
    {"__int128_t",            Prim::Int128},
    {"unsigned __int128_t",   Prim::UInt128},
  };

  for (size_t i = 0; i < sizeof(aliases)/sizeof(aliases[0]); i++) {
    if (std::strcmp(buf, aliases[i].key) == 0) { *out = aliases[i].val; return true; }
  }
  return false;
}

// ---------------------- Sanity Checks (Compile-Time) ----------------------

static_assert(static_cast<size_t>(Prim::_COUNT) == 24, "Update kTypeTable when modifying Prim.");
static_assert(sizeof(void*) == 8, "This table assumes LP64 (8-byte pointers).");
static_assert(sizeof(wchar_t) == 4, "Darwin wchar_t expected to be 4 bytes.");

}