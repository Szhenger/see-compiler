#include <cctype>
#include <cstring>

#include "token.hpp"

namespace see {

// ----------------------------- Metadata Table -----------------------------

struct RowInit {
  Prim          k;
  const char*   spelling;
  bool          is_integer;
  bool          is_signed;
  bool          is_floating;
  bool          is_character;
  std::uint8_t  sx86, ax86, sarm, aarm;
  std::uint8_t  irank, frank;
};

static constexpr TypeInfo TI(const RowInit& r) {
  return TypeInfo{
    r.k, r.spelling,
    r.is_integer, r.is_signed, r.is_floating, r.is_character,
    r.sx86, r.ax86, r.sarm, r.aarm,
    r.irank, r.frank
  };
}

static constexpr TypeInfo kTypeTable[] = {
  // Void / null-like
  TI({Prim::Void,     "void",                 false,false,false,false, 0,0, 0,0, 0,0}),
  TI({Prim::Nullptr,  "std::nullptr_t",      false,false,false,false, 8,8, 8,8, 0,0}),

  // Booleans
  TI({Prim::Bool,     "bool",                 true,false,false,false, 1,1, 1,1, 1,0}),
  TI({Prim::CBool,    "_Bool",                true,false,false,false, 1,1, 1,1, 1,0}),

  // Character family
  TI({Prim::Char,     "char",                 true,false,false,true,  1,1, 1,1, 2,0}), // signedness impl-defined → report false
  TI({Prim::SChar,    "signed char",          true,true, false,true,  1,1, 1,1, 2,0}),
  TI({Prim::UChar,    "unsigned char",        true,false,false,true,  1,1, 1,1, 2,0}),
  TI({Prim::Char8,    "char8_t",              true,false,false,true,  1,1, 1,1, 2,0}),
  TI({Prim::Char16,   "char16_t",             true,false,false,true,  2,2, 2,2, 3,0}),
  TI({Prim::Char32,   "char32_t",             true,false,false,true,  4,4, 4,4, 4,0}),
  TI({Prim::WChar,    "wchar_t",              true,false,false,true,  4,4, 4,4, 4,0}), // arm64 wchar_t is 4B

  // Integers
  TI({Prim::Short,    "short",                true,true, false,false, 2,2, 2,2, 3,0}),
  TI({Prim::UShort,   "unsigned short",       true,false,false,false, 2,2, 2,2, 3,0}),
  TI({Prim::Int,      "int",                  true,true, false,false, 4,4, 4,4, 4,0}),
  TI({Prim::UInt,     "unsigned int",         true,false,false,false, 4,4, 4,4, 4,0}),
  TI({Prim::Long,     "long",                 true,true, false,false, 8,8, 8,8, 5,0}),
  TI({Prim::ULong,    "unsigned long",        true,false,false,false, 8,8, 8,8, 5,0}),
  TI({Prim::LongLong, "long long",            true,true, false,false, 8,8, 8,8, 6,0}),
  TI({Prim::ULongLong,"unsigned long long",   true,false,false,false, 8,8, 8,8, 6,0}),
  TI({Prim::Int128,   "__int128",             true,true, false,false,16,16,16,16, 7,0}),
  TI({Prim::UInt128,  "unsigned __int128",    true,false,false,false,16,16,16,16, 7,0}),

  // Floating
  TI({Prim::Float,     "float",               false,false,true, false, 4,4, 4,4, 0,1}),
  TI({Prim::Double,    "double",              false,false,true, false, 8,8, 8,8, 0,2}),
  TI({Prim::LongDouble,"long double",         false,false,true, false,16,16, 8,8, 0,3}),
};

static_assert(sizeof(kTypeTable) / sizeof(kTypeTable[0]) == static_cast<std::size_t>(Prim::_COUNT),
              "kTypeTable must cover all Prim values");

// ----------------------------- Public API -----------------------------

const TypeInfo* type_info(Prim k) noexcept {
  auto idx = static_cast<std::size_t>(k);
  if (idx >= static_cast<std::size_t>(Prim::_COUNT)) return nullptr;
  return &kTypeTable[idx];
}

std::uint8_t size_of(Prim k, Arch a) noexcept {
  const TypeInfo* t = type_info(k);
  if (!t) return 0;
  return (a == Arch::X86_64) ? t->size_x86_64 : t->size_arm64;
}

std::uint8_t align_of(Prim k, Arch a) noexcept {
  const TypeInfo* t = type_info(k);
  if (!t) return 0;
  return (a == Arch::X86_64) ? t->align_x86_64 : t->align_arm64;
}

bool is_integer(Prim k)   noexcept { const auto* t = type_info(k); return t ? t->is_integer   : false; }
bool is_floating(Prim k)  noexcept { const auto* t = type_info(k); return t ? t->is_floating  : false; }
bool is_signed(Prim k)    noexcept { const auto* t = type_info(k); return t ? t->is_signed    : false; }
bool is_character(Prim k) noexcept { const auto* t = type_info(k); return t ? t->is_character : false; }

std::uint8_t integer_rank(Prim k) noexcept { const auto* t = type_info(k); return t ? t->int_rank   : 0; }
std::uint8_t float_rank(Prim k)   noexcept { const auto* t = type_info(k); return t ? t->float_rank : 0; }

const char* to_string(Prim k) noexcept {
  const TypeInfo* t = type_info(k);
  return t ? t->spelling : "<invalid>";
}

struct SpEntry { const char* key; Prim k; };

// Canonical, strict table (must match TypeInfo.spelling).
static constexpr SpEntry kStrictMap[] = {
  {"void",                 Prim::Void},
  {"std::nullptr_t",       Prim::Nullptr},

  {"bool",                 Prim::Bool},
  {"_Bool",                Prim::CBool},

  {"char",                 Prim::Char},
  {"signed char",          Prim::SChar},
  {"unsigned char",        Prim::UChar},
  {"char8_t",              Prim::Char8},
  {"char16_t",             Prim::Char16},
  {"char32_t",             Prim::Char32},
  {"wchar_t",              Prim::WChar},

  {"short",                Prim::Short},
  {"unsigned short",       Prim::UShort},
  {"int",                  Prim::Int},
  {"unsigned int",         Prim::UInt},
  {"long",                 Prim::Long},
  {"unsigned long",        Prim::ULong},
  {"long long",            Prim::LongLong},
  {"unsigned long long",   Prim::ULongLong},
  {"__int128",             Prim::Int128},
  {"unsigned __int128",    Prim::UInt128},

  {"float",                Prim::Float},
  {"double",               Prim::Double},
  {"long double",          Prim::LongDouble},
};

bool from_spelling(const char* s, Prim* out) noexcept {
  if (!s || !out) return false;
  for (const auto& e : kStrictMap) {
    const char* a = e.key;
    const char* b = s;
    if (std::strcmp(a, b) == 0) { *out = e.k; return true; }
  }
  return false;
}

static inline char lc(char c) {
  unsigned uc = static_cast<unsigned char>(c);
  if (uc >= 'A' && uc <= 'Z') return static_cast<char>(uc - 'A' + 'a');
  return static_cast<char>(uc);
}

static void normalize_word(const char* s, char* dst, std::size_t dst_n) {
  if (!s || dst_n == 0) { if (dst_n) dst[0] = '\0'; return; }

  std::size_t w = 0;
  bool in_space = true;
  auto push = [&](char c) {
    if (w + 1 < dst_n) dst[w++] = c;
  };

  for (const char* p = s; *p; ++p) {
    char c = lc(*p);
    if (c == '_') c = ' ';
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') {
      if (!in_space) { push(' '); in_space = true; }
      continue;
    }
    push(c);
    in_space = false;
  }

  if (w > 0 && dst[w-1] == ' ') --w;
  dst[w] = '\0';
}

static void strip_trailing_int(char* s) {
  std::size_t n = std::strlen(s);
  if (n < 4) return;

  if (n >= 4 && s[n-1]=='t' && s[n-2]=='n' && s[n-3]=='i' && s[n-4]==' ') {
    s[n-4] = '\0';
  }
}

static constexpr SpEntry kFlexMap[] = {
  {"void",                   Prim::Void},
  {"std::nullptr_t",         Prim::Nullptr},
  {"nullptr",                Prim::Nullptr}, 

  {"bool",                   Prim::Bool},
  {"_bool",                  Prim::CBool},

  {"char",                   Prim::Char},
  {"signed char",            Prim::SChar},
  {"unsigned char",          Prim::UChar},
  {"char8 t",                Prim::Char8},    // normalized "char8_t" → "char8 t"
  {"char16 t",               Prim::Char16},
  {"char32 t",               Prim::Char32},
  {"wchar t",                Prim::WChar},    // "wchar_t" → "wchar t"

  {"short",                  Prim::Short},
  {"unsigned short",         Prim::UShort},
  {"int",                    Prim::Int},
  {"unsigned int",           Prim::UInt},
  {"unsigned",               Prim::UInt},     // alias
  {"signed",                 Prim::Int},      // alias for "signed int"

  {"long",                   Prim::Long},
  {"unsigned long",          Prim::ULong},
  {"long long",              Prim::LongLong},
  {"unsigned long long",     Prim::ULongLong},

  {"__int128",               Prim::Int128},
  {"unsigned __int128",      Prim::UInt128},
  {"unsigned__int128",       Prim::UInt128},  // tolerate missing space after normalization mishaps

  {"float",                  Prim::Float},
  {"double",                 Prim::Double},
  {"long double",            Prim::LongDouble},
};

bool from_flexible_spelling(const char* s, Prim* out) noexcept {
  if (!s || !out) return false;

  char buf[128];
  normalize_word(s, buf, sizeof(buf));

  strip_trailing_int(buf);

  auto starts_with = [](const char* s2, const char* pref)->bool{
    std::size_t lp = std::strlen(pref);
    return std::strncmp(s2, pref, lp) == 0;
  };

  if (starts_with(buf, "signed long long")) {
    std::strcpy(buf, "long long");
  } else if (starts_with(buf, "signed long")) {
    std::strcpy(buf, "long");
  } else if (starts_with(buf, "signed short")) {
    std::strcpy(buf, "short");
  } else if (std::strcmp(buf, "signed") == 0) {
    std::strcpy(buf, "int");
  }

  for (const auto& e : kFlexMap) {
    if (std::strcmp(e.key, buf) == 0) { *out = e.k; return true; }
  }

  if (from_spelling(s, out)) return true;
  
  {
    if (starts_with(buf, "unsigned")) {
      const char* p = buf + std::strlen("unsigned");
      int nlong = 0;
      while (*p) {
        while (*p == ' ') ++p;
        if (std::strncmp(p, "long", 4) == 0) { ++nlong; p += 4; continue; }
        break;
      }
      if (nlong == 1) { *out = Prim::ULong;     return true; }
      if (nlong >= 2) { *out = Prim::ULongLong; return true; }
    }
  }

  return false;
}

}

