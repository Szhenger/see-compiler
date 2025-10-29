#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <utility>
#include <cassert>
#include <cstdio>

#include "token.hpp"
#include "ast.hpp"
#include "lexer.cpp"   

namespace see {

// ------------------------------ Parser ------------------------------

class Parser {
public:
  explicit Parser(Lexer& lex)
  : lex_(lex) {
    advance(); // fill cur_
    advance(); // fill nxt_
  }

  // Parse a full translation unit, producing a finished AST owned by ASTBuilder.
  TranslationUnit parse_translation_unit() {
    while (cur().kind != TokKind::End) {
      if (!parse_toplevel_decl_or_func())
        sync_to_toplevel();
    }
    return builder_.finish();
  }

private:
  // --------- token navigation ----------
  const Token& cur() const { return cur_; }
  const Token& nxt() const { return nxt_; }

  void advance() {
    prev_ = cur_;
    cur_ = nxt_;
    nxt_ = lex_.next();
  }

  bool is(TokKind k) const { return cur().kind == k; }
  bool is_op(std::string_view s) const {
    return cur().kind == TokKind::Operator && cur().lexeme == s;
  }

  bool match(TokKind k) {
    if (cur().kind == k) { advance(); return true; }
    return false;
  }
  bool match_op(std::string_view s) {
    if (is_op(s)) { advance(); return true; }
    return false;
  }

  bool expect(TokKind k, const char* msg) {
    if (match(k)) return true;
    error_here(msg);
    return false;
  }
  bool expect_op(std::string_view s, const char* msg) {
    if (match_op(s)) return true;
    error_here(msg);
    return false;
  }

  [[noreturn]] void error_here(const char* msg) {
    std::fprintf(stderr, "Parse error at line %u col %u: %s (token '%.*s')\n",
                 cur().line, cur().column, msg,
                 int(cur().lexeme.size()), cur().lexeme.data());
    // For now, abort hard. You can improve with error recovery.
    std::abort();
  }

  void sync_to_toplevel() {
    // Crude recovery: skip to ';' or '}' or newline-like points
    while (cur().kind != TokKind::End &&
           !(cur().kind == TokKind::Operator && (cur().lexeme == ";" || cur().lexeme == "}"))) {
      advance();
    }
    if (cur().kind != TokKind::End) advance();
  }

  // --------- helpers: source range ----------
  static SourceRange range_from(const Token& t0, const Token& t1) {
    SourceRange r;
    r.begin = t0.lexeme.data();
    r.end   = t1.lexeme.data() + t1.lexeme.size();
    r.start = {t0.line, t0.column};
    return r;
  }
  static SourceRange range_token(const Token& t) {
    return range_from(t, t);
  }

  // ========================= TOP LEVEL =========================

  // toplevel-declaration:
  //   declaration
  //   function-definition
  //   typedef-declaration
  bool parse_toplevel_decl_or_func() {
    // Preprocessor lines are ignored by the parser (already tokenized as single tokens).
    if (match(TokKind::Preprocessor)) return true;

    // Storage and qualifiers are collected; the core type comes from Type tokens (see::Prim) and
    // the declarator shapes build pointers/arrays/fns.
    QualType base = parse_decl_specifiers(); // type and qualifiers/storage

    // Could be nothing (e.g., only 'typedef' with no type) — guard:
    if (base.type == kInvalidId && base.storage == S_None && base.qual == Q_None) {
      error_here("expected declaration specifiers");
    }

    // (Possibly multiple declarators separated by commas.)
    // But if it's a function definition, handle the first declarator specially.
    DeclOrFunc first = parse_init_declarator_or_func(base);
    if (first.is_function) {
      // If has a body, it's a definition; else, declaration (already pushed)
      return true;
    }

    // Handle comma-separated more declarators for the same base specs
    while (match_op(",")) {
      DeclOrFunc more = parse_init_declarator_or_func(base);
      (void)more; // already emitted into ASTBuilder
    }

    expect_op(";", "expected ';' after declaration");
    return true;
  }

  // --------- declaration specifiers (storage + qualifiers + primitive/record/enum) ---------

  struct QualType {
    TypeId type{kInvalidId};         // base type (without pointers/arrays/fn)
    std::uint32_t qual{Q_None};      // qualifiers
    std::uint32_t storage{S_None};   // storage class flags
    SourceRange where{};
  };

  QualType parse_decl_specifiers() {
    QualType qt{};
    const Token startTok = cur();

    bool saw_type = false;

    while (true) {
      // Qualifiers / storage (keywords tokenized as Keyword)
      if (is(TokKind::Keyword)) {
        std::string_view k = cur().lexeme;
        if (k == "const")      { qt.qual    |= Q_Const;      advance(); continue; }
        if (k == "volatile")   { qt.qual    |= Q_Volatile;   advance(); continue; }
        if (k == "restrict")   { qt.qual    |= Q_Restrict;   advance(); continue; }
        if (k == "_Atomic")    { qt.qual    |= Q_Atomic;     advance(); continue; }
        if (k == "extern")     { qt.storage |= S_Extern;     advance(); continue; }
        if (k == "static")     { qt.storage |= S_Static;     advance(); continue; }
        if (k == "register")   { qt.storage |= S_Register;   advance(); continue; }
        if (k == "_Thread_local" || k == "thread_local") { qt.storage |= S_ThreadLocal; advance(); continue; }
        if (k == "inline")     { qt.storage |= S_Inline;     advance(); continue; }
        if (k == "typedef")    { // keep note; actual typedef handled at end by seeing 'typedef' + declarators
          // We won't set a flag here; instead we'll detect typedef by storage later if needed.
          advance();
          // Piggyback on storage bit to mark typedef (not in S_*; we’ll signal via a local flag)
          typedef_pending_ = true;
          continue;
        }

        // struct/union/enum tags
        if (k == "struct" || k == "union") {
          RecordTag tag = (k == "struct") ? RecordTag::Struct : RecordTag::Union;
          advance();
          qt.type = parse_record_type(tag);
          saw_type = true;
          continue;
        }
        if (k == "enum") {
          advance();
          qt.type = parse_enum_type();
          saw_type = true;
          continue;
        }

        // Otherwise a non-type keyword: stop (let declarator parse hit it as error)
        break;
      }

      // Primitive types arrive as TokKind::Type with a Prim payload.
      if (cur().kind == TokKind::Type) {
        saw_type = true;
        Prim p = cur().prim;
        qt.type = builder_.type_primitive(p, range_token(cur()));
        advance();
        continue;
      }

      // No more specifiers
      break;
    }

    if (!saw_type && qt.qual == Q_None && qt.storage == S_None) {
      // No type yet — leave qt.type invalid; higher-level code will diagnose as needed.
    }

    qt.where = range_from(startTok, prev_);
    return qt;
  }

  // Parse a struct/union definition or reference:  struct ID { fields }  |  struct ID  |  struct { fields }
  TypeId parse_record_type(RecordTag tag) {
    std::string_view name;
    if (cur().kind == TokKind::Identifier) { name = cur().lexeme; advance(); }

    if (match_op("{")) {
      // definition
      std::vector<Field> fields;
      while (!match_op("}")) {
        // field: <specs> <declarator> [ ':' <const-expr> ] ';'
        QualType ft = parse_decl_specifiers();
        Field f{};
        f.name = parse_declarator_name_only(); // consume name and ptr/arr modifiers into type
        TypeId fieldTy = apply_declarator_to_type(ft, /*allow_func=*/false, &f);
        (void)fieldTy;
        if (match_op(":")) {
          // bit-field width: parse as constant expression (for now, scan one integer literal)
          ExprId w = parse_constant_expr();
          f.bit_width = w;
        }
        expect_op(";", "expected ';' after struct/union field");
        fields.push_back(f);
      }
      return builder_.type_record(tag, name, fields, {});
    } else {
      // forward-declared / named tag reference
      // We'll represent as a Named type to be resolved later,
      // or as an empty Record with no fields.
      if (!name.empty()) {
        // Create a record with no fields (forward decl)
        std::vector<Field> none;
        return builder_.type_record(tag, name, none, {});
      } else {
        error_here("expected identifier or '{' after struct/union");
      }
    }
  }

  // enum [ID] { enumerators } | enum ID
  TypeId parse_enum_type() {
    std::string_view name;
    if (cur().kind == TokKind::Identifier) { name = cur().lexeme; advance(); }

    if (match_op("{")) {
      std::vector<Enumerator> list;
      while (!match_op("}")) {
        if (cur().kind != TokKind::Identifier) error_here("expected enumerator name");
        Enumerator e{};
        e.name = cur().lexeme; advance();
        if (match_op("=")) {
          // parse constant expression
          e.value = parse_constant_expr();
        }
        list.push_back(e);
        if (!match_op(",")) {
          expect_op("}", "expected '}' to close enum");
          break;
        }
      }
      return builder_.type_enum(name, list, {});
    } else {
      // forward-declared reference
      if (!name.empty()) {
        std::vector<Enumerator> none;
        return builder_.type_enum(name, none, {});
      } else {
        error_here("expected identifier or '{' after enum");
      }
    }
  }

  // --------- declarators, initializers, and function bodies ---------

  struct DeclOrFunc {
    bool is_function{false};
  };

  DeclOrFunc parse_init_declarator_or_func(const QualType& base) {
    // Declarator:
    //   * ...  name  (...)  [arrays]   (we apply ptr/arr/func layers on top of base.type)
    std::string_view name = parse_declarator_name_only();
    bool allow_func = true;
    TypeId finalT = apply_declarator_to_type(base, allow_func, /*field_out*/nullptr);

    // If the result type is a function AND a '{' follows, parse a definition.
    if (is_function_type(finalT) && cur().kind == TokKind::Operator && cur().lexeme == "{") {
      // function definition
      StmtId body = parse_compound_stmt();
      DeclId fn = builder_.decl_func(name, finalT, base.storage, body, {});
      builder_.push_toplevel(fn);
      return {true};
    }

    // Otherwise, maybe an initializer (for variables) or just a declaration
    ExprId initExpr = kInvalidId;
    if (match_op("=")) {
      initExpr = parse_assignment_expr();
    }

    DeclId var = builder_.decl_var(name, finalT, base.storage, /*is_def*/ match_op(";") ? false : true, initExpr, {});
    // If we consumed ';' as part of is_def detection above, we're done; else caller will expect ';' or ','.
    if (prev_.kind == TokKind::Operator && prev_.lexeme == ";") {
      // already pushed var; mark as full statement handled
      builder_.push_toplevel(var);
      return {false};
    } else {
      builder_.push_toplevel(var);
      return {false};
    }
  }

  // Parse only the *name* portion (and track we saw some declarator). We’ll apply ptr/arr/func around base separately.
  std::string_view parse_declarator_name_only() {
    // For simplicity, we only support direct declarators like:  name  |  (*name)  |  name[]  |  name(...)
    // Complex nested declarators can be added later.
    if (cur().kind != TokKind::Identifier) error_here("expected identifier for declarator");
    std::string_view name = cur().lexeme;
    advance();
    return name;
  }

  // Apply pointers, arrays, and function parameter lists to the base type.
  // For this initial version, we parse trailing `()` and `[]` after reading the name,
  // plus a prefix of '*' immediately before the name is *not* handled here to keep things simple.
  // (You can extend with true full declarator parsing later.)
  TypeId apply_declarator_to_type(const QualType& base, bool allow_func, Field* field_out) {
    TypeId ty = base.type;
    // Trailing suffixes: function params and array dimensions
    while (true) {
      if (allow_func && match_op("(")) {
        std::vector<Param> params;
        if (!match_op(")")) {
          do {
            // Parse parameter: specifiers + declarator (name optional)
            QualType pbase = parse_decl_specifiers();
            std::string_view pname;
            if (cur().kind == TokKind::Identifier) { pname = cur().lexeme; advance(); }
            Param param{};
            param.name = pname;
            // For now we ignore per-parameter ptr/arr suffixes and just assign base
            param.type = pbase.type;
            params.push_back(param);
          } while (match_op(","));
          expect_op(")", "expected ')' to close parameter list");
        }
        ty = builder_.type_function(ty, params, F_None, {});
        continue;
      }
      if (match_op("[")) {
        // Optional constant expression length
        ExprId len = kInvalidId;
        if (!is_op("]")) {
          len = parse_constant_expr();
        }
        expect_op("]", "expected ']' after array size");
        ty = builder_.type_array(ty, len, {});
        continue;
      }
      break;
    }

    // Apply qualifiers if any
    if (base.qual != Q_None) {
      ty = builder_.type_qualified(ty, base.qual, {});
    }

    // field_out unused for now (bitfield already handled earlier)
    (void)field_out;
    return ty;
  }

  bool is_function_type(TypeId t) const {
    const Type& TT = builder_.type(t);
    return TT.kind == TypeKind::Function;
  }

  // ========================= STATEMENTS =========================

  StmtId parse_statement() {
    // Compound
    if (match_op("{")) {
      // put back the '{' into prev_ state and call compound routine that expects it
      // (We already consumed it; adjust by creating a block with trailing '}' expectation)
      // But easier: we have a specialized function that assumes '{' is current token:
      // so use helper that received we already consumed '{' — write a small variant:
      return parse_compound_after_lbrace();
    }

    // if
    if (cur().kind == TokKind::Keyword && cur().lexeme == "if") {
      advance();
      expect_op("(", "expected '(' after if");
      ExprId cond = parse_expression();
      expect_op(")", "expected ')' after condition");
      StmtId thenS = parse_statement();
      StmtId elseS = kInvalidId;
      if (cur().kind == TokKind::Keyword && cur().lexeme == "else") { advance(); elseS = parse_statement(); }
      Stmt s; s.kind = StmtKind::If; s.as.iff = {cond, thenS, elseS};
      return builder_.push_stmt(std::move(s)); // use private push? expose helper instead:
    }

    // while
    if (cur().kind == TokKind::Keyword && cur().lexeme == "while") {
      advance();
      expect_op("(", "expected '(' after while");
      ExprId cond = parse_expression();
      expect_op(")", "expected ')' after condition");
      StmtId body = parse_statement();
      Stmt s; s.kind = StmtKind::While; s.as.whil = {cond, body};
      return builder_.push_stmt(std::move(s));
    }

    // return
    if (cur().kind == TokKind::Keyword && cur().lexeme == "return") {
      advance();
      ExprId e = kInvalidId;
      if (!match_op(";")) {
        e = parse_expression();
        expect_op(";", "expected ';' after return expression");
      }
      return builder_.stmt_return(e, {});
    }

    // expression statement
    ExprId e = parse_expression();
    expect_op(";", "expected ';' after expression");
    return builder_.stmt_expr(e, {});
  }

  StmtId parse_compound_stmt() {
    expect_op("{", "expected '{' to start compound statement");
    return parse_compound_after_lbrace();
  }

  StmtId parse_compound_after_lbrace() {
    std::vector<StmtId> stmts;
    while (!match_op("}")) {
      // simple heuristic: if starts like a type (Type token), parse a local declaration as a DeclStmt?
      // For now, parse only statements and ignore local declarations (extend later).
      StmtId s = parse_statement();
      stmts.push_back(s);
    }
    return builder_.stmt_compound(stmts, {});
  }

  // ========================= EXPRESSIONS =========================

  ExprId parse_expression() { return parse_comma(); }

  ExprId parse_comma() {
    ExprId lhs = parse_conditional();
    while (match_op(",")) {
      ExprId rhs = parse_conditional();
      lhs = builder_.expr_binary(ExprKind::Comma, lhs, rhs, {});
    }
    return lhs;
  }

  ExprId parse_conditional() {
    ExprId cond = parse_assignment_expr();
    if (match_op("?")) {
      ExprId thenE = parse_expression();
      expect_op(":", "expected ':' in conditional expression");
      ExprId elseE = parse_assignment_expr();
      return builder_.expr_conditional(cond, thenE, elseE, {});
    }
    return cond;
  }

  ExprId parse_assignment_expr() {
    ExprId lhs = parse_logical_or();
    // assignment or compound assignment
    if (is(TokKind::Operator)) {
      std::string_view op = cur().lexeme;
      if (op == "=" || op == "+=" || op == "-=" || op == "*=" || op == "/=" || op == "%=" ||
          op == "<<=" || op == ">>=" || op == "&=" || op == "^=" || op == "|=") {
        advance();
        ExprId rhs = parse_assignment_expr();
        ExprKind k = ExprKind::Assign;
        if (op == "+=") k = ExprKind::AddAssign; else if (op == "-=") k = ExprKind::SubAssign;
        else if (op == "*=") k = ExprKind::MulAssign; else if (op == "/=") k = ExprKind::DivAssign;
        else if (op == "%=") k = ExprKind::ModAssign; else if (op == "<<=") k = ExprKind::ShlAssign;
        else if (op == ">>=") k = ExprKind::ShrAssign; else if (op == "&=") k = ExprKind::AndAssign;
        else if (op == "^=") k = ExprKind::XorAssign; else if (op == "|=") k = ExprKind::OrAssign;
        return builder_.expr_assign(k, lhs, rhs, {});
      }
    }
    return lhs;
  }

  // precedence: || > && > | > ^ > & > ==/!= > </>/<=/>= > << >> > + - > * / %
  ExprId parse_logical_or() {
    ExprId lhs = parse_logical_and();
    while (match_op("||")) {
      ExprId rhs = parse_logical_and();
      lhs = builder_.expr_binary(ExprKind::LogOr, lhs, rhs, {});
    }
    return lhs;
  }
  ExprId parse_logical_and() {
    ExprId lhs = parse_bit_or();
    while (match_op("&&")) {
      ExprId rhs = parse_bit_or();
      lhs = builder_.expr_binary(ExprKind::LogAnd, lhs, rhs, {});
    }
    return lhs;
  }
  ExprId parse_bit_or() {
    ExprId lhs = parse_bit_xor();
    while (match_op("|")) {
      ExprId rhs = parse_bit_xor();
      lhs = builder_.expr_binary(ExprKind::BitOr, lhs, rhs, {});
    }
    return lhs;
  }
  ExprId parse_bit_xor() {
    ExprId lhs = parse_bit_and();
    while (match_op("^")) {
      ExprId rhs = parse_bit_and();
      lhs = builder_.expr_binary(ExprKind::BitXor, lhs, rhs, {});
    }
    return lhs;
  }
  ExprId parse_bit_and() {
    ExprId lhs = parse_equality();
    while (match_op("&")) {
      ExprId rhs = parse_equality();
      lhs = builder_.expr_binary(ExprKind::BitAnd, lhs, rhs, {});
    }
    return lhs;
  }
  ExprId parse_equality() {
    ExprId lhs = parse_relational();
    while (is(TokKind::Operator) && (cur().lexeme == "==" || cur().lexeme == "!=")) {
      std::string_view op = cur().lexeme; advance();
      ExprId rhs = parse_relational();
      lhs = builder_.expr_binary(op == "==" ? ExprKind::Eq : ExprKind::Ne, lhs, rhs, {});
    }
    return lhs;
  }
  ExprId parse_relational() {
    ExprId lhs = parse_shifts();
    while (is(TokKind::Operator) &&
           (cur().lexeme == "<" || cur().lexeme == ">" || cur().lexeme == "<=" || cur().lexeme == ">=")) {
      std::string_view op = cur().lexeme; advance();
      ExprId rhs = parse_shifts();
      ExprKind k = ExprKind::Lt;
      if (op == "<") k = ExprKind::Lt; else if (op == "<=") k = ExprKind::Le;
      else if (op == ">") k = ExprKind::Gt; else if (op == ">=") k = ExprKind::Ge;
      lhs = builder_.expr_binary(k, lhs, rhs, {});
    }
    return lhs;
  }
  ExprId parse_shifts() {
    ExprId lhs = parse_additive();
    while (is(TokKind::Operator) && (cur().lexeme == "<<" || cur().lexeme == ">>")) {
      std::string_view op = cur().lexeme; advance();
      ExprId rhs = parse_additive();
      lhs = builder_.expr_binary(op == "<<" ? ExprKind::Shl : ExprKind::Shr, lhs, rhs, {});
    }
    return lhs;
  }
  ExprId parse_additive() {
    ExprId lhs = parse_multiplicative();
    while (is(TokKind::Operator) && (cur().lexeme == "+" || cur().lexeme == "-")) {
      std::string_view op = cur().lexeme; advance();
      ExprId rhs = parse_multiplicative();
      lhs = builder_.expr_binary(op == "+" ? ExprKind::Add : ExprKind::Sub, lhs, rhs, {});
    }
    return lhs;
  }
  ExprId parse_multiplicative() {
    ExprId lhs = parse_unary();
    while (is(TokKind::Operator) && (cur().lexeme == "*" || cur().lexeme == "/" || cur().lexeme == "%")) {
      std::string_view op = cur().lexeme; advance();
      ExprId rhs = parse_unary();
      ExprKind k = ExprKind::Mul;
      if (op == "*") k = ExprKind::Mul; else if (op == "/") k = ExprKind::Div; else if (op == "%") k = ExprKind::Mod;
      lhs = builder_.expr_binary(k, lhs, rhs, {});
    }
    return lhs;
  }

  ExprId parse_unary() {
    // ++ -- & * + - ! ~ sizeof alignof
    if (is(TokKind::Operator)) {
      std::string_view op = cur().lexeme;
      if (op == "++" || op == "--" || op == "&" || op == "*" ||
          op == "+" || op == "-" || op == "!" || op == "~") {
        advance();
        ExprId sub = parse_unary();
        ExprKind k = ExprKind::Plus;
        if (op == "+") k = ExprKind::Plus; else if (op == "-") k = ExprKind::Minus;
        else if (op == "&") k = ExprKind::AddressOf; else if (op == "*") k = ExprKind::Deref;
        else if (op == "!") k = ExprKind::LogNot; else if (op == "~") k = ExprKind::BitNot;
        else if (op == "++") k = ExprKind::PreInc; else if (op == "--") k = ExprKind::PreDec;
        return builder_.expr_unary(k, sub, {});
      }
    }
    if (cur().kind == TokKind::Keyword && (cur().lexeme == "sizeof" || cur().lexeme == "alignof")) {
      bool is_sizeof = (cur().lexeme == "sizeof");
      advance();
      if (match_op("(")) {
        // Try to parse type name very loosely: if next is a Type token, treat as sizeof(type)
        if (cur().kind == TokKind::Type) {
          Prim p = cur().prim; advance();
          TypeId t = builder_.type_primitive(p, {});
          expect_op(")", "expected ')' after type");
          return is_sizeof ? builder_.expr_sizeof_type(t, {}) : builder_.expr_sizeof_type(t, {});
        } else {
          ExprId e = parse_expression();
          expect_op(")", "expected ')'");
          return is_sizeof ? builder_.expr_sizeof_expr(e, {}) : builder_.expr_sizeof_expr(e, {});
        }
      } else {
        ExprId e = parse_unary();
        return is_sizeof ? builder_.expr_sizeof_expr(e, {}) : builder_.expr_sizeof_expr(e, {});
      }
    }
    return parse_postfix();
  }

  ExprId parse_postfix() {
    ExprId e = parse_primary();
    while (true) {
      if (match_op("(")) {
        std::vector<Arg> args;
        if (!match_op(")")) {
          do {
            Arg a{}; a.expr = parse_assignment_expr();
            args.push_back(a);
          } while (match_op(","));
          expect_op(")", "expected ')'");
        }
        e = builder_.expr_call(e, args, {});
        continue;
      }
      if (match_op("[")) {
        ExprId idx = parse_expression();
        expect_op("]", "expected ']'");
        e = builder_.expr_index(e, idx, {});
        continue;
      }
      if (match_op(".")) {
        if (cur().kind != TokKind::Identifier) error_here("expected member name after '.'");
        std::string_view name = cur().lexeme; advance();
        e = builder_.expr_member(/*ptr=*/false, e, name, {});
        continue;
      }
      if (match_op("->")) {
        if (cur().kind != TokKind::Identifier) error_here("expected member name after '->'");
        std::string_view name = cur().lexeme; advance();
        e = builder_.expr_member(/*ptr=*/true, e, name, {});
        continue;
      }
      if (match_op("++")) {
        // PostInc
        // For simplicity we’ll store as binary with a dummy; you can add dedicated PostInc kind
        // but ExprKind already has PostInc/PostDec — extend builder if needed.
        // Here, map to PreInc for MVP (adjust later).
        e = builder_.expr_unary(ExprKind::PreInc, e, {});
        continue;
      }
      if (match_op("--")) {
        e = builder_.expr_unary(ExprKind::PreDec, e, {});
        continue;
      }
      break;
    }
    return e;
  }

  ExprId parse_primary() {
    if (cur().kind == TokKind::Identifier) {
      std::string_view name = cur().lexeme; advance();
      return builder_.expr_identifier(name, {});
    }
    if (cur().kind == TokKind::Integer) {
      std::string_view txt = cur().lexeme; advance();
      return builder_.expr_integer(txt, {});
    }
    if (cur().kind == TokKind::Floating) {
      std::string_view txt = cur().lexeme; advance();
      return builder_.expr_floating(txt, {});
    }
    if (cur().kind == TokKind::StringLit) {
      std::string_view txt = cur().lexeme; advance();
      return builder_.expr_string(txt, {});
    }
    if (cur().kind == TokKind::CharLit) {
      std::string_view txt = cur().lexeme; advance();
      return builder_.expr_char(txt, {});
    }
    if (match_op("(")) {
      ExprId e = parse_expression();
      expect_op(")", "expected ')'");
      return builder_.expr_paren(e, {});
    }
    error_here("expected primary expression");
    std::abort();
  }

  // ---------- constant expr: reuse assignment_expr for MVP ----------
  ExprId parse_constant_expr() { return parse_assignment_expr(); }

private:
  Lexer&      lex_;
  Token       prev_{};
  Token       cur_{};
  Token       nxt_{};
  ASTBuilder  builder_;
  bool        typedef_pending_{false};
};

// ------------------------------ entry helper ------------------------------
// Convenience function: given a source buffer, lex + parse into a TU.

TranslationUnit parse_buffer(const char* src, size_t len) {
  Lexer lex(src, len);
  Parser p(lex);
  return p.parse_translation_unit();
}

} // namespace see
