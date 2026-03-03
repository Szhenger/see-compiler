#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <memory>
#include <map>
#include <variant>
#include <optional>
#include <span>
#include <cassert>
#include <cstdint>
#include <functional>

// =============================================================================
// SeeC++ Static Intermediate Representation (SIR)
//
// Design philosophy (MLIR-inspired):
//   - SSA form: every Value is defined exactly once
//   - Use-def chains enable DCE, CSE, and Autodiff traversal
//   - Operations are dialect-namespaced (sc_high.* -> sc_low.* lowering)
//   - Blocks are the unit of scheduling; Regions compose Blocks
//   - Attributes are compile-time constants; never mutable post-construction
// =============================================================================

namespace seecpp::sir {

// Forward declarations
class Operation;
class Block;
class Region;

// =============================================================================
// 1. Type System
// =============================================================================

enum class DataType : uint8_t {
    F16,   // Half-precision float   (inference / memory bandwidth)
    BF16,  // Brain float            (training; wider exponent than F16)
    F32,   // Single-precision float (default training dtype)
    F64,   // Double-precision float (scientific / testing)
    I8,    // 8-bit signed int       (quantized inference)
    I32,   // 32-bit signed int      (indices, counts)
    I64,   // 64-bit signed int      (large-scale indexing)
    Bool,  // 1-bit boolean          (masks, predicates)
};

/// Returns the size in bytes of a scalar element of the given DataType.
constexpr size_t dtypeByteWidth(DataType dt) {
    switch (dt) {
        case DataType::Bool:
        case DataType::I8:   return 1;
        case DataType::F16:
        case DataType::BF16: return 2;
        case DataType::F32:
        case DataType::I32:  return 4;
        case DataType::F64:
        case DataType::I64:  return 8;
    }
    return 0; // unreachable
}

/// Returns a human-readable name for a DataType (useful for IR printing).
constexpr std::string_view dtypeName(DataType dt) {
    switch (dt) {
        case DataType::F16:  return "f16";
        case DataType::BF16: return "bf16";
        case DataType::F32:  return "f32";
        case DataType::F64:  return "f64";
        case DataType::I8:   return "i8";
        case DataType::I32:  return "i32";
        case DataType::I64:  return "i64";
        case DataType::Bool: return "bool";
    }
    return "unknown";
}

/// Tensor shape descriptor.
/// A dimension value of -1 (kDynamic) indicates a runtime-determined size,
/// following the convention used by MLIR's RankedTensorType.
struct Shape {
    static constexpr int64_t kDynamic = -1;

    std::vector<int64_t> dims;

    Shape() = default;
    explicit Shape(std::vector<int64_t> d) : dims(std::move(d)) {}

    /// Scalar: rank-0 tensor.
    static Shape scalar() { return Shape{}; }

    int64_t rank() const { return static_cast<int64_t>(dims.size()); }
    bool isScalar() const { return dims.empty(); }

    /// True iff all dimensions are statically known.
    bool isFullyStatic() const {
        for (auto d : dims)
            if (d == kDynamic) return false;
        return true;
    }

    /// Number of static elements. Returns kDynamic if any dim is dynamic.
    int64_t volume() const {
        int64_t v = 1;
        for (auto d : dims) {
            if (d == kDynamic) return kDynamic;
            v *= d;
        }
        return v;
    }

    /// Memory footprint in bytes for a given dtype. Returns 0 if dynamic.
    size_t byteSize(DataType dt) const {
        auto vol = volume();
        if (vol == kDynamic) return 0;
        return static_cast<size_t>(vol) * dtypeByteWidth(dt);
    }

    bool operator==(const Shape& o) const { return dims == o.dims; }
    bool operator!=(const Shape& o) const { return !(*this == o); }
};

// =============================================================================
// 2. Attributes
// =============================================================================

/// Compile-time constant attached to an Operation.
/// Variants mirror the attribute kinds used in ONNX / MLIR:
///   - Integer scalars and float scalars
///   - Strings (e.g. padding mode: "SAME", "VALID")
///   - Integer lists (strides, dilations, pads, kernel sizes)
///   - Float lists  (scale factors for quantisation)
using AttributeValue = std::variant<
    int64_t,
    float,
    double,
    std::string,
    std::vector<int64_t>,
    std::vector<float>
>;

// =============================================================================
// 3. Value — SSA node with full use-def bookkeeping
// =============================================================================

/// A typed, shaped SSA value in the SIR graph.
///
/// Ownership model:
///   - Results are *owned* by their defining Operation (via unique_ptr).
///   - Block arguments are *owned* by their Block.
///   - All other references (operands, users) are non-owning raw pointers.
///
/// Thread safety: none — IR construction is single-threaded by convention.
class Value {
public:
    Value(std::string id, DataType dt, Shape sh, Operation* def_op)
        : id_(std::move(id)), dtype_(dt), shape_(std::move(sh)),
          defining_op_(def_op) {}

    // Non-copyable; pointer identity matters.
    Value(const Value&)            = delete;
    Value& operator=(const Value&) = delete;
    Value(Value&&)                 = delete;
    Value& operator=(Value&&)      = delete;

    // --- Accessors ---
    std::string_view id()       const { return id_; }
    DataType         dtype()    const { return dtype_; }
    const Shape&     shape()    const { return shape_; }
    Operation*       definingOp()     { return defining_op_; }
    const Operation* definingOp() const { return defining_op_; }

    /// True for block arguments (inputs with no defining op in this block).
    bool isBlockArgument() const { return defining_op_ == nullptr; }

    // --- Use-Def ---
    std::span<Operation* const> users() const { return users_; }
    bool hasOneUse()  const { return users_.size() == 1; }
    bool hasNoUses()  const { return users_.empty(); }

    /// Replace all uses of this value with `newVal` in the user list.
    /// This is the fundamental primitive for algebraic rewriting passes.
    void replaceAllUsesWith(Value* newVal);

    /// Internal: called by Operation::addOperand only.
    void addUser(Operation* op) { users_.push_back(op); }
    /// Internal: called by Operation::removeOperand only.
    void removeUser(Operation* op);

    // --- Shape mutation (used by ShapeInferencePass only) ---
    void setShape(Shape sh) { shape_ = std::move(sh); }

private:
    std::string  id_;
    DataType     dtype_;
    Shape        shape_;
    Operation*   defining_op_;           // null => block argument
    std::vector<Operation*> users_;      // non-owning; maintained by addOperand
};

// =============================================================================
// 4. Operation — the fundamental unit of computation
// =============================================================================

/// An Operation is the fundamental node in the SIR computation graph.
///
/// An operation has:
///   - A mnemonic  : dialect-namespaced opcode, e.g. "sc_high.conv2d"
///   - Operands    : ordered list of non-owning Value* inputs
///   - Results     : ordered list of owned Value outputs (unique_ptr)
///   - Attributes  : compile-time metadata (strides, padding, etc.)
///   - A parent    : the Block that contains this operation (non-owning)
///
/// Dialect convention:
///   sc_high.*  — High-level ops (Conv2D, Gemm, BatchNorm). Pre-lowering.
///   sc_low.*   — Low-level ops (Im2Col, MatMul). Post-lowering.
///   sc_mem.*   — Memory ops (Alloc, DeallocOp, Copy). Code generation.
///   sc_ctrl.*  — Control flow (If, For, While).
class Operation {
public:
    explicit Operation(std::string mnemonic, Block* parent = nullptr)
        : mnemonic_(std::move(mnemonic)), parent_block_(parent) {}

    // Non-copyable; pointer identity matters.
    Operation(const Operation&)            = delete;
    Operation& operator=(const Operation&) = delete;

    // --- Identity ---
    std::string_view mnemonic()    const { return mnemonic_; }
    Block*           parentBlock()       { return parent_block_; }
    const Block*     parentBlock() const { return parent_block_; }
    void             setParentBlock(Block* b) { parent_block_ = b; }

    // --- Dialect predicates ---
    bool isHighLevel()   const { return mnemonic_.rfind("sc_high.", 0) == 0; }
    bool isLowLevel()    const { return mnemonic_.rfind("sc_low.",  0) == 0; }
    bool isMemoryOp()    const { return mnemonic_.rfind("sc_mem.",  0) == 0; }
    bool isControlFlow() const { return mnemonic_.rfind("sc_ctrl.", 0) == 0; }

    // --- Operands ---
    std::span<Value* const>  operands() const { return operands_; }
    Value*                   operand(size_t i) const {
        assert(i < operands_.size() && "operand index out of range");
        return operands_[i];
    }
    size_t numOperands() const { return operands_.size(); }

    void addOperand(Value* v);
    void setOperand(size_t i, Value* newVal);

    // --- Results ---
    std::span<const std::unique_ptr<Value>> results() const { return results_; }
    Value* result(size_t i = 0) const {
        assert(i < results_.size() && "result index out of range");
        return results_[i].get();
    }
    size_t numResults() const { return results_.size(); }

    /// Create and register a new result Value. If `id` is empty, an SSA name
    /// is auto-generated (%0, %1, …). Returns the raw non-owning pointer.
    Value* addResult(std::string id, DataType dt, Shape sh);

    // --- Attributes ---
    void setAttribute(std::string key, AttributeValue val) {
        attributes_[std::move(key)] = std::move(val);
    }
    const AttributeValue* getAttribute(std::string_view key) const {
        auto it = attributes_.find(std::string(key));
        return (it != attributes_.end()) ? &it->second : nullptr;
    }
    bool hasAttribute(std::string_view key) const {
        return getAttribute(key) != nullptr;
    }

    /// Typed attribute accessor. Returns nullopt if absent or wrong type.
    template <typename T>
    std::optional<T> getAttrAs(std::string_view key) const {
        if (auto* av = getAttribute(key))
            if (auto* v = std::get_if<T>(av))
                return *v;
        return std::nullopt;
    }

    // --- IR printing (debugging) ---
    void print(std::ostream& os) const;
    std::string toString() const;

private:
    std::string                                        mnemonic_;
    std::vector<Value*>                                operands_;   // non-owning
    std::vector<std::unique_ptr<Value>>                results_;    // owning
    std::map<std::string, AttributeValue>              attributes_;
    Block*                                             parent_block_ = nullptr;

    // Monotonic SSA name counter — shared across all Operations.
    static std::atomic<size_t> id_counter_;
};

// =============================================================================
// 5. Block — an ordered, linear sequence of Operations
// =============================================================================

/// A Block is a straight-line sequence of Operations ending in a terminator.
/// Blocks own their operations and block-argument Values.
///
/// In a future Region-based extension, a Block also carries predecessor /
/// successor edges for CFG-based analyses (dominator trees, liveness).
class Block {
public:
    Block() = default;

    // Non-copyable.
    Block(const Block&)            = delete;
    Block& operator=(const Block&) = delete;

    // --- Block arguments (analogous to MLIR's bbargs / phi nodes) ---
    Value* addArgument(DataType dt, Shape sh);
    std::span<const std::unique_ptr<Value>> arguments() const { return args_; }

    // --- Operation management ---
    std::span<const std::unique_ptr<Operation>> operations() const {
        return ops_;
    }
    size_t numOps() const { return ops_.size(); }

    /// Append a new operation and take ownership. Returns a non-owning pointer.
    Operation* appendOp(std::string name);

    /// Insert an existing (heap-allocated) Operation at the back.
    Operation* appendOp(std::unique_ptr<Operation> op);

    /// Remove and return an operation by pointer (for rewriting passes).
    std::unique_ptr<Operation> removeOp(Operation* op);

    // --- Validation ---
    /// Basic structural invariant check (SSA uniqueness, operand defs reachable).
    bool validate() const;
    bool isValidated() const { return is_validated_; }

    // --- Iteration helpers for passes ---
    /// Walk all operations in forward order.
    void walk(std::function<void(Operation*)> fn);
    /// Walk all operations in reverse order (for Autodiff / liveness).
    void walkReverse(std::function<void(Operation*)> fn);

    // --- IR printing ---
    void print(std::ostream& os) const;

private:
    std::vector<std::unique_ptr<Value>>     args_;   // block-argument Values
    std::vector<std::unique_ptr<Operation>> ops_;
    mutable bool                            is_validated_ = false;
};

// =============================================================================
// 6. Region — a collection of Blocks (for control-flow ops)
// =============================================================================

/// A Region is a list of Blocks. High-level ops with sub-computation
/// (e.g., sc_ctrl.if, sc_ctrl.for) embed a Region rather than a flat Block.
class Region {
public:
    Block* addBlock();
    Block* entryBlock() {
        assert(!blocks_.empty() && "region has no blocks");
        return blocks_.front().get();
    }
    std::span<const std::unique_ptr<Block>> blocks() const { return blocks_; }

private:
    std::vector<std::unique_ptr<Block>> blocks_;
};

// =============================================================================
// 7. Op Builder — dialect factory helpers
// =============================================================================

/// Centralised factory for well-typed Operation construction.
/// Each method enforces the expected operand / attribute / result structure
/// so that callers cannot accidentally build a malformed Op.
///
/// All returned Operations are *unowned* — the caller must append them to a
/// Block (which takes ownership) or wrap them in a unique_ptr.
struct OpBuilder {

    // ---- sc_high dialect -----------------------------------------------

    /// 2-D convolution (ONNX Conv).
    ///   operands : [input, filter]              (bias is optional)
    ///   strides  : [sh, sw]
    ///   pads     : [top, left, bottom, right]   (default: 0)
    ///   dilations: [dh, dw]                     (default: 1)
    ///   group    : depthwise multiplier          (default: 1)
    static std::unique_ptr<Operation> conv2d(
        Value*                   input,
        Value*                   filter,
        Value*                   bias,          // may be nullptr
        std::vector<int64_t>     strides,
        std::vector<int64_t>     pads      = {0,0,0,0},
        std::vector<int64_t>     dilations = {1,1},
        int64_t                  group     = 1
    );

    /// Batch Normalisation (ONNX BatchNormalization).
    ///   operands: [input, scale, bias, running_mean, running_var]
    static std::unique_ptr<Operation> batchNorm(
        Value* input, Value* scale, Value* bias,
        Value* running_mean, Value* running_var,
        float epsilon = 1e-5f
    );

    /// General Matrix Multiply (ONNX Gemm / MatMul).
    ///   operands: [A, B] — bias is optional
    static std::unique_ptr<Operation> gemm(
        Value* A, Value* B, Value* bias = nullptr,
        bool trans_a = false, bool trans_b = false
    );

    /// Element-wise ReLU.
    static std::unique_ptr<Operation> relu(Value* input);

    // ---- sc_low dialect ------------------------------------------------

    /// Im2Col — unfolds a 4-D input into a 2-D column matrix.
    static std::unique_ptr<Operation> im2col(
        Value*               input,
        std::vector<int64_t> kernel_shape,
        std::vector<int64_t> strides,
        std::vector<int64_t> pads
    );
};

} // namespace seecpp::sir