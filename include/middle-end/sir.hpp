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
#include <atomic>
#include <algorithm>

namespace seecpp::sir {

class Operation;
class Block;
class Region;

enum class DataType : uint8_t {
    F16, BF16, F32, F64,
    I8, I32, I64,
    Bool
};

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
        default: return 0;
    }
}

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
        default: return "unknown";
    }
}

struct Shape {
    static constexpr int64_t kDynamic = -1;

    std::vector<int64_t> dims;

    Shape() = default;
    explicit Shape(std::vector<int64_t> d) : dims(std::move(d)) {}
    Shape(std::initializer_list<int64_t> d) : dims(d) {}

    static Shape scalar() { return Shape{}; }

    int64_t rank() const { return static_cast<int64_t>(dims.size()); }
    bool isScalar() const { return dims.empty(); }

    bool isFullyStatic() const {
        return std::none_of(dims.begin(), dims.end(), [](int64_t d) { return d == kDynamic; });
    }

    int64_t volume() const {
        int64_t v = 1;
        for (auto d : dims) {
            if (d == kDynamic) return kDynamic;
            v *= d;
        }
        return v;
    }

    size_t byteSize(DataType dt) const {
        auto vol = volume();
        return (vol == kDynamic) ? 0 : static_cast<size_t>(vol) * dtypeByteWidth(dt);
    }

    bool operator==(const Shape& o) const = default;
};

using AttributeValue = std::variant<
    int64_t,
    float,
    double,
    std::string,
    std::vector<int64_t>,
    std::vector<float>
>;

class Value {
public:
    Value(std::string id, DataType dt, Shape sh, Operation* def_op)
        : id_(std::move(id)), dtype_(dt), shape_(std::move(sh)),
          defining_op_(def_op) {}

    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;

    std::string_view id() const { return id_; }
    DataType dtype() const { return dtype_; }
    const Shape& shape() const { return shape_; }
    Operation* definingOp() { return defining_op_; }
    const Operation* definingOp() const { return defining_op_; }

    bool isBlockArgument() const { return defining_op_ == nullptr; }

    std::span<Operation* const> users() const { return users_; }
    bool hasOneUse() const { return users_.size() == 1; }
    bool hasNoUses() const { return users_.empty(); }

    void replaceAllUsesWith(Value* newVal);

    void addUser(Operation* op) { users_.push_back(op); }
    void removeUser(Operation* op) {
        auto it = std::remove(users_.begin(), users_.end(), op);
        users_.erase(it, users_.end());
    }

    void setShape(Shape sh) { shape_ = std::move(sh); }

private:
    std::string id_;
    DataType dtype_;
    Shape shape_;
    Operation* defining_op_;
    std::vector<Operation*> users_;
};

class Operation {
public:
    explicit Operation(std::string mnemonic, Block* parent = nullptr)
        : mnemonic_(std::move(mnemonic)), parent_block_(parent) {}

    Operation(const Operation&) = delete;

    std::string_view mnemonic() const { return mnemonic_; }
    Block* parentBlock() { return parent_block_; }
    const Block* parentBlock() const { return parent_block_; }
    void setParentBlock(Block* b) { parent_block_ = b; }

    bool isHighLevel() const { return mnemonic_.starts_with("sc_high."); }
    bool isLowLevel() const { return mnemonic_.starts_with("sc_low."); }
    bool isMemoryOp() const { return mnemonic_.starts_with("sc_mem."); }
    bool isControlFlow() const { return mnemonic_.starts_with("sc_ctrl."); }

    std::span<Value* const> operands() const { return operands_; }
    Value* operand(size_t i) const { return operands_.at(i); }
    size_t numOperands() const { return operands_.size(); }

    void addOperand(Value* v) {
        operands_.push_back(v);
        if (v) v->addUser(this);
    }

    void setOperand(size_t i, Value* newVal) {
        if (operands_[i]) operands_[i]->removeUser(this);
        operands_[i] = newVal;
        if (newVal) newVal->addUser(this);
    }

    std::span<const std::unique_ptr<Value>> results() const { return results_; }
    Value* result(size_t i = 0) const { return results_.at(i).get(); }
    size_t numResults() const { return results_.size(); }

    Value* addResult(std::string id, DataType dt, Shape sh) {
        if (id.empty()) {
            id = "%" + std::to_string(id_counter_.fetch_add(1, std::memory_order_relaxed));
        }
        results_.push_back(std::make_unique<Value>(std::move(id), dt, std::move(sh), this));
        return results_.back().get();
    }

    void setAttribute(std::string key, AttributeValue val) {
        attributes_[std::move(key)] = std::move(val);
    }

    const AttributeValue* getAttribute(std::string_view key) const {
        if (auto it = attributes_.find(std::string(key)); it != attributes_.end())
            return &it->second;
        return nullptr;
    }

    bool hasAttribute(std::string_view key) const { return getAttribute(key) != nullptr; }

    template <typename T>
    std::optional<T> getAttrAs(std::string_view key) const {
        if (auto* av = getAttribute(key))
            if (auto* v = std::get_if<T>(av))
                return *v;
        return std::nullopt;
    }

private:
    std::string mnemonic_;
    std::vector<Value*> operands_;
    std::vector<std::unique_ptr<Value>> results_;
    std::map<std::string, AttributeValue, std::less<>> attributes_;
    Block* parent_block_ = nullptr;

    inline static std::atomic<size_t> id_counter_{0};
};

class Block {
public:
    Block() = default;
    Block(const Block&) = delete;

    Value* addArgument(DataType dt, Shape sh) {
        args_.push_back(std::make_unique<Value>("arg" + std::to_string(args_.size()), dt, std::move(sh), nullptr));
        return args_.back().get();
    }

    std::span<const std::unique_ptr<Value>> arguments() const { return args_; }
    std::span<const std::unique_ptr<Operation>> operations() const { return ops_; }
    size_t numOps() const { return ops_.size(); }

    Operation* appendOp(std::string name) {
        return appendOp(std::make_unique<Operation>(std::move(name), this));
    }

    Operation* appendOp(std::unique_ptr<Operation> op) {
        op->setParentBlock(this);
        ops_.push_back(std::move(op));
        return ops_.back().get();
    }

    std::unique_ptr<Operation> removeOp(Operation* op) {
        auto it = std::find_if(ops_.begin(), ops_.end(), [&](const auto& p) { return p.get() == op; });
        if (it == ops_.end()) return nullptr;
        auto ptr = std::move(*it);
        ops_.erase(it);
        return ptr;
    }

    void walk(std::function<void(Operation*)> fn) {
        for (auto& op : ops_) fn(op.get());
    }

    void walkReverse(std::function<void(Operation*)> fn) {
        for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) fn(it->get());
    }

private:
    std::vector<std::unique_ptr<Value>> args_;
    std::vector<std::unique_ptr<Operation>> ops_;
};

class Region {
public:
    Block* addBlock() {
        blocks_.push_back(std::make_unique<Block>());
        return blocks_.back().get();
    }

    Block* entryBlock() {
        assert(!blocks_.empty());
        return blocks_.front().get();
    }

    std::span<const std::unique_ptr<Block>> blocks() const { return blocks_; }

private:
    std::vector<std::unique_ptr<Block>> blocks_;
};

struct OpBuilder {
    static std::unique_ptr<Operation> conv2d(
        Value* input, Value* filter, Value* bias,
        std::vector<int64_t> strides,
        std::vector<int64_t> pads = {0, 0, 0, 0},
        std::vector<int64_t> dilations = {1, 1},
        int64_t group = 1
    ) {
        auto op = std::make_unique<Operation>("sc_high.conv2d");
        op->addOperand(input);
        op->addOperand(filter);
        if (bias) op->addOperand(bias);
        op->setAttribute("strides", std::move(strides));
        op->setAttribute("pads", std::move(pads));
        op->setAttribute("dilations", std::move(dilations));
        op->setAttribute("group", group);
        return op;
    }

    static std::unique_ptr<Operation> batchNorm(
        Value* input, Value* scale, Value* bias,
        Value* running_mean, Value* running_var,
        float epsilon = 1e-5f
    ) {
        auto op = std::make_unique<Operation>("sc_high.batch_norm");
        op->addOperand(input);
        op->addOperand(scale);
        op->addOperand(bias);
        op->addOperand(running_mean);
        op->addOperand(running_var);
        op->setAttribute("epsilon", epsilon);
        return op;
    }

    static std::unique_ptr<Operation> gemm(
        Value* A, Value* B, Value* bias = nullptr,
        bool trans_a = false, bool trans_b = false
    ) {
        auto op = std::make_unique<Operation>("sc_high.gemm");
        op->addOperand(A);
        op->addOperand(B);
        if (bias) op->addOperand(bias);
        op->setAttribute("transA", static_cast<int64_t>(trans_a));
        op->setAttribute("transB", static_cast<int64_t>(trans_b));
        return op;
    }

    static std::unique_ptr<Operation> relu(Value* input) {
        auto op = std::make_unique<Operation>("sc_high.relu");
        op->addOperand(input);
        return op;
    }

    static std::unique_ptr<Operation> im2col(
        Value* input,
        std::vector<int64_t> kernel_shape,
        std::vector<int64_t> strides,
        std::vector<int64_t> pads
    ) {
        auto op = std::make_unique<Operation>("sc_low.im2col");
        op->addOperand(input);
        op->setAttribute("kernel_shape", std::move(kernel_shape));
        op->setAttribute("strides", std::move(strides));
        op->setAttribute("pads", std::move(pads));
        return op;
    }
};

inline void Value::replaceAllUsesWith(Value* newVal) {
    auto current_users = users_;
    for (auto* user : current_users) {
        for (size_t i = 0; i < user->numOperands(); ++i) {
            if (user->operand(i) == this) {
                user->setOperand(i, newVal);
            }
        }
    }
}

} // namespace seecpp::sir
