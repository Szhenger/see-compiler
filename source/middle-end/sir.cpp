#include "include/middle-end/sir.hpp"

#include <atomic>
#include <ostream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cassert>

namespace seecpp::sir {

// =============================================================================
// 3. Value
// =============================================================================

void Value::replaceAllUsesWith(Value* newVal) {
    assert(newVal && "replaceAllUsesWith called with null value");
    assert(newVal != this && "replaceAllUsesWith called with same value");

    // For every operation that uses `this`, swap the operand pointer.
    for (Operation* user : users_) {
        for (size_t i = 0; i < user->numOperands(); ++i) {
            if (user->operand(i) == this)
                user->setOperand(i, newVal);
        }
    }
    // Transfer the user list to newVal.
    for (Operation* user : users_)
        newVal->addUser(user);
    users_.clear();
}

void Value::removeUser(Operation* op) {
    auto it = std::find(users_.begin(), users_.end(), op);
    if (it != users_.end())
        users_.erase(it);
}

// =============================================================================
// 4. Operation
// =============================================================================

// Static SSA name counter — one per process, shared across all Operations.
std::atomic<size_t> Operation::id_counter_{0};

void Operation::addOperand(Value* v) {
    assert(v && "addOperand: null Value*");
    operands_.push_back(v);
    v->addUser(this);
}

void Operation::setOperand(size_t i, Value* newVal) {
    assert(i < operands_.size() && "setOperand: index out of range");
    assert(newVal && "setOperand: null Value*");

    // Unregister from old value's user list.
    operands_[i]->removeUser(this);
    operands_[i] = newVal;
    newVal->addUser(this);
}

Value* Operation::addResult(std::string id, DataType dt, Shape sh) {
    // Auto-generate an SSA name (%0, %1, …) when no explicit id is given.
    if (id.empty())
        id = "%" + std::to_string(id_counter_.fetch_add(1, std::memory_order_relaxed));

    auto val = std::make_unique<Value>(std::move(id), dt, std::move(sh), this);
    results_.push_back(std::move(val));
    return results_.back().get();
}

// --- IR printing ------------------------------------------------------------

void Operation::print(std::ostream& os) const {
    // Print results
    if (!results_.empty()) {
        for (size_t i = 0; i < results_.size(); ++i) {
            if (i) os << ", ";
            os << results_[i]->id()
               << " : " << dtypeName(results_[i]->dtype());
            const auto& dims = results_[i]->shape().dims;
            if (!dims.empty()) {
                os << "<";
                for (size_t d = 0; d < dims.size(); ++d) {
                    if (d) os << "x";
                    if (dims[d] == Shape::kDynamic) os << "?";
                    else os << dims[d];
                }
                os << ">";
            }
        }
        os << " = ";
    }

    // Print mnemonic and operands
    os << mnemonic_ << "(";
    for (size_t i = 0; i < operands_.size(); ++i) {
        if (i) os << ", ";
        os << operands_[i]->id();
    }
    os << ")";

    // Print attributes
    if (!attributes_.empty()) {
        os << " {";
        bool first = true;
        for (const auto& [k, v] : attributes_) {
            if (!first) os << ", ";
            first = false;
            os << k << " = ";
            std::visit([&os](const auto& val) {
                using T = std::decay_t<decltype(val)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    os << '"' << val << '"';
                } else if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                                     std::is_same_v<T, std::vector<float>>) {
                    os << "[";
                    for (size_t i = 0; i < val.size(); ++i) {
                        if (i) os << ", ";
                        os << val[i];
                    }
                    os << "]";
                } else {
                    os << val;
                }
            }, v);
        }
        os << "}";
    }
}

std::string Operation::toString() const {
    std::ostringstream oss;
    print(oss);
    return oss.str();
}

// =============================================================================
// 5. Block
// =============================================================================

Value* Block::addArgument(DataType dt, Shape sh) {
    std::string id = "%" + std::to_string(Operation::id_counter_.fetch_add(
                               1, std::memory_order_relaxed));
    auto val = std::make_unique<Value>(std::move(id), dt, std::move(sh),
                                       nullptr /* block arg: no defining op */);
    args_.push_back(std::move(val));
    return args_.back().get();
}

Operation* Block::appendOp(std::string name) {
    auto op = std::make_unique<Operation>(std::move(name), this);
    ops_.push_back(std::move(op));
    return ops_.back().get();
}

Operation* Block::appendOp(std::unique_ptr<Operation> op) {
    assert(op && "appendOp: null operation");
    op->setParentBlock(this);
    ops_.push_back(std::move(op));
    return ops_.back().get();
}

std::unique_ptr<Operation> Block::removeOp(Operation* op) {
    auto it = std::find_if(ops_.begin(), ops_.end(),
                           [op](const auto& p) { return p.get() == op; });
    assert(it != ops_.end() && "removeOp: operation not found in block");

    // Unregister use-def edges so the IR stays consistent.
    for (size_t i = 0; i < op->numOperands(); ++i)
        op->operand(i)->removeUser(op);

    auto owned = std::move(*it);
    ops_.erase(it);
    owned->setParentBlock(nullptr);
    return owned;
}

bool Block::validate() const {
    // Rule 1: Every operand of every op must have been defined earlier
    //         in this block (or be a block argument). This is the core
    //         SSA dominance requirement for a linear block.
    std::unordered_set<const Value*> defined;
    for (const auto& arg : args_)
        defined.insert(arg.get());

    for (const auto& op : ops_) {
        for (const Value* operand : op->operands()) {
            if (defined.find(operand) == defined.end()) {
                // Operand used before definition — SSA violation.
                return false;
            }
        }
        for (const auto& res : op->results())
            defined.insert(res.get());
    }

    is_validated_ = true;
    return true;
}

void Block::walk(std::function<void(Operation*)> fn) {
    for (auto& op : ops_)
        fn(op.get());
}

void Block::walkReverse(std::function<void(Operation*)> fn) {
    for (auto it = ops_.rbegin(); it != ops_.rend(); ++it)
        fn(it->get());
}

void Block::print(std::ostream& os) const {
    if (!args_.empty()) {
        os << "(";
        for (size_t i = 0; i < args_.size(); ++i) {
            if (i) os << ", ";
            os << args_[i]->id()
               << ": " << dtypeName(args_[i]->dtype());
        }
        os << "):\n";
    }
    for (const auto& op : ops_) {
        os << "  ";
        op->print(os);
        os << "\n";
    }
}

// =============================================================================
// 6. Region
// =============================================================================

Block* Region::addBlock() {
    blocks_.push_back(std::make_unique<Block>());
    return blocks_.back().get();
}

// =============================================================================
// 7. OpBuilder — dialect factory helpers
// =============================================================================

// ---- sc_high dialect -------------------------------------------------------

std::unique_ptr<Operation> OpBuilder::conv2d(
    Value*               input,
    Value*               filter,
    Value*               bias,
    std::vector<int64_t> strides,
    std::vector<int64_t> pads,
    std::vector<int64_t> dilations,
    int64_t              group)
{
    assert(input  && "conv2d: null input");
    assert(filter && "conv2d: null filter");

    auto op = std::make_unique<Operation>("sc_high.conv2d");
    op->addOperand(input);
    op->addOperand(filter);
    if (bias) op->addOperand(bias);

    op->setAttribute("strides",   std::move(strides));
    op->setAttribute("pads",      std::move(pads));
    op->setAttribute("dilations", std::move(dilations));
    op->setAttribute("group",     group);

    // Result shape is intentionally left empty (dynamic-rank placeholder).
    // ShapeInferencePass will fill this in after construction.
    op->addResult("", input->dtype(), Shape{});
    return op;
}

std::unique_ptr<Operation> OpBuilder::batchNorm(
    Value* input, Value* scale, Value* bias,
    Value* running_mean, Value* running_var,
    float  epsilon)
{
    assert(input       && "batchNorm: null input");
    assert(scale       && "batchNorm: null scale");
    assert(bias        && "batchNorm: null bias");
    assert(running_mean && "batchNorm: null running_mean");
    assert(running_var  && "batchNorm: null running_var");

    auto op = std::make_unique<Operation>("sc_high.batch_norm");
    op->addOperand(input);
    op->addOperand(scale);
    op->addOperand(bias);
    op->addOperand(running_mean);
    op->addOperand(running_var);
    op->setAttribute("epsilon", epsilon);

    // BatchNorm output has same shape / dtype as input.
    op->addResult("", input->dtype(), input->shape());
    return op;
}

std::unique_ptr<Operation> OpBuilder::gemm(
    Value* A, Value* B, Value* bias,
    bool trans_a, bool trans_b)
{
    assert(A && "gemm: null A");
    assert(B && "gemm: null B");

    auto op = std::make_unique<Operation>("sc_high.gemm");
    op->addOperand(A);
    op->addOperand(B);
    if (bias) op->addOperand(bias);

    op->setAttribute("trans_a", static_cast<int64_t>(trans_a));
    op->setAttribute("trans_b", static_cast<int64_t>(trans_b));
    op->addResult("", A->dtype(), Shape{});
    return op;
}

std::unique_ptr<Operation> OpBuilder::relu(Value* input) {
    assert(input && "relu: null input");
    auto op = std::make_unique<Operation>("sc_high.relu");
    op->addOperand(input);
    // ReLU is shape- and dtype-preserving.
    op->addResult("", input->dtype(), input->shape());
    return op;
}

// ---- sc_low dialect --------------------------------------------------------

std::unique_ptr<Operation> OpBuilder::im2col(
    Value*               input,
    std::vector<int64_t> kernel_shape,
    std::vector<int64_t> strides,
    std::vector<int64_t> pads)
{
    assert(input && "im2col: null input");
    auto op = std::make_unique<Operation>("sc_low.im2col");
    op->addOperand(input);
    op->setAttribute("kernel_shape", std::move(kernel_shape));
    op->setAttribute("strides",      std::move(strides));
    op->setAttribute("pads",         std::move(pads));
    // Output is a 2-D column matrix; shape resolved by ShapeInferencePass.
    op->addResult("", input->dtype(), Shape{});
    return op;
}

} // namespace seecpp::sir