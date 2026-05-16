#include "middle-end/op_fusion_pass.hpp"
#include "utility/logger.hpp"

#include <cmath>
#include <format>
#include <string_view>
#include <unordered_set>

namespace seecpp::middle {

PassResult OperatorFusionPass::runOnBlock(sir::Block& block) {
    fused_conv_bn_     = 0;
    fused_matmul_relu_ = 0;
    fused_elementwise_ = 0;

    // We use a tombstone set to prevent use-after-free bugs when 
    // iterating over pre-collected vectors of operations.
    std::unordered_set<sir::Operation*> dead_ops;

    // --- Pass 1: Conv-BN fusion (sc_high, pre-lowering) ---
    {
        std::vector<sir::Operation*> bn_ops;
        block.walk([&](sir::Operation* op) {
            if (op->mnemonic() == "sc_high.batch_norm") bn_ops.push_back(op);
        });
        for (auto* bn : bn_ops) {
            fuseConvBatchNorm(block, bn, dead_ops);
        }
    }

    // --- Pass 2: MatMul/Conv-Relu fusion ---
    {
        std::vector<sir::Operation*> relu_ops;
        block.walk([&](sir::Operation* op) {
            if (op->mnemonic() == "sc_high.relu" || op->mnemonic() == "sc_low.relu") {
                relu_ops.push_back(op);
            }
        });
        for (auto* relu : relu_ops) {
            // Skip if the ReLU was already deleted by a previous fusion
            if (dead_ops.count(relu)) continue; 
            fuseMatMulRelu(block, relu, dead_ops);
        }
    }

    // --- Pass 3: Generic elementwise chain fusion ---
    {
        std::vector<sir::Operation*> ew_ops;
        block.walk([&](sir::Operation* op) {
            std::string_view mn = op->mnemonic();
            if (mn == "sc_high.add" || mn == "sc_high.mul" ||
                mn == "sc_high.sub" || mn == "sc_high.div" || 
                mn == "sc_high.fused_ew") { // Allow cascading fusions
                ew_ops.push_back(op);
            }
        });
        for (auto* ew : ew_ops) {
            if (dead_ops.count(ew)) continue;
            fuseElementwiseChain(block, ew, dead_ops);
        }
    }

    utility::Logger::info(std::format(
        "OperatorFusionPass: fused {} conv+bn, {} matmul+relu, {} elementwise chain(s)",
        fused_conv_bn_, fused_matmul_relu_, fused_elementwise_));

    return {};
}

bool OperatorFusionPass::fuseConvBatchNorm(
    sir::Block& block, sir::Operation* bn_op, std::unordered_set<sir::Operation*>& dead_ops) 
{
    if (bn_op->numOperands() < 5) return false;

    sir::Value*     bn_input = bn_op->operand(0);
    sir::Operation* conv_op  = bn_input->definingOp();
    
    if (!conv_op || conv_op->mnemonic() != "sc_high.conv2d") return false;
    if (!bn_input->hasOneUse()) return false;

    auto isConstant = [](sir::Value* v) -> bool {
        return v && v->definingOp() && v->definingOp()->mnemonic() == "sc_high.constant";
    };

    if (!isConstant(bn_op->operand(1)) || !isConstant(bn_op->operand(2)) || 
        !isConstant(bn_op->operand(3)) || !isConstant(bn_op->operand(4))) {
        return false;
    }

    float epsilon = bn_op->getAttrAs<float>("epsilon").value_or(1e-5f);

    conv_op->setAttribute("fused_bn",    int64_t(1));
    conv_op->setAttribute("bn_epsilon",  epsilon);
    conv_op->setAttribute("bn_scale_id", std::string(bn_op->operand(1)->id()));
    conv_op->setAttribute("bn_bias_id",  std::string(bn_op->operand(2)->id()));
    conv_op->setAttribute("bn_mean_id",  std::string(bn_op->operand(3)->id()));
    conv_op->setAttribute("bn_var_id",   std::string(bn_op->operand(4)->id()));

    bn_op->result(0)->replaceAllUsesWith(conv_op->result(0));
    conv_op->result(0)->setShape(bn_op->result(0)->shape());

    dead_ops.insert(bn_op);
    block.removeOp(bn_op);
    ++fused_conv_bn_;
    
    return true;
}

bool OperatorFusionPass::fuseMatMulRelu(
    sir::Block& block, sir::Operation* relu_op, std::unordered_set<sir::Operation*>& dead_ops) 
{
    if (relu_op->numOperands() < 1) return false;

    sir::Value*     relu_input = relu_op->operand(0);
    sir::Operation* producer   = relu_input->definingOp();
    
    if (!producer || dead_ops.count(producer)) return false;

    std::string_view mn = producer->mnemonic();
    const bool is_fusible_producer = (
        mn == "sc_low.matmul"  || mn == "sc_high.matmul" || 
        mn == "sc_high.gemm"   || mn == "sc_high.conv2d" || // Fixed: Added Conv2D
        mn == "sc_low.conv2d"
    );

    if (!is_fusible_producer || !relu_input->hasOneUse()) return false;

    producer->setAttribute("activation", std::string("relu"));
    relu_op->result(0)->replaceAllUsesWith(producer->result(0));

    dead_ops.insert(relu_op);
    block.removeOp(relu_op);
    ++fused_matmul_relu_;
    
    return true;
}

bool OperatorFusionPass::fuseElementwiseChain(
    sir::Block& block, sir::Operation* consumer_op, std::unordered_set<sir::Operation*>& dead_ops) 
{
    for (size_t i = 0; i < consumer_op->numOperands(); ++i) {
        sir::Value*     v        = consumer_op->operand(i);
        sir::Operation* producer = v->definingOp();
        
        if (!producer || dead_ops.count(producer)) continue;

        std::string_view pmn = producer->mnemonic();
        const bool is_ew_producer = (
            pmn == "sc_high.add" || pmn == "sc_high.mul" ||
            pmn == "sc_high.sub" || pmn == "sc_high.div" || 
            pmn == "sc_high.fused_ew"
        );

        if (!is_ew_producer || !v->hasOneUse()) continue;

        // Extract prior sequence if the producer is already a fused op
        std::string p_seq = (pmn == "sc_high.fused_ew") 
            ? producer->getAttrAs<std::string>("op_sequence").value_or("unknown")
            : std::string(pmn);

        std::string c_seq = (consumer_op->mnemonic() == "sc_high.fused_ew")
            ? consumer_op->getAttrAs<std::string>("op_sequence").value_or("unknown")
            : std::string(consumer_op->mnemonic());

        std::string op_seq = std::format("{}+{}", p_seq, c_seq);

        // Fixed: Topological insertion instead of appending to block end
        auto fused_op = block.insertOpBefore("sc_high.fused_ew", consumer_op);
        fused_op->setAttribute("op_sequence", op_seq);

        for (size_t j = 0; j < producer->numOperands(); ++j) {
            fused_op->addOperand(producer->operand(j));
        }
        for (size_t j = 0; j < consumer_op->numOperands(); ++j) {
            if (consumer_op->operand(j) != v) {
                fused_op->addOperand(consumer_op->operand(j));
            }
        }

        fused_op->addResult("", consumer_op->result(0)->dtype(), consumer_op->result(0)->shape());
        consumer_op->result(0)->replaceAllUsesWith(fused_op->result(0));

        dead_ops.insert(consumer_op);
        dead_ops.insert(producer);
        
        block.removeOp(consumer_op);
        block.removeOp(producer);
        
        ++fused_elementwise_;
        return true;
    }
    return false;
}

} // namespace seecpp::middle
