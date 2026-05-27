#include "source/middle_end/transforms/kernel_fuser.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "include/utility/logger.hpp"
#include "seecpp/sir/sir.h"

namespace seecpp::middle_end::transforms {

namespace {

// Registry of operations that map 1:1 on their input tensors without 
// requiring spatial reductions or cross-thread synchronization.
const std::unordered_set<std::string_view>& GetElementwiseOps() {
  static const auto* const kElementwise =
      new std::unordered_set<std::string_view>{
          "Add", "Sub", "Mul", "Div", 
          "Relu", "Sigmoid", "Tanh", "Exp", "Log",
          "ReluGrad" // Adjoint operations are also highly fuse-able
      };
  return *kElementwise;
}

}  // namespace

bool KernelFuser::Run(sir::Block& block) {
  bool graph_changed = false;
  bool pass_changed;

  // Fixed-point iteration: fusing A->B might allow the new AB node to fuse 
  // with C, creating ABC. We loop until no more fusions are possible.
  do {
    pass_changed = false;
    std::unordered_map<std::string, int> use_counts = ComputeUseCounts(block);
    std::vector<sir::Operation> current_ops = block.operations();

    for (auto& consumer : current_ops) {
      if (!block.HasOperation(consumer.name())) continue;

      if (TryFusePair(consumer, block, use_counts)) {
        pass_changed = true;
        graph_changed = true;
        // Break early to recompute use_counts on the modified graph to 
        // guarantee safety and iterator stability.
        break; 
      }
    }
  } while (pass_changed);

  if (graph_changed) {
    utility::Logger::debug("KernelFuser: Operator fusion converged.");
  }
  
  return graph_changed;
}

bool KernelFuser::TryFusePair(
    sir::Operation& consumer, sir::Block& block,
    const std::unordered_map<std::string, int>& use_counts) {
  
  if (!IsElementwise(consumer.mnemonic())) return false;

  const std::vector<std::string>& operands = consumer.operand_names();

  for (size_t i = 0; i < operands.size(); ++i) {
    const std::string& producer_name = operands[i];
    
    // Condition 1: Producer must have exactly one consumer. If it fans out,
    // fusing it into this path would require either duplicating the math or
    // writing it to RAM anyway, defeating the purpose of the fusion.
    auto count_it = use_counts.find(producer_name);
    if (count_it == use_counts.end() || count_it->second != 1) {
      continue;
    }

    const sir::Operation* producer = block.GetOperation(producer_name);
    if (!producer) continue;

    // Condition 2: Producer must also be element-wise.
    if (!IsElementwise(producer->mnemonic())) {
      continue;
    }

    // FUSION ACCEPTED!
    // We compose a new operation that takes the unique inputs of both the 
    // producer and consumer.
    
    // 1. Create the composite mnemonic (e.g., "Add" + "Relu" -> "Fused_Add_Relu")
    std::string new_mnemonic = "Fused_";
    
    // Strip "Fused_" prefix if the producer is already a fused kernel to 
    // prevent names like Fused_Fused_Add_Relu.
    std::string prod_mnem = std::string(producer->mnemonic());
    if (prod_mnem.rfind("Fused_", 0) == 0) {
      prod_mnem = prod_mnem.substr(6); 
    }
    std::string cons_mnem = std::string(consumer.mnemonic());
    if (cons_mnem.rfind("Fused_", 0) == 0) {
      cons_mnem = cons_mnem.substr(6);
    }
    new_mnemonic += prod_mnem + "_" + cons_mnem;

    // 2. Splice the operands together. We take the consumer's operands, 
    // but replace the intermediate 'producer_name' with the producer's inputs.
    std::vector<std::string> fused_operands;
    for (size_t j = 0; j < operands.size(); ++j) {
      if (j == i) {
        const auto& prod_operands = producer->operand_names();
        fused_operands.insert(fused_operands.end(), prod_operands.begin(),
                              prod_operands.end());
      } else {
        fused_operands.push_back(operands[j]);
      }
    }

    // 3. Inject the fused node and wire it up.
    // We reuse the consumer's name so downstream nodes auto-target the new kernel.
    sir::Operation fused_kernel = sir::Operation::Create(
        consumer.name(), new_mnemonic, fused_operands);
    
    block.ReplaceOperation(consumer.name(), fused_kernel);
    
    // 4. Dead Code Elimination: The intermediate node is now completely unused.
    block.EraseOperation(producer_name);

    return true; 
  }

  return false;
}

std::unordered_map<std::string, int> KernelFuser::ComputeUseCounts(
    const sir::Block& block) const {
  std::unordered_map<std::string, int> use_counts;
  
  for (const auto& op : block.operations()) {
    for (const auto& operand : op.operand_names()) {
      use_counts[operand]++;
    }
  }
  return use_counts;
}

bool KernelFuser::IsElementwise(std::string_view mnemonic) const {
  // If it's a kernel we've already fused, it is still element-wise.
  if (mnemonic.rfind("Fused_", 0) == 0) {
    return true;
  }
  const auto& elementwise_ops = GetElementwiseOps();
  return elementwise_ops.find(mnemonic) != elementwise_ops.end();
}

}  // namespace seecpp::middle_end::transforms
