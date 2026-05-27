#include "source/middle_end/transforms/dead_code_elimination.h"

#include <format>
#include <unordered_set>
#include <vector>

#include "include/utility/logger.hpp"

namespace seecpp::middle_end::transforms {

bool DeadCodeElimination::Run(sir::Block& block) {
  std::unordered_set<sir::Operation*> live_ops;
  std::vector<sir::Operation*> worklist;

  // 1. Identify roots (Mark Phase)
  block.walk([&](sir::Operation* op) {
    if (IsRootOperation(op)) {
      live_ops.insert(op);
      worklist.push_back(op);
    }
  });

  // 2. Propagate liveness backwards via operand definitions
  while (!worklist.empty()) {
    sir::Operation* current = worklist.back();
    worklist.pop_back();

    for (size_t i = 0; i < current->numOperands(); ++i) {
      sir::Value* operand = current->operand(i);
      
      if (sir::Operation* def_op = operand->definingOp()) {
        if (live_ops.insert(def_op).second) {
          worklist.push_back(def_op);
        }
      }
    }
  }

  // 3. Identify dead operations (Sweep Phase)
  std::vector<sir::Operation*> dead_ops;
  block.walk([&](sir::Operation* op) {
    if (!live_ops.count(op)) {
      dead_ops.push_back(op);
    }
  });

  if (dead_ops.empty()) {
    return false; 
  }

  // 4. Safe topological removal
  for (auto it = dead_ops.rbegin(); it != dead_ops.rend(); ++it) {
    block.removeOp(*it);
  }

  utility::Logger::info(std::format(
      "DeadCodeElimination: removed {} dead op(s).", dead_ops.size()));

  return true;
}

bool DeadCodeElimination::IsRootOperation(const sir::Operation* op) const {
  // Explicitly guard return/yield nodes alongside memory and control flow
  std::string_view mnem = op->mnemonic();
  if (mnem == "sc_high.return" || mnem == "sc_low.return" || mnem == "sc_high.yield") {
    return true;
  }
  
  return op->isMemoryOp() || op->isControlFlow();
}

}  // namespace seecpp::middle_end::transforms
