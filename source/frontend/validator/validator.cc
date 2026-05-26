#include "source/frontend/validator.h"

#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "include/utility/logger.hpp"
#include "seecpp/sir/sir.h"

namespace seecpp::frontend {

const std::unordered_set<std::string> Validator::kSupportedOps = {
    "sc_high.matmul", "sc_high.gemm",      "sc_high.conv2d",
    "sc_high.relu",   "sc_high.add",       "sc_high.sub",
    "sc_high.mul",    "sc_high.div",       "sc_high.batch_norm",
    "sc_high.reshape","sc_high.maxpool",   "sc_high.avgpool",
    "sc_high.concat", "sc_high.transpose", "sc_high.constant",
};

ValidationReport Validator::Validate(const sir::Block& block) const {
  ValidationReport report;

  utility::Logger::info("Validator: starting multi-pass validation...");

  CheckSsaLinks(block, report);
  CheckTopologicalOrder(block, report);

  for (const auto& owned_op : block.operations()) {
    const sir::Operation& op = *owned_op;
    const std::string mnemonic(op.mnemonic());

    if (kSupportedOps.find(mnemonic) == kSupportedOps.end()) {
      report.diagnostics.push_back({
          ValidationError::Severity::Error, mnemonic, "",
          "Unsupported op '" + mnemonic +
              "' — resolve sc_high.unknown nodes before validation"});
      continue;
    }

    CheckOpConstraints(op, report);
    CheckTypeConsistency(op, report);
    CheckShapeConsistency(op, report);
  }

  if (report.HasErrors()) {
    utility::Logger::error("Validator: " +
                           std::to_string(report.diagnostics.size()) +
                           " issue(s) found.");
  } else {
    utility::Logger::info("Validator: all passes clean.");
  }

  return report;
}

void Validator::CheckSsaLinks(const sir::Block& block,
                              ValidationReport& report) const {
  for (const auto& owned_op : block.operations()) {
    const sir::Operation* op = owned_op.get();
    const std::string mn(op->mnemonic());

    for (size_t i = 0; i < op->numOperands(); ++i) {
      if (op->operand(i) == nullptr) {
        report.diagnostics.push_back({ValidationError::Severity::Error, mn, "",
                                      "Operand " + std::to_string(i) +
                                          " is a null pointer"});
      }
    }

    for (size_t i = 0; i < op->numResults(); ++i) {
      const sir::Value* res = op->result(i);
      if (!res) {
        report.diagnostics.push_back({ValidationError::Severity::Error, mn, "",
                                      "Result " + std::to_string(i) +
                                          " is a null pointer"});
      } else if (res->id().empty()) {
        report.diagnostics.push_back({ValidationError::Severity::Error, mn, "",
                                      "Result " + std::to_string(i) +
                                          " has an empty SSA id"});
      }
    }

    for (size_t i = 0; i < op->numOperands(); ++i) {
      const sir::Value* v = op->operand(i);
      if (!v) continue;
      
      if (!v->isBlockArgument() && v->definingOp() == nullptr) {
        report.diagnostics.push_back({
            ValidationError::Severity::Error, mn, std::string(v->id()),
            "Operand '" + std::string(v->id()) +
                "' is not a block argument but has no defining op"});
      }
    }
  }
}

void Validator::CheckTopologicalOrder(const sir::Block& block,
                                      ValidationReport& report) const {
  std::unordered_set<const sir::Value*> defined;
  for (const auto& arg : block.arguments()) {
    defined.insert(arg.get());
  }

  for (const auto& owned_op : block.operations()) {
    const sir::Operation* op = owned_op.get();

    for (size_t i = 0; i < op->numOperands(); ++i) {
      const sir::Value* v = op->operand(i);
      if (!v) continue;
      
      if (defined.find(v) == defined.end()) {
        report.diagnostics.push_back({
            ValidationError::Severity::Error, std::string(op->mnemonic()),
            std::string(v->id()),
            "SSA violation: '" + std::string(v->id()) +
                "' used before definition"});
      }
    }

    for (size_t i = 0; i < op->numResults(); ++i) {
      if (op->result(i)) defined.insert(op->result(i));
    }
  }
}

void Validator::CheckOpConstraints(const sir::Operation& op,
                                   ValidationReport& report) const {
  const std::string mn(op.mnemonic());

  auto requireOperands = [&](size_t expected) {
    if (op.numOperands() != expected) {
      report.diagnostics.push_back({
          ValidationError::Severity::Error, mn, "",
          mn + " requires exactly " + std::to_string(expected) +
              " operand(s), got " + std::to_string(op.numOperands())});
    }
  };

  auto requireAttr = [&](std::string_view key) {
    if (!op.hasAttribute(key)) {
      report.diagnostics.push_back({
          ValidationError::Severity::Error, mn, "",
          mn + " is missing required attribute '" + std::string(key) + "'"});
    }
  };

  if (mn == "sc_high.matmul") {
    requireOperands(2);
  } else if (mn == "sc_high.gemm") {
    if (op.numOperands() < 2 || op.numOperands() > 3) {
      report.diagnostics.push_back({
          ValidationError::Severity::Error, mn, "",
          "Gemm requires 2 or 3 operands, got " +
              std::to_string(op.numOperands())});
    }
  } else if (mn == "sc_high.conv2d") {
    if (op.numOperands() < 2 || op.numOperands() > 3) {
      report.diagnostics.push_back({
          ValidationError::Severity::Error, mn, "",
          "Conv2D requires 2 or 3 operands (bias optional), got " +
              std::to_string(op.numOperands())});
    }
    requireAttr("strides");
    requireAttr("pads");
    requireAttr("dilations");
  } else if (mn == "sc_high.batch_norm") {
    requireOperands(5);
    requireAttr("epsilon");
  } else if (mn == "sc_high.relu" || mn == "sc_high.transpose" ||
             mn == "sc_high.reshape") {
    requireOperands(1);
  } else if (mn == "sc_high.add" || mn == "sc_high.sub" ||
             mn == "sc_high.mul" || mn == "sc_high.div") {
    requireOperands(2);
  } else if (mn == "sc_high.maxpool" || mn == "sc_high.avgpool") {
    requireOperands(1);
    requireAttr("kernel_shape");
    requireAttr("strides");
  } else if (mn == "sc_high.concat") {
    if (op.numOperands() < 2) {
      report.diagnostics.push_back({ValidationError::Severity::Error, mn, "",
                                    "Concat requires at least 2 operands"});
    }
    requireAttr("axis");
  } else if (mn == "sc_high.constant") {
    if (op.numOperands() != 0) {
      report.diagnostics.push_back({ValidationError::Severity::Error, mn, "",
                                    "Constant op must have zero operands"});
    }
    if (op.numResults() != 1) {
      report.diagnostics.push_back({
          ValidationError::Severity::Error, mn, "",
          "Constant op must have exactly one result"});
    }
  }
}

void Validator::CheckTypeConsistency(const sir::Operation& op,
                                     ValidationReport& report) const {
  static const std::unordered_set<std::string> kHomogeneousOps = {
      "sc_high.matmul", "sc_high.gemm", "sc_high.add",   "sc_high.sub",
      "sc_high.mul",    "sc_high.div",  "sc_high.conv2d",
  };

  const std::string mn(op.mnemonic());
  if (kHomogeneousOps.find(mn) == kHomogeneousOps.end()) return;
  if (op.numOperands() < 2) return;

  const sir::DataType expected = op.operand(0)->dtype();
  for (size_t i = 1; i < op.numOperands(); ++i) {
    if (op.operand(i)->dtype() != expected) {
      report.diagnostics.push_back({
          ValidationError::Severity::Error, mn,
          std::string(op.operand(i)->id()),
          mn + ": operand " + std::to_string(i) + " dtype '" +
              std::string(sir::dtypeName(op.operand(i)->dtype())) +
              "' does not match operand 0 dtype '" +
              std::string(sir::dtypeName(expected)) + "'"});
    }
  }
}

void Validator::CheckShapeConsistency(const sir::Operation& op,
                                      ValidationReport& report) const {
  const std::string mn(op.mnemonic());

  for (size_t i = 0; i < op.numResults(); ++i) {
    const sir::Value* res = op.result(i);
    if (!res) continue;
    if (res->shape().dims.empty()) {
      report.diagnostics.push_back({
          ValidationError::Severity::Error, mn, std::string(res->id()),
          "Result '" + std::string(res->id()) +
              "' still has an unresolved placeholder shape — "
              "run ShapeInferencePass before Validator"});
    }
  }

  if (mn == "sc_high.conv2d" && op.numOperands() >= 2) {
    auto checkRank4 = [&](size_t idx, std::string_view label) {
      if (op.operand(idx)->shape().dims.size() != 4) {
        report.diagnostics.push_back({
            ValidationError::Severity::Error, mn,
            std::string(op.operand(idx)->id()),
            "Conv2D " + std::string(label) + " must be rank-4 (NCHW)"});
      }
    };
    checkRank4(0, "input");
    checkRank4(1, "filter");

    if (op.operand(0)->shape().dims.size() == 4 &&
        op.operand(1)->shape().dims.size() == 4) {
      int64_t group = op.getAttrAs<int64_t>("group").value_or(1);
      int64_t c_in = op.operand(0)->shape().dims[1];
      int64_t c_filter = op.operand(1)->shape().dims[1];
      
      if (c_in != sir::Shape::kDynamic && c_filter != sir::Shape::kDynamic &&
          c_in != c_filter * group) {
        report.diagnostics.push_back({
            ValidationError::Severity::Error, mn, "",
            "Conv2D: input channels (" + std::to_string(c_in) +
                ") != filter_channels (" + std::to_string(c_filter) +
                ") * group (" + std::to_string(group) + ")"});
      }
    }
  }

  if (mn == "sc_high.matmul" && op.numOperands() == 2) {
    const auto& dim_a = op.operand(0)->shape().dims;
    const auto& dim_b = op.operand(1)->shape().dims;
    
    if (dim_a.size() >= 2 && dim_b.size() >= 2) {
      int64_t k_a = dim_a.back();
      int64_t k_b = dim_b[dim_b.size() - 2];
      
      if (k_a != sir::Shape::kDynamic && k_b != sir::Shape::kDynamic &&
          k_a != k_b) {
        report.diagnostics.push_back({
            ValidationError::Severity::Error, mn, "",
            "MatMul: inner dimensions mismatch (" + std::to_string(k_a) +
                " vs " + std::to_string(k_b) + ")"});
      }
    }
  }

  if (mn == "sc_high.batch_norm" && op.numOperands() == 5) {
    const auto& input_dims = op.operand(0)->shape().dims;
    if (input_dims.size() >= 2) {
      int64_t channels = input_dims[1];
      
      for (size_t i : {size_t(1), size_t(2)}) {
        const auto& d = op.operand(i)->shape().dims;
        if (d.size() != 1) {
          report.diagnostics.push_back({
              ValidationError::Severity::Error, mn,
              std::string(op.operand(i)->id()),
              "BatchNorm operand " + std::to_string(i) + " must be rank-1"});
        } else if (d[0] != sir::Shape::kDynamic &&
                   channels != sir::Shape::kDynamic && d[0] != channels) {
          report.diagnostics.push_back({
              ValidationError::Severity::Error, mn,
              std::string(op.operand(i)->id()),
              "BatchNorm operand " + std::to_string(i) + " size (" +
                  std::to_string(d[0]) + ") != input channel count (" +
                  std::to_string(channels) + ")"});
        }
      }
    }
  }
}

}  // namespace seecpp::frontend
