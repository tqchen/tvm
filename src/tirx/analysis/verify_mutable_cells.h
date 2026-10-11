/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_TIRX_ANALYSIS_VERIFY_MUTABLE_CELLS_H_
#define TVM_TIRX_ANALYSIS_VERIFY_MUTABLE_CELLS_H_

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/mutable_cell.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/type.h>

#include <unordered_map>
#include <unordered_set>

#include "../ir/tir_visitor_with_path.h"

namespace tvm::tirx {

inline bool ContainsMutableCellType(const Type& type) {
  auto found = ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      type, [](const MutableCellTypeNode*) -> ffi::Expected<ffi::WalkResult> {
        return ffi::WalkResult::Interrupt();
      });
  return found.has_value();
}

/*! \brief Verify only the local, non-aliasing cell contract.
 *
 * Ordinary variables and other operation signatures are deliberately outside
 * this verifier, so backend entry points can check cells without imposing a
 * different dialect's full well-formedness requirements.
 */
template <typename PathVisitor>
class MutableCellVerifier : public Verifier<MutableCellVerifier<PathVisitor>, PathVisitor> {
  using Verifier = tirx::Verifier<MutableCellVerifier<PathVisitor>, PathVisitor>;
  using AccessPath = ffi::reflection::AccessPath;

 public:
  using Verifier::Verifier;
  using Verifier::Verify;

 private:
  using Verifier::Visit;

  void Visit(const Function& function, AccessPath path) override {
    Verify(!ContainsMutableCellType(function->ret_type))
        << "Mutable cells cannot be function results at " << path;
    PathVisitor::Visit(function, path);
    defined_.clear();
    seen_.clear();
  }

  void Visit(const Expr& value, AccessPath path) override {
    if (value.template as<LambdaExprNode>()) {
      // A closure cannot retain a cell allocated in its enclosing function.
      ++parallel_depth_;
      PathVisitor::Visit(value, path);
      --parallel_depth_;
      return;
    }
    if (ContainsMutableCellType(value->ty) && !value.template as<VarNode>() &&
        !value.template as<CallNode>()) {
      Verify(false) << "Mutable cell handles cannot be embedded in values at " << path;
    }
    PathVisitor::Visit(value, path);
  }

  void EnterDef(const Var& var, AccessPath path) override {
    if (!var->ty.template as<MutableCellTypeNode>()) {
      Verify(!ContainsMutableCellType(var->ty))
          << "Mutable cells cannot be embedded in variable types at " << path;
      return;
    }
    Verify(var.get() == binding_)
        << "Mutable cells must be bound directly to mutable_cell_alloc; cell parameters and "
           "aliases are unsupported at "
        << path;
    Verify(vector_depth_ == 0) << "Mutable cells are unsupported in vectorized loops at " << path;
    Verify(seen_.insert(var.get()).second)
        << "Mutable cell variable has more than one definition at " << path;
    defined_[var.get()] = parallel_depth_;
  }

  void ExitDef(const Var& var, AccessPath path) override { defined_.erase(var.get()); }

  void Dispatch_(const VarNode* var, AccessPath path) override {
    Verify(!ContainsMutableCellType(var->ty))
        << "Mutable cell handles may only be used directly by mutable_cell_load/store at " << path;
  }

  void Dispatch_(const BindNode* op, AccessPath path) override {
    const auto* call = op->value.template as<CallNode>();
    bool alloc = call && call->op.same_as(mutable_cell_alloc_op());
    bool cell = op->var->ty.template as<MutableCellTypeNode>() != nullptr;
    if (cell || alloc) {
      Verify(cell && alloc && ffi::StructuralEqual()(op->var->ty, op->value->ty))
          << "Mutable cell binding requires a matching mutable_cell_alloc result at " << path;
    }
    const VarNode* previous_binding = binding_;
    const CallNode* previous_alloc = allocation_;
    binding_ = cell && alloc ? op->var.get() : nullptr;
    allocation_ = cell && alloc ? call : nullptr;
    PathVisitor::Dispatch_(op, path);
    binding_ = previous_binding;
    allocation_ = previous_alloc;
  }

  void Dispatch_(const EvaluateNode* op, AccessPath path) override {
    const CallNode* previous = store_;
    store_ = op->value.template as<CallNode>();
    PathVisitor::Dispatch_(op, path);
    store_ = previous;
  }

  void Dispatch_(const CallNode* op, AccessPath path) override {
    bool alloc = op->op.same_as(mutable_cell_alloc_op());
    bool load = op->op.same_as(mutable_cell_load_op());
    bool store = op->op.same_as(mutable_cell_store_op());
    if (alloc || load || store) {
      Verify(vector_depth_ == 0) << "Mutable cells are unsupported in vectorized loops at " << path;
      try {
        ffi::GetRef<Call>(op).Validate();
      } catch (const ffi::Error& error) {
        Verify(false) << "Invalid mutable cell operation at " << path << ": " << error.what();
        return;
      }
      if (alloc) {
        Verify(op == allocation_) << "mutable_cell_alloc must be bound directly to a cell Var at "
                                  << path;
        // Do not allow a shared allocation node to appear inside its initializer.
        const CallNode* previous = allocation_;
        allocation_ = nullptr;
        Visit(op->args[0], path->Attr("args")->ArrayItem(0));
        allocation_ = previous;
      } else {
        if (store) {
          Verify(op == store_) << "mutable_cell_store must be a direct Evaluate statement at "
                               << path;
        }
        Var cell = op->args[0].template as_or_throw<Var>();
        auto it = defined_.find(cell.get());
        Verify(it != defined_.end())
            << "Mutable cell is undefined or outside its lexical lifetime at " << path;
        if (it != defined_.end()) {
          Verify(it->second == parallel_depth_)
              << "Mutable cells cannot be captured across parallel or thread scopes at " << path;
        }
        if (store) Visit(op->args[1], path->Attr("args")->ArrayItem(1));
      }
      return;
    }
    Verify(!ContainsMutableCellType(op->ty))
        << "Only mutable_cell_alloc may produce a mutable cell handle at " << path;
    Verify(!ContainsMutableCellType(op->op->ty))
        << "Mutable cell handles cannot be called at " << path;
    if (op->op.same_as(address_of_op()) && !op->args.empty()) {
      const auto* load = op->args[0].template as<CallNode>();
      Verify(!load || !load->op.same_as(mutable_cell_load_op()))
          << "Taking the address of a mutable cell is unsupported at " << path;
    }
    PathVisitor::Dispatch_(op, path);
  }

  void Dispatch_(const ForNode* op, AccessPath path) override {
    // Headers execute in the enclosing scope, before any iteration starts.
    Visit(op->min, path->Attr("min"));
    Visit(op->extent, path->Attr("extent"));
    Visit(op->step, path->Attr("step"));
    auto context = this->WithDef(op->loop_var, path->Attr("loop_var"));
    bool parallel = op->kind == ForKind::kParallel;
    bool vector = op->kind == ForKind::kVectorized;
    parallel_depth_ += parallel;
    vector_depth_ += vector;
    this->bind_scope_.WithNewScope([&]() { Visit(op->body, path->Attr("body")); });
    vector_depth_ -= vector;
    parallel_depth_ -= parallel;
  }

  void Dispatch_(const RegionStmtNode* op, AccessPath path) override {
    bool parallel = op->op.same_as(launch_thread_op()) || op->op.same_as(parallel_launch_op()) ||
                    op->op.same_as(device_entry_op());
    // Region operands are evaluated by the enclosing thread.
    Visit(op->args, path->Attr("args"));
    Visit(ffi::AnyView(op->attrs), path->Attr("attrs"));
    for (size_t i = 0; i < op->body_params.size(); ++i) {
      Visit(op->body_params[i]->ty, path->Attr("body_params")->ArrayItem(i)->Attr("ty"));
    }
    parallel_depth_ += parallel;
    this->bind_scope_.WithNewScope([&]() {
      for (size_t i = 0; i < op->body_params.size(); ++i) {
        this->bind_scope_.Current().push_back(
            this->WithDef(op->body_params[i], path->Attr("body_params")->ArrayItem(i), false));
      }
      Visit(op->body, path->Attr("body"));
    });
    parallel_depth_ -= parallel;
    for (size_t i = 0; i < op->result_vars.size(); ++i) {
      this->bind_scope_.Current().push_back(
          this->WithDef(op->result_vars[i], path->Attr("result_vars")->ArrayItem(i)));
    }
  }

  const VarNode* binding_{nullptr};
  const CallNode* allocation_{nullptr};
  const CallNode* store_{nullptr};
  int parallel_depth_{0};
  int vector_depth_{0};
  std::unordered_map<const VarNode*, int> defined_;
  std::unordered_set<const VarNode*> seen_;
};

}  // namespace tvm::tirx
#endif  // TVM_TIRX_ANALYSIS_VERIFY_MUTABLE_CELLS_H_
