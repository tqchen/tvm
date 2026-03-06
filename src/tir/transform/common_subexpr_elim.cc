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

/*!
 * \file common_subexpr_elim.cc
 * \brief Two-phase Common Subexpression Elimination for TIR.
 *
 * Clean-room implementation:
 *   CSEPlanner: scans tree, builds scope tree + expression table, returns
 *     (insert_before, expr_remap) plan.
 *   CSERewriter: mechanical insertion of Bind stmts + expression substitution.
 *   Cascade loop handles multi-level CSE opportunities.
 */

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../analysis/check_contains.h"

namespace tvm {
namespace tir {

// ============================================================================
// Types for the plan interface (internal, C++ only)
// ============================================================================

/*! \brief Map from expression to CSE variable, using structural equality. */
using ExprRemapTable = std::unordered_map<PrimExpr, Var, ffi::StructuralHash, ExprDeepEqual>;

/*! \brief Map from Stmt (by pointer identity) to list of Bind stmts to insert. */
using InsertBeforeTable =
    std::unordered_map<Stmt, std::vector<Stmt>, ObjectPtrHash, ObjectPtrEqual>;

// ============================================================================
// Eligibility predicates
// ============================================================================

static bool ForbiddenComputation(const PrimExpr& expr) {
  return (expr.as<CallNode>() != nullptr || expr.as<BufferLoadNode>() != nullptr);
}

static bool IsEligible(const PrimExpr& expr) {
  // Leaf nodes: not eligible (no computation to save)
  if (expr.as<IntImmNode>() || expr.as<FloatImmNode>() || expr.as<StringImmNode>() ||
      expr.as<VarNode>()) {
    return false;
  }
  // Forbidden top-level ops
  if (ForbiddenComputation(expr)) return false;
  // Ramp and Broadcast: not eligible
  if (expr.as<RampNode>() || expr.as<BroadcastNode>()) return false;
  // Contains forbidden sub-expression
  if (CheckContains::ExprContains(expr, ForbiddenComputation)) return false;
  return true;
}

// ============================================================================
// CSEPlanner: Phase 1 -- scan tree, build scope tree + expression table
// ============================================================================

/*! \brief One node in the scope tree. */
struct ScopeEntry {
  int parent;        /*!< Parent scope ID (-1 for root) */
  int depth;         /*!< Tree depth (root = 0) */
  Stmt creator_stmt; /*!< For/IfThenElse/etc that created this scope */
};

/*! \brief Metadata per unique expression. */
struct ExprEntry {
  int count{0};            /*!< Number of occurrences */
  int expr_depth{0};       /*!< Expression tree height (leaf=0) */
  PrimExpr repr;           /*!< The expression itself */
  int lca_scope{-1};       /*!< LCA of all scopes containing an occurrence */
  int first_use_scope{-1}; /*!< Scope ID of the first occurrence */
  Stmt first_use_stmt;     /*!< First SeqStmt child containing this expr */
};

/*! \brief Expression table keyed by structural equality. */
using ExprTable = std::unordered_map<PrimExpr, ExprEntry, ffi::StructuralHash, ExprDeepEqual>;

/*! \brief Compute expression depth (nesting level of eligible sub-expressions). */
static int ComputeExprDepth(const PrimExpr& e) {
  struct DepthVisitor : public ExprVisitor {
    int max_depth = 0;
    int current = 0;
    void VisitExpr(const PrimExpr& expr) override {
      bool eligible = IsEligible(expr);
      if (eligible) current++;
      max_depth = std::max(max_depth, current);
      ExprVisitor::VisitExpr(expr);
      if (eligible) current--;
    }
  };
  DepthVisitor v;
  v.VisitExpr(e);
  return v.max_depth;
}

class CSEPlanner : public StmtExprVisitor {
 public:
  /*!
   * \brief Run the planner on a function body.
   * \param body The function body.
   * \param params The function parameters (always in scope at root).
   * \return A pair of (insert_before, expr_remap).
   */
  static std::pair<InsertBeforeTable, ExprRemapTable> Plan(const Stmt& body,
                                                           const ffi::Array<Var>& params) {
    CSEPlanner planner;
    // Root scope
    planner.scopes_.push_back({-1, 0, Stmt()});
    planner.current_scope_ = 0;
    // Initialize current_seq_child_ to the body itself, so that when the
    // body is not a SeqStmt, first_use_stmt still points to a valid stmt.
    planner.current_seq_child_ = body;
    // Scan
    planner.VisitStmt(body);
    // Compute plan
    return planner.ComputePlan();
  }

 protected:
  // ------------------------------------------------------------------
  // Scope tree operations
  // ------------------------------------------------------------------
  int AllocScope(int parent, Stmt creator_stmt) {
    int id = static_cast<int>(scopes_.size());
    scopes_.push_back({parent, scopes_[parent].depth + 1, std::move(creator_stmt)});
    return id;
  }

  int LCA(int a, int b) const {
    while (scopes_[a].depth > scopes_[b].depth) a = scopes_[a].parent;
    while (scopes_[b].depth > scopes_[a].depth) b = scopes_[b].parent;
    while (a != b) {
      a = scopes_[a].parent;
      b = scopes_[b].parent;
    }
    return a;
  }

  Stmt FindInsertionStmt(int first_use_scope, int lca_scope) const {
    int s = first_use_scope;
    while (scopes_[s].parent != lca_scope) s = scopes_[s].parent;
    return scopes_[s].creator_stmt;
  }

  // ------------------------------------------------------------------
  // Expression recording
  // ------------------------------------------------------------------
  void RecordExpr(const PrimExpr& e) {
    if (!IsEligible(e)) return;

    ExprEntry& entry = table_[e];
    if (entry.count == 0) {
      entry.lca_scope = current_scope_;
      entry.first_use_scope = current_scope_;
      entry.first_use_stmt = current_seq_child_;
      entry.repr = e;
      entry.expr_depth = ComputeExprDepth(e);
    } else {
      entry.lca_scope = LCA(entry.lca_scope, current_scope_);
    }
    entry.count += 1;
  }

  // ------------------------------------------------------------------
  // Visitor overrides
  // ------------------------------------------------------------------
  using StmtExprVisitor::VisitExpr_;

  // Binary arithmetic ops
  void VisitExpr_(const AddNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const SubNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const MulNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const DivNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const ModNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const FloorDivNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const FloorModNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const MinNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const MaxNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  // Comparison ops
  void VisitExpr_(const EQNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const NENode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const LTNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const LENode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const GTNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const GENode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  // Logical ops
  void VisitExpr_(const AndNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const OrNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  void VisitExpr_(const NotNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  // Cast
  void VisitExpr_(const CastNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }
  // Select
  void VisitExpr_(const SelectNode* op) override {
    StmtExprVisitor::VisitExpr_(op);
    RecordExpr(ffi::GetRef<PrimExpr>(op));
  }

  // Statement visitors
  void VisitStmt_(const SeqStmtNode* op) override {
    for (const auto& child : op->seq) {
      current_seq_child_ = child;
      VisitStmt(child);
    }
  }

  void VisitStmt_(const ForNode* op) override {
    int saved = current_scope_;
    current_scope_ = AllocScope(saved, ffi::GetRef<Stmt>(op));
    StmtExprVisitor::VisitStmt_(op);
    current_scope_ = saved;
  }

  void VisitStmt_(const IfThenElseNode* op) override {
    // Condition is in the parent scope
    VisitExpr(op->condition);
    int saved = current_scope_;
    // Then-scope
    current_scope_ = AllocScope(saved, ffi::GetRef<Stmt>(op));
    VisitStmt(op->then_case);
    // Else-scope
    if (op->else_case) {
      current_scope_ = AllocScope(saved, ffi::GetRef<Stmt>(op));
      VisitStmt(op->else_case.value());
    }
    current_scope_ = saved;
  }

  void VisitStmt_(const WhileNode* op) override {
    VisitExpr(op->condition);
    int saved = current_scope_;
    current_scope_ = AllocScope(saved, ffi::GetRef<Stmt>(op));
    VisitStmt(op->body);
    current_scope_ = saved;
  }

  void VisitStmt_(const AttrStmtNode* op) override {
    VisitExpr(op->value);
    int saved = current_scope_;
    current_scope_ = AllocScope(saved, ffi::GetRef<Stmt>(op));
    VisitStmt(op->body);
    current_scope_ = saved;
  }

  void VisitStmt_(const AllocBufferNode* op) override { VisitBufferDef(op->buffer, true); }

  void VisitStmt_(const DeclBufferNode* op) override { VisitBufferDef(op->buffer, false); }

  // ------------------------------------------------------------------
  // ComputePlan: convert internal state to output maps
  // ------------------------------------------------------------------
  std::pair<InsertBeforeTable, ExprRemapTable> ComputePlan() {
    // Sort all entries by depth descending, then by structural hash for determinism
    std::vector<std::pair<PrimExpr, ExprEntry*>> all_entries;
    for (auto& kv : table_) {
      all_entries.push_back({kv.first, &kv.second});
    }

    std::stable_sort(
        all_entries.begin(), all_entries.end(),
        [](const std::pair<PrimExpr, ExprEntry*>& a, const std::pair<PrimExpr, ExprEntry*>& b) {
          if (a.second->expr_depth != b.second->expr_depth)
            return a.second->expr_depth > b.second->expr_depth;
          // Tie-break by structural hash
          ffi::StructuralHash hasher;
          size_t ha = hasher(a.first);
          size_t hb = hasher(b.first);
          return ha < hb;
        });

    InsertBeforeTable insert_before;
    ExprRemapTable expr_remap;
    int counter = 0;
    ExprDeepEqual expr_eq;

    for (auto& [expr, entry] : all_entries) {
      // After subtracting sub-expression counts, check if still a candidate
      if (entry->count < 2) continue;

      // Determine insertion point
      Stmt insert_at;
      if (entry->first_use_scope == entry->lca_scope) {
        insert_at = entry->first_use_stmt;
      } else {
        insert_at = FindInsertionStmt(entry->first_use_scope, entry->lca_scope);
      }

      // Create CSE var and Bind
      ++counter;
      std::string name = "cse_v" + std::to_string(counter);
      Var cse_var(name, entry->repr.dtype());
      Stmt bind = Bind(cse_var, entry->repr);

      // Append to insert_before list for this stmt
      insert_before[insert_at].push_back(bind);
      expr_remap[entry->repr] = cse_var;

      // Subtract this expression's count from all its eligible sub-expressions.
      // After CSE of this expr, occurrences of its sub-expressions that were
      // inside this expr are removed (they become part of the Bind value, not
      // separate occurrences). The cascade loop will find new opportunities.
      int parent_count = entry->count;
      for (auto& [sub_expr, sub_entry] : all_entries) {
        if (sub_entry == entry) continue;
        if (sub_entry->expr_depth >= entry->expr_depth) continue;
        // Check if sub_expr is a sub-expression of expr
        if (CheckContains::ExprContains(
                expr, [&sub_expr, &expr_eq](const PrimExpr& e) { return expr_eq(e, sub_expr); })) {
          sub_entry->count -= parent_count;
        }
      }
    }

    return {insert_before, expr_remap};
  }

 private:
  std::vector<ScopeEntry> scopes_;
  ExprTable table_;
  int current_scope_ = 0;
  Stmt current_seq_child_;
};

// ============================================================================
// CSERewriter: Phase 2 -- mechanical insertion + substitution
// ============================================================================

class CSERewriter : public StmtExprMutator {
 public:
  CSERewriter(InsertBeforeTable insert_before, ExprRemapTable expr_remap)
      : insert_before_(std::move(insert_before)), expr_remap_(std::move(expr_remap)) {}

  Stmt Rewrite(const Stmt& body) { return VisitBody(body); }

 protected:
  using StmtExprMutator::VisitExpr;
  using StmtExprMutator::VisitExpr_;

  PrimExpr VisitExpr(const PrimExpr& e) override {
    // Check for remap before recursing -- match using structural equality
    auto it = expr_remap_.find(e);
    if (it != expr_remap_.end()) {
      return it->second;
    }
    return StmtExprMutator::VisitExpr(e);
  }

  /*!
   * \brief Visit a body statement, prepending any insert_before binds.
   *
   * If the body statement itself has insert_before entries, wrap with SeqStmt.
   * This handles cases where the insertion point is a direct body (not wrapped
   * in SeqStmt), e.g., the body of IfThenElse.then_case is an IfThenElse.
   */
  Stmt VisitBody(const Stmt& body) {
    auto it = insert_before_.find(body);
    if (it != insert_before_.end()) {
      ffi::Array<Stmt> new_stmts;
      for (const auto& bind_stmt : it->second) {
        new_stmts.push_back(bind_stmt);
      }
      new_stmts.push_back(VisitStmt(body));
      return SeqStmt(new_stmts);
    }
    return VisitStmt(body);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) override {
    ffi::Array<Stmt> new_stmts;
    for (const auto& child : op->seq) {
      // Insert planned Bind stmts before this child (pointer identity match)
      auto it = insert_before_.find(child);
      if (it != insert_before_.end()) {
        for (const auto& bind_stmt : it->second) {
          new_stmts.push_back(bind_stmt);
        }
      }
      new_stmts.push_back(VisitStmt(child));
    }
    return SeqStmt(new_stmts);
  }

  Stmt VisitStmt_(const ForNode* op) override {
    Stmt body_new = VisitBody(op->body);
    PrimExpr min_new = VisitExpr(op->min);
    PrimExpr extent_new = VisitExpr(op->extent);
    if (body_new.same_as(op->body) && min_new.same_as(op->min) && extent_new.same_as(op->extent)) {
      return ffi::GetRef<Stmt>(op);
    }
    return For(op->loop_var, min_new, extent_new, op->kind, body_new, op->thread_binding,
               op->annotations, op->step, op->span);
  }

  Stmt VisitStmt_(const IfThenElseNode* op) override {
    PrimExpr cond_new = VisitExpr(op->condition);
    Stmt then_new = VisitBody(op->then_case);
    ffi::Optional<Stmt> else_new;
    if (op->else_case) {
      else_new = VisitBody(op->else_case.value());
    }
    if (cond_new.same_as(op->condition) && then_new.same_as(op->then_case) &&
        else_new.same_as(op->else_case)) {
      return ffi::GetRef<Stmt>(op);
    }
    return IfThenElse(cond_new, then_new, else_new, op->span);
  }

  Stmt VisitStmt_(const WhileNode* op) override {
    PrimExpr cond_new = VisitExpr(op->condition);
    Stmt body_new = VisitBody(op->body);
    if (cond_new.same_as(op->condition) && body_new.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }
    return While(cond_new, body_new, op->span);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) override {
    PrimExpr value_new = VisitExpr(op->value);
    Stmt body_new = VisitBody(op->body);
    if (value_new.same_as(op->value) && body_new.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }
    return AttrStmt(op->node, op->attr_key, value_new, body_new, op->span);
  }

  // AllocBuffer/DeclBuffer are flat (no body) — base class handles them.

 private:
  InsertBeforeTable insert_before_;
  ExprRemapTable expr_remap_;
};

// ============================================================================
// Cascade loop
// ============================================================================

static Stmt CommonSubExprElim(Stmt body, ffi::Array<Var> params, bool identify_equiv_terms) {
  while (true) {
    auto [insert_before, expr_remap] = CSEPlanner::Plan(body, params);
    if (insert_before.empty()) break;
    body = CSERewriter(insert_before, expr_remap).Rewrite(body);
  }
  return body;
}

// ============================================================================
// Pass registration
// ============================================================================

namespace transform {

Pass CommonSubexprElimTIR(bool enable_cse_tir, bool identify_equiv_terms) {
  auto pass_func = [enable_cse_tir, identify_equiv_terms](PrimFunc f, IRModule m, PassContext ctx) {
    if (enable_cse_tir) {
      auto* n = f.CopyOnWrite();
      n->body = CommonSubExprElim(std::move(f->body), f->params, identify_equiv_terms);
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.CommonSubexprElimTIR", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.transform.CommonSubexprElimTIR", CommonSubexprElimTIR);
}

}  // namespace transform
}  // namespace tir
}  // namespace tvm
