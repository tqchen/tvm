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
 * \file unwrap_vector_expr.cc
 * \brief Utility for tracking currently active constraints
 */

#include "unwrap_vector_expr.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/cast.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>

#include <unordered_map>

namespace tvm {
namespace arith {

using namespace tirx;

class Scalarizer : public ExprMutator {
 public:
  explicit Scalarizer(PrimExpr lane) : lane_(lane) {}

  PrimExpr VisitExpr_(const RampNode* op) final { return op->base + lane_ * op->stride; }

  PrimExpr VisitExpr_(const BroadcastNode* op) final { return op->value; }

  PrimExpr VisitExpr_(const VarNode* op) final { return ExprMutator::VisitExpr_(op); }

 private:
  // The lane to extract
  PrimExpr lane_;
};

PrimExpr UnwrapVectorExpr(const PrimExpr& vector_expr, const PrimExpr& lane) {
  return Scalarizer(lane)(vector_expr);
}

}  // namespace arith
}  // namespace tvm
