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

#include "verify_mutable_cells.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>

namespace tvm::tirx {

bool VerifyMutableCells(const Function& function, bool assert_mode) {
  // Functions without cells retain the surrounding dialect's validation rules.
  auto present = ffi::StructuralVisit(
      function,
      [](const MutableCellTypeNode*, ffi::StructuralVisitorObj*)
          -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> { return ffi::VisitInterrupt(); },
      [](const VarNode* var,
         ffi::StructuralVisitorObj*) -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        if (ContainsMutableCellType(var->ty)) return ffi::VisitInterrupt();
        return std::nullopt;
      },
      [](const CallNode* call,
         ffi::StructuralVisitorObj* visitor) -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        if (call->op.same_as(mutable_cell_alloc_op()) || call->op.same_as(mutable_cell_load_op()) ||
            call->op.same_as(mutable_cell_store_op())) {
          return ffi::VisitInterrupt();
        }
        return visitor->DefaultVisitExpected(call);
      });
  if (!present.has_value()) return true;
  return MutableCellVerifier<TIRVisitorWithPath>::Verify(function, assert_mode);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tirx.analysis.VerifyMutableCells", VerifyMutableCells);
}

}  // namespace tvm::tirx
