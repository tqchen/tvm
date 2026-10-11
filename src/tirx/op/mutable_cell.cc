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

/*! \file tirx/op/mutable_cell.cc
 *  \brief Initialized mutable local cell operations.
 */
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/tirx/op/mutable_cell.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/type.h>

namespace tvm::tirx {
namespace {

void ValidateSignature(const CallNode* call, size_t count) {
  TVM_FFI_CHECK_EQ(call->args.size(), count, ValueError)
      << "Mutable cell operation has an incorrect operand count";
  TVM_FFI_CHECK(!call->attrs.has_value() && call->ty_args.empty(), ValueError)
      << "Mutable cell operations do not accept attrs or type arguments";
}

MutableCellType CellOperandType(const CallNode* call, size_t count) {
  ValidateSignature(call, count);
  Var cell = call->args[0].as_or_throw<Var>();
  MutableCellType type = cell->ty.as_or_throw<MutableCellType>();
  type->Validate();
  return type;
}

Type InferAlloc(const CallNode* call) {
  ValidateSignature(call, 1);
  MutableCellType type = call->ty.as_or_throw<MutableCellType>();
  type->Validate();
  return type;
}

Type InferLoad(const CallNode* call) { return CellOperandType(call, 1)->element_type; }

void ValidateAlloc(const CallNode* call) {
  Type type = InferAlloc(call);
  TVM_FFI_CHECK(
      ffi::StructuralEqual()(call->args[0]->ty, type.as_or_throw<MutableCellType>()->element_type),
      TypeError)
      << "Mutable cell initializer must match its element type exactly";
}

void ValidateLoad(const CallNode* call) {
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty, InferLoad(call)), TypeError)
      << "Mutable cell load result must match its element type";
}

void ValidateStore(const CallNode* call) {
  MutableCellType type = CellOperandType(call, 2);
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->args[1]->ty, type->element_type), TypeError)
      << "Mutable cell store value must match its element type exactly";
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty, PrimType::Void()), TypeError)
      << "Mutable cell store must have void result type";
}

}  // namespace

const Op& mutable_cell_alloc_op() {
  static const Op op = Op::Get("tirx.mutable_cell_alloc");
  return op;
}

const Op& mutable_cell_load_op() {
  static const Op op = Op::Get("tirx.mutable_cell_load");
  return op;
}

const Op& mutable_cell_store_op() {
  static const Op op = Op::Get("tirx.mutable_cell_store");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  using Validator = ffi::reflection::NativeFunctionView<void(const CallNode*)>;
  OpDef("tirx.mutable_cell_alloc")
      .set_validator(Validator::FromNative<&ValidateAlloc>())
      .signature(sig::arg<PrimExpr>("value", "The initial element value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType, FInferType::FromNative<&InferAlloc>())
      .set_attr<TIRxOpCategory>(op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kUpdateState));
  OpDef("tirx.mutable_cell_load")
      .set_validator(Validator::FromNative<&ValidateLoad>())
      .signature(sig::arg<Var>("cell", "The local cell variable."))
      .set_attr<FInferType>(tvm::op_attr::kInferType, FInferType::FromNative<&InferLoad>())
      .set_attr<TIRxOpCategory>(op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kReadState));
  OpDef("tirx.mutable_cell_store")
      .set_validator(Validator::FromNative<&ValidateStore>())
      .signature(sig::arg<Var>("cell", "The local cell variable."),
                 sig::arg<PrimExpr>("value", "The new element value."))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr<TIRxOpCategory>(op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kUpdateState));
}

}  // namespace tvm::tirx
