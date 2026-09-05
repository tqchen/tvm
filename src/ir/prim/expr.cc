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
 * \file expr.cc
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>

#include <utility>

namespace tvm {
namespace prim {

namespace refl = tvm::ffi::reflection;

namespace {

// Primitive-expression types are statically known and source spans are deliberately skipped.
// Let variables are visited and mapped under a non-recursive definition region.

int GetLanesOrVScaleFactor(const PrimType& ty) {
  return ty.IsScalableVector() ? ty.VScaleFactor() : ty.lanes();
}

TVM_FFI_INLINE const PrimTypeNode* GetPrimTypeNode(const PrimExpr& expr) {
  const auto* node = expr.get();
  TVM_FFI_DCHECK(node != nullptr);
  TVM_FFI_DCHECK(!node->ExprNode::ty.IsMissing());
  const auto* prim_ty = node->ExprNode::ty.as<PrimTypeNode>();
  TVM_FFI_DCHECK(prim_ty != nullptr);
  return prim_ty;
}

}  // namespace

/* \brief Convert an object to a PrimExpr
 *
 * All conversions to a PrimExpr are performed as part of the FFI,
 * when calling a function that accepts a PrimExpr as an argument.  If
 * a function must normalize to a PrimExpr (e.g. before accessing the
 * `expr.dtype` field), this function allows the FFI conversions to be
 * explicitly invoked.
 */
#define TVM_DEFINE_BINOP_CONSTRUCTOR(Name)                                        \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                                 \
    using T = Name::ContainerType;                                                \
    TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined\n";                 \
    TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined\n";                 \
    const PrimTypeNode* a_ty = GetPrimTypeNode(a);                                \
    const PrimTypeNode* b_ty = GetPrimTypeNode(b);                                \
    TVM_FFI_CHECK(a_ty->dtype == b_ty->dtype, TypeError)                          \
        << "mismatched types. " << a_ty->dtype << " vs. " << b_ty->dtype << "\n"; \
    ffi::ObjectPtr<T> node = ffi::make_object<T>();                               \
    node->ExprNode::ty = a.get()->ExprNode::ty;                                   \
    node->a = std::move(a);                                                       \
    node->b = std::move(b);                                                       \
    node->span = std::move(span);                                                 \
    data_ = std::move(node);                                                      \
  }

#define TVM_DEFINE_CMPOP_CONSTRUCTOR(Name)                                        \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                                 \
    using T = Name::ContainerType;                                                \
    TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined\n";                 \
    TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined\n";                 \
    const PrimTypeNode* a_ty = GetPrimTypeNode(a);                                \
    const PrimTypeNode* b_ty = GetPrimTypeNode(b);                                \
    TVM_FFI_CHECK(a_ty->dtype == b_ty->dtype, TypeError)                          \
        << "mismatched types. " << a_ty->dtype << " vs. " << b_ty->dtype << "\n"; \
    ffi::ObjectPtr<T> node = ffi::make_object<T>();                               \
    node->ExprNode::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});     \
    node->a = std::move(a);                                                       \
    node->b = std::move(b);                                                       \
    node->span = std::move(span);                                                 \
    data_ = std::move(node);                                                      \
  }

// Structural hooks traverse child Stmt/Expr and definition-site fields, matching the established
// visitor/mutator contract.  They deliberately skip constant annotations, strings, dtypes, source
// spans, and leaves that stay reflected for StructuralEqual/StructuralHash.  A node with no hook
// falls back to wider full reflected-field traversal, so every new node needs a hook written to
// this rule.

// Binary operators
namespace {

template <typename TNode>
TVMFFIAny BinaryVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const TNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value);
  auto a_result = visitor->VisitExpected(self->a);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(a_result);
  auto b_result = visitor->VisitExpected(self->b);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(b_result);
  return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(b_result));
}

template <typename TNode>
TVMFFIAny BinaryMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const TNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, a, mutator->MutateExpected(self->a));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, b, mutator->MutateExpected(self->b));
  if (a.same_as(self->a) && b.same_as(self->b)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  ffi::ObjectPtr<TNode> copy = ffi::make_object<TNode>(*self);
  // StructuralMap preserves node types.  Operators that reduce or widen their operands must
  // provide their own result-type logic instead of making this generic traversal infer it.
  copy->a = std::move(a);
  copy->b = std::move(b);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

template <typename TNode>
TVMFFIAny BinaryMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                   ffi::AnyView value) noexcept {
  TNode* self = const_cast<TNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, a,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->a));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, b,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->b));
  if (!a.same_as(self->a)) self->a = std::move(a);
  if (!b.same_as(self->b)) self->b = std::move(b);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
}

}  // namespace

// Ramp
namespace {

TVMFFIAny RampVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const RampNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RampNode>(value);
  auto base_result = visitor->VisitExpected(self->base);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(base_result);
  auto stride_result = visitor->VisitExpected(self->stride);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(stride_result);
  auto lanes_result = visitor->VisitExpected(self->lanes);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(lanes_result);
  return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(lanes_result));
}

TVMFFIAny RampMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const RampNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RampNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_base, mutator->MutateExpected(self->base));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_stride, mutator->MutateExpected(self->stride));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_lanes, mutator->MutateExpected(self->lanes));
  if (mapped_base.same_as(self->base) && mapped_stride.same_as(self->stride) &&
      mapped_lanes.same_as(self->lanes)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  ffi::ObjectPtr<RampNode> copy = ffi::make_object<RampNode>(*self);
  copy->base = std::move(mapped_base);
  copy->stride = std::move(mapped_stride);
  copy->lanes = std::move(mapped_lanes);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(ffi::ObjectRef(std::move(copy))));
}

TVMFFIAny RampMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  RampNode* self = const_cast<RampNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RampNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_base,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->base));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_stride,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->stride));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_lanes,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->lanes));
  if (mapped_base.same_as(self->base) && mapped_stride.same_as(self->stride) &&
      mapped_lanes.same_as(self->lanes)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  self->base = std::move(mapped_base);
  self->stride = std::move(mapped_stride);
  self->lanes = std::move(mapped_lanes);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  RampNode::RegisterReflection();
  refl::TypeAttrDef<RampNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&RampVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&RampMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&RampMaybeInplaceMutate));
}

// Broadcast
namespace {

TVMFFIAny BroadcastVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const BroadcastNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BroadcastNode>(value);
  auto value_result = visitor->VisitExpected(self->value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(value_result);
  auto lanes_result = visitor->VisitExpected(self->lanes);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(lanes_result);
  return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(lanes_result));
}

TVMFFIAny BroadcastMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const BroadcastNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BroadcastNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_value, mutator->MutateExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_lanes, mutator->MutateExpected(self->lanes));
  if (mapped_value.same_as(self->value) && mapped_lanes.same_as(self->lanes)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  ffi::ObjectPtr<BroadcastNode> copy = ffi::make_object<BroadcastNode>(*self);
  copy->value = std::move(mapped_value);
  copy->lanes = std::move(mapped_lanes);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(ffi::ObjectRef(std::move(copy))));
}

TVMFFIAny BroadcastMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                      ffi::AnyView value) noexcept {
  BroadcastNode* self = const_cast<BroadcastNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BroadcastNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_lanes,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->lanes));
  if (mapped_value.same_as(self->value) && mapped_lanes.same_as(self->lanes)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  self->value = std::move(mapped_value);
  self->lanes = std::move(mapped_lanes);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  BroadcastNode::RegisterReflection();
  refl::TypeAttrDef<BroadcastNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BroadcastVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BroadcastMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BroadcastMaybeInplaceMutate));
}

// Shuffle
namespace {

TVMFFIAny ShuffleVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const ShuffleNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShuffleNode>(value);
  auto vectors_result = visitor->VisitExpected(self->vectors);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(vectors_result);
  auto indices_result = visitor->VisitExpected(self->indices);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(indices_result);
  return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(indices_result));
}

TVMFFIAny ShuffleMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const ShuffleNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShuffleNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<PrimExpr>, mapped_vectors,
                                    mutator->MutateExpected(self->vectors));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<PrimExpr>, mapped_indices,
                                    mutator->MutateExpected(self->indices));
  if (mapped_vectors.same_as(self->vectors) && mapped_indices.same_as(self->indices)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  ffi::ObjectPtr<ShuffleNode> copy = ffi::make_object<ShuffleNode>(*self);
  copy->vectors = std::move(mapped_vectors);
  copy->indices = std::move(mapped_indices);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(ffi::ObjectRef(std::move(copy))));
}

TVMFFIAny ShuffleMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                    ffi::AnyView value) noexcept {
  ShuffleNode* self = const_cast<ShuffleNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShuffleNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<PrimExpr>, mapped_vectors,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->vectors));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<PrimExpr>, mapped_indices,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->indices));
  if (mapped_vectors.same_as(self->vectors) && mapped_indices.same_as(self->indices)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  self->vectors = std::move(mapped_vectors);
  self->indices = std::move(mapped_indices);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  ShuffleNode::RegisterReflection();
  refl::TypeAttrDef<ShuffleNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&ShuffleVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&ShuffleMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&ShuffleMaybeInplaceMutate));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  refl::GlobalDef().def("tirx.convert",
                        [](ffi::Variant<PrimExpr, ffi::Array<PrimExpr>> expr) { return expr; });
  // Note: kRepr for VarNode is registered via TVM_REGISTER_SCRIPT_AS_REPR in
  // src/script/printer/tirx/expr.cc (-> ReprPrintTIR which delegates to TVMScriptPrinter).
}

// StringImm
StringImm::StringImm(ffi::String value, Span span) {
  ffi::ObjectPtr<StringImmNode> node = ffi::make_object<StringImmNode>();
  node->ExprNode::ty = PrimType::Void();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

namespace {

TVMFFIAny StringImmVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  // value is a constant: reflected for StructuralEqual/Hash,
  // not traversed by the visitor/mutator contract.
  return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
      ffi::Expected<ffi::Optional<ffi::VisitInterrupt>>(std::nullopt));
}

TVMFFIAny StringImmMutate(ffi::StructuralMutatorObj*, ffi::AnyView value) noexcept {
  // value is a constant: reflected for StructuralEqual/Hash,
  // not traversed by the visitor/mutator contract.
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
}

TVMFFIAny StringImmMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView value) noexcept {
  // value is a constant: reflected for StructuralEqual/Hash,
  // not traversed by the visitor/mutator contract.
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  StringImmNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.StringImm",
                        [](ffi::String value, Span span) { return StringImm(value, span); });
  refl::TypeAttrDef<StringImmNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&StringImmVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&StringImmMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&StringImmMaybeInplaceMutate));
}

// Cast
Cast::Cast(PrimType value_ty, PrimExpr value, Span span) {
  TVM_FFI_ICHECK(value.defined());
  PrimType value_expr_ty = value.ty();
  TVM_FFI_ICHECK_EQ(value_ty->dtype.lanes, value_expr_ty->dtype.lanes);
  ffi::ObjectPtr<CastNode> node = ffi::make_object<CastNode>();
  node->ExprNode::ty = std::move(value_ty);
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

namespace {

TVMFFIAny CastVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const CastNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CastNode>(value);
  auto value_result = visitor->VisitExpected(self->value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(value_result);
  return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(value_result));
}

TVMFFIAny CastMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const CastNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CastNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_value, mutator->MutateExpected(self->value));
  if (mapped_value.same_as(self->value)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  ffi::ObjectPtr<CastNode> copy = ffi::make_object<CastNode>(*self);
  copy->value = std::move(mapped_value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(ffi::ObjectRef(std::move(copy))));
}

TVMFFIAny CastMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  CastNode* self = const_cast<CastNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CastNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  if (mapped_value.same_as(self->value)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  self->value = std::move(mapped_value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  CastNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.Cast", [](PrimType dtype, PrimExpr value, Span span) {
    return Cast(dtype, value, span);
  });
  refl::TypeAttrDef<CastNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&CastVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&CastMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&CastMaybeInplaceMutate));
}

// Add
TVM_DEFINE_BINOP_CONSTRUCTOR(Add);

TVM_FFI_STATIC_INIT_BLOCK() {
  AddNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.Add",
                        [](PrimExpr a, PrimExpr b, Span span) { return Add(a, b, span); });
  refl::TypeAttrDef<AddNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<AddNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<AddNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<AddNode>));
}

// Sub
TVM_DEFINE_BINOP_CONSTRUCTOR(Sub);

TVM_FFI_STATIC_INIT_BLOCK() {
  SubNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.Sub",
                        [](PrimExpr a, PrimExpr b, Span span) { return Sub(a, b, span); });
  refl::TypeAttrDef<SubNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<SubNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<SubNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<SubNode>));
}

// Mul
TVM_DEFINE_BINOP_CONSTRUCTOR(Mul);

TVM_FFI_STATIC_INIT_BLOCK() {
  MulNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.Mul",
                        [](PrimExpr a, PrimExpr b, Span span) { return Mul(a, b, span); });
  refl::TypeAttrDef<MulNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<MulNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<MulNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<MulNode>));
}

// Div
TVM_DEFINE_BINOP_CONSTRUCTOR(Div);

TVM_FFI_STATIC_INIT_BLOCK() {
  DivNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.Div",
                        [](PrimExpr a, PrimExpr b, Span span) { return Div(a, b, span); });
  refl::TypeAttrDef<DivNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<DivNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<DivNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<DivNode>));
}

// Mod
TVM_DEFINE_BINOP_CONSTRUCTOR(Mod);

TVM_FFI_STATIC_INIT_BLOCK() {
  ModNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.Mod",
                        [](PrimExpr a, PrimExpr b, Span span) { return Mod(a, b, span); });
  refl::TypeAttrDef<ModNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<ModNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<ModNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<ModNode>));
}

// FloorDiv
TVM_DEFINE_BINOP_CONSTRUCTOR(FloorDiv);

TVM_FFI_STATIC_INIT_BLOCK() {
  FloorDivNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.FloorDiv",
                        [](PrimExpr a, PrimExpr b, Span span) { return FloorDiv(a, b, span); });
  refl::TypeAttrDef<FloorDivNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<FloorDivNode>))
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(&BinaryMutate<FloorDivNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<FloorDivNode>));
}

// FloorMod
TVM_DEFINE_BINOP_CONSTRUCTOR(FloorMod);

TVM_FFI_STATIC_INIT_BLOCK() {
  FloorModNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.FloorMod",
                        [](PrimExpr a, PrimExpr b, Span span) { return FloorMod(a, b, span); });
  refl::TypeAttrDef<FloorModNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<FloorModNode>))
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(&BinaryMutate<FloorModNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<FloorModNode>));
}

// Min
TVM_DEFINE_BINOP_CONSTRUCTOR(Min);

TVM_FFI_STATIC_INIT_BLOCK() {
  MinNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.Min",
                        [](PrimExpr a, PrimExpr b, Span span) { return Min(a, b, span); });
  refl::TypeAttrDef<MinNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<MinNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<MinNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<MinNode>));
}

// Max
TVM_DEFINE_BINOP_CONSTRUCTOR(Max);

TVM_FFI_STATIC_INIT_BLOCK() {
  MaxNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.Max",
                        [](PrimExpr a, PrimExpr b, Span span) { return Max(a, b, span); });
  refl::TypeAttrDef<MaxNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<MaxNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<MaxNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<MaxNode>));
}

// EQ
TVM_DEFINE_CMPOP_CONSTRUCTOR(EQ);

TVM_FFI_STATIC_INIT_BLOCK() {
  EQNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.EQ",
                        [](PrimExpr a, PrimExpr b, Span span) { return EQ(a, b, span); });
  refl::TypeAttrDef<EQNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<EQNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<EQNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<EQNode>));
}

// NE
TVM_DEFINE_CMPOP_CONSTRUCTOR(NE);

TVM_FFI_STATIC_INIT_BLOCK() {
  NENode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.NE",
                        [](PrimExpr a, PrimExpr b, Span span) { return NE(a, b, span); });
  refl::TypeAttrDef<NENode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<NENode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<NENode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<NENode>));
}

// LT
TVM_DEFINE_CMPOP_CONSTRUCTOR(LT);

TVM_FFI_STATIC_INIT_BLOCK() {
  LTNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.LT",
                        [](PrimExpr a, PrimExpr b, Span span) { return LT(a, b, span); });
  refl::TypeAttrDef<LTNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<LTNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<LTNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<LTNode>));
}

// LE
TVM_DEFINE_CMPOP_CONSTRUCTOR(LE);

TVM_FFI_STATIC_INIT_BLOCK() {
  LENode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.LE",
                        [](PrimExpr a, PrimExpr b, Span span) { return LE(a, b, span); });
  refl::TypeAttrDef<LENode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<LENode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<LENode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<LENode>));
}

// GT
TVM_DEFINE_CMPOP_CONSTRUCTOR(GT);

TVM_FFI_STATIC_INIT_BLOCK() {
  GTNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.GT",
                        [](PrimExpr a, PrimExpr b, Span span) { return GT(a, b, span); });
  refl::TypeAttrDef<GTNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<GTNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<GTNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<GTNode>));
}

// GE
TVM_DEFINE_CMPOP_CONSTRUCTOR(GE);

TVM_FFI_STATIC_INIT_BLOCK() {
  GENode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.GE",
                        [](PrimExpr a, PrimExpr b, Span span) { return GE(a, b, span); });
  refl::TypeAttrDef<GENode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<GENode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<GENode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<GENode>));
}

// And
And::And(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined";
  PrimType a_ty = a.ty();
  PrimType b_ty = b.ty();
  TVM_FFI_ICHECK(a_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_ICHECK(b_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_CHECK(a_ty == b_ty, TypeError) << "mismatched types";

  ffi::ObjectPtr<AndNode> node = ffi::make_object<AndNode>();
  node->ExprNode::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  AndNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.And",
                        [](PrimExpr a, PrimExpr b, Span span) { return And(a, b, span); });
  refl::TypeAttrDef<AndNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<AndNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<AndNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<AndNode>));
}

// Or
Or::Or(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined";
  PrimType a_ty = a.ty();
  PrimType b_ty = b.ty();
  TVM_FFI_ICHECK(a_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_ICHECK(b_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_CHECK(a_ty == b_ty, TypeError) << "mismatched types";

  ffi::ObjectPtr<OrNode> node = ffi::make_object<OrNode>();
  node->ExprNode::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OrNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.Or",
                        [](PrimExpr a, PrimExpr b, Span span) { return Or(a, b, span); });
  refl::TypeAttrDef<OrNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<OrNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<OrNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<OrNode>));
}

// Not
Not::Not(PrimExpr a, Span span) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  PrimType a_ty = a.ty();
  TVM_FFI_ICHECK(a_ty.MatchesCode(DLDataTypeCode::kDLBool));

  ffi::ObjectPtr<NotNode> node = ffi::make_object<NotNode>();
  node->ExprNode::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});
  node->a = std::move(a);
  node->span = std::move(span);
  data_ = std::move(node);
}

namespace {

TVMFFIAny NotVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const NotNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const NotNode>(value);
  auto a_result = visitor->VisitExpected(self->a);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(a_result);
  return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(a_result));
}

TVMFFIAny NotMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const NotNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const NotNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_a, mutator->MutateExpected(self->a));
  if (mapped_a.same_as(self->a)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  ffi::ObjectPtr<NotNode> copy = ffi::make_object<NotNode>(*self);
  copy->a = std::move(mapped_a);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(ffi::ObjectRef(std::move(copy))));
}

TVMFFIAny NotMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  NotNode* self = const_cast<NotNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const NotNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_a,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->a));
  if (mapped_a.same_as(self->a)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  self->a = std::move(mapped_a);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  NotNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.Not", [](PrimExpr a, Span span) { return Not(a, span); });
  refl::TypeAttrDef<NotNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&NotVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&NotMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&NotMaybeInplaceMutate));
}

// Select
Select::Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
  TVM_FFI_CHECK(condition.defined(), ValueError) << "condition is undefined";
  TVM_FFI_CHECK(true_value.defined(), ValueError) << "true_value is undefined";
  TVM_FFI_CHECK(false_value.defined(), ValueError) << "true_value is undefined";
  PrimType condition_ty = condition.ty();
  PrimType true_ty = true_value.ty();
  PrimType false_ty = false_value.ty();
  TVM_FFI_ICHECK(condition_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_ICHECK(GetLanesOrVScaleFactor(condition_ty) == GetLanesOrVScaleFactor(true_ty) ||
                 condition_ty.IsScalar());
  TVM_FFI_CHECK(false_ty == true_ty, TypeError)
      << "mismatched types. "
      << "False type: " << false_ty->dtype << "; True type: " << true_ty->dtype;

  ffi::ObjectPtr<SelectNode> node = ffi::make_object<SelectNode>();
  node->ExprNode::ty = true_ty;
  node->condition = std::move(condition);
  node->true_value = std::move(true_value);
  node->false_value = std::move(false_value);
  node->span = std::move(span);
  data_ = std::move(node);
}

namespace {

TVMFFIAny SelectVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const SelectNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SelectNode>(value);
  auto condition_result = visitor->VisitExpected(self->condition);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(condition_result);
  auto true_value_result = visitor->VisitExpected(self->true_value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(true_value_result);
  auto false_value_result = visitor->VisitExpected(self->false_value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(false_value_result);
  return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(false_value_result));
}

TVMFFIAny SelectMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const SelectNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SelectNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_condition,
                                    mutator->MutateExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_true_value,
                                    mutator->MutateExpected(self->true_value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_false_value,
                                    mutator->MutateExpected(self->false_value));
  if (mapped_condition.same_as(self->condition) && mapped_true_value.same_as(self->true_value) &&
      mapped_false_value.same_as(self->false_value)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  ffi::ObjectPtr<SelectNode> copy = ffi::make_object<SelectNode>(*self);
  copy->condition = std::move(mapped_condition);
  copy->true_value = std::move(mapped_true_value);
  copy->false_value = std::move(mapped_false_value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(ffi::ObjectRef(std::move(copy))));
}

TVMFFIAny SelectMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                   ffi::AnyView value) noexcept {
  SelectNode* self = const_cast<SelectNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SelectNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_condition,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_true_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->true_value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_false_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->false_value));
  if (mapped_condition.same_as(self->condition) && mapped_true_value.same_as(self->true_value) &&
      mapped_false_value.same_as(self->false_value)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  self->condition = std::move(mapped_condition);
  self->true_value = std::move(mapped_true_value);
  self->false_value = std::move(mapped_false_value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  SelectNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.Select",
                        [](PrimExpr condition, PrimExpr true_value, PrimExpr false_value,
                           Span span) { return Select(condition, true_value, false_value, span); });
  refl::TypeAttrDef<SelectNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&SelectVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&SelectMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&SelectMaybeInplaceMutate));
}

// Let
Let::Let(Var var, PrimExpr value, PrimExpr body, Span span) {
  TVM_FFI_ICHECK(value.defined());
  TVM_FFI_ICHECK(body.defined());
  TVM_FFI_ICHECK(value.ty() == var->ty.as_or_throw<PrimType>());

  ffi::ObjectPtr<LetNode> node = ffi::make_object<LetNode>();
  node->ExprNode::ty = body.ty();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

namespace {

TVMFFIAny LetVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const LetNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LetNode>(value);
  auto var_result = visitor->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive,
                                               [&]() { return visitor->VisitExpected(self->var); });
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(var_result);
  auto value_result = visitor->VisitExpected(self->value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(value_result);
  auto body_result = visitor->VisitExpected(self->body);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(body_result);
  return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(body_result));
}

TVMFFIAny LetMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const LetNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LetNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      Var, mapped_var, mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MutateExpected(self->var);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_value, mutator->MutateExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_body, mutator->MutateExpected(self->body));
  if (mapped_var.same_as(self->var) && mapped_value.same_as(self->value) &&
      mapped_body.same_as(self->body)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  ffi::ObjectPtr<LetNode> copy = ffi::make_object<LetNode>(*self);
  copy->var = std::move(mapped_var);
  copy->value = std::move(mapped_value);
  copy->body = std::move(mapped_body);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(ffi::ObjectRef(std::move(copy))));
}

TVMFFIAny LetMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  LetNode* self = const_cast<LetNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LetNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      Var, mapped_var, mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->var);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_body,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->body));
  if (mapped_var.same_as(self->var) && mapped_value.same_as(self->value) &&
      mapped_body.same_as(self->body)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  self->var = std::move(mapped_var);
  self->value = std::move(mapped_value);
  self->body = std::move(mapped_body);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  LetNode::RegisterReflection();
  refl::GlobalDef().def("ir.prim.Let", [](Var var, PrimExpr value, PrimExpr body, Span span) {
    return Let(var, value, body, span);
  });
  refl::TypeAttrDef<LetNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&LetVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&LetMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&LetMaybeInplaceMutate));
}

}  // namespace prim

}  // namespace tvm
