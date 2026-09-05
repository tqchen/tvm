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
 * \file iter_var.cc
 * \brief Iteration-variable definitions.
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/var.h>

#include <utility>

namespace tvm {
namespace tirx {

namespace refl = tvm::ffi::reflection;

// Structural hooks follow the established visitor/mutator field set.  Source spans and constant
// metadata stay reflected for StructuralEqual/StructuralHash but are skipped by traversal.  A
// node without a hook falls back to wider reflected-field traversal, so every new node needs a
// hook written to this rule.

// IterVar
IterVar::IterVar(Range dom, PrimVar var, IterVarType t, ffi::String thread_tag, Span span) {
  ffi::ObjectPtr<IterVarNode> n = ffi::make_object<IterVarNode>();
  if (dom.defined() && dom->extent.defined()) {
    PrimType extent_ty = dom->extent.ty();
    PrimType var_ty = var.ty();
    TVM_FFI_ICHECK(extent_ty.code() == DLDataTypeCode::kDLInt)
        << "The dtype of the domain of an IterVar must be an integer type. However, the domain's "
           "dtype is "
        << extent_ty->dtype;
    TVM_FFI_ICHECK(extent_ty == var_ty)
        << "The dtype of the extent of an IterVar (" << extent_ty->dtype
        << ") must match its associated Var's dtype (" << var_ty->dtype << ")";
  }
  n->dom = dom;
  n->var = var;
  n->iter_type = t;
  n->thread_tag = thread_tag;
  n->span = std::move(span);
  data_ = std::move(n);
}

static TVMFFIAny IterVarVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: iter_type, thread_tag
  const IterVarNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IterVarNode>(value);
  auto dom_result = visitor->VisitExpected(self->dom);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(dom_result);
  auto var_result = visitor->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive,
                                               [&]() { return visitor->VisitExpected(self->var); });
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(var_result);
  return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(var_result));
}

static TVMFFIAny IterVarMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: iter_type, thread_tag
  const IterVarNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IterVarNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Range, mapped_dom, mutator->MutateExpected(self->dom));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      PrimVar, mapped_var, mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MutateExpected(self->var);
      }));
  if (mapped_dom.same_as(self->dom) && mapped_var.same_as(self->var)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  ffi::ObjectPtr<IterVarNode> copy = ffi::make_object<IterVarNode>(*self);
  copy->dom = std::move(mapped_dom);
  copy->var = std::move(mapped_var);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(ffi::ObjectRef(std::move(copy))));
}

static TVMFFIAny IterVarMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                           ffi::AnyView value) noexcept {
  // skips: iter_type, thread_tag
  IterVarNode* self = const_cast<IterVarNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IterVarNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Range, mapped_dom,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->dom));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      PrimVar, mapped_var, mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->var);
      }));
  if (mapped_dom.same_as(self->dom) && mapped_var.same_as(self->var)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
  }
  self->dom = std::move(mapped_dom);
  self->var = std::move(mapped_var);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IterVarNode::RegisterReflection();
  refl::GlobalDef().def(
      "tirx.IterVar", [](Range dom, PrimVar var, int iter_type, ffi::String thread_tag, Span span) {
        return IterVar(dom, var, static_cast<IterVarType>(iter_type), thread_tag, span);
      });
  refl::TypeAttrDef<IterVarNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&IterVarVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&IterVarMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&IterVarMaybeInplaceMutate));
}

}  // namespace tirx
}  // namespace tvm
