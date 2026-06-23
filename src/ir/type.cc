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
 * \file src/ir/type.cc
 * \brief Common type system AST nodes throughout the IR.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/type.h>

#include <cstdint>
#include <unordered_map>

namespace tvm {

namespace {

bool IsFloat8Code(DLDataTypeCode code) {
  return code == DLDataTypeCode::kDLFloat8_e3m4 || code == DLDataTypeCode::kDLFloat8_e4m3 ||
         code == DLDataTypeCode::kDLFloat8_e4m3b11fnuz ||
         code == DLDataTypeCode::kDLFloat8_e4m3fn || code == DLDataTypeCode::kDLFloat8_e4m3fnuz ||
         code == DLDataTypeCode::kDLFloat8_e5m2 || code == DLDataTypeCode::kDLFloat8_e5m2fnuz ||
         code == DLDataTypeCode::kDLFloat8_e8m0fnu;
}

bool IsFloat6Code(DLDataTypeCode code) {
  return code == DLDataTypeCode::kDLFloat6_e2m3fn || code == DLDataTypeCode::kDLFloat6_e3m2fn;
}

void ValidatePrimTypeSpec(DLDataTypeCode code, int bits, int16_t encoded_lanes) {
  TVM_FFI_ICHECK_GE(bits, 0);
  TVM_FFI_ICHECK_LT(bits, 256);
  if (encoded_lanes < 0) {
    TVM_FFI_ICHECK_LT(encoded_lanes, -1) << "Invalid scalable vector vscale factor";
  }
  if (code == DLDataTypeCode::kDLBfloat) {
    TVM_FFI_ICHECK_EQ(bits, 16);
  }
  if (IsFloat8Code(code)) {
    TVM_FFI_ICHECK_EQ(bits, 8);
  }
  if (IsFloat6Code(code)) {
    TVM_FFI_ICHECK_EQ(bits, 6);
  }
  if (code == DLDataTypeCode::kDLFloat4_e2m1fn) {
    TVM_FFI_ICHECK_EQ(bits, 4);
  }
}

DLDataType MakeDLDataType(DLDataTypeCode code, int bits, int lanes, bool is_scalable = false) {
  if (is_scalable) {
    TVM_FFI_ICHECK_GT(lanes, 1) << "Invalid value for vscale factor " << lanes;
  } else {
    TVM_FFI_ICHECK_GE(lanes, 0);
  }
  TVM_FFI_ICHECK_LT(lanes, 32768);
  int16_t encoded_lanes = is_scalable ? static_cast<int16_t>(-lanes) : static_cast<int16_t>(lanes);
  ValidatePrimTypeSpec(code, bits, encoded_lanes);
  return DLDataType{static_cast<uint8_t>(code), static_cast<uint8_t>(bits),
                    static_cast<uint16_t>(encoded_lanes)};
}

ffi::ObjectPtr<PrimTypeNode> MakePrimTypeNode(DLDataType dtype) {
  ffi::ObjectPtr<PrimTypeNode> n = ffi::make_object<PrimTypeNode>();
  n->dtype = dtype;
  return n;
}

uint32_t PackDataTypeKey(DLDataType dtype) {
  return (static_cast<uint32_t>(dtype.code) << 24) | (static_cast<uint32_t>(dtype.bits) << 16) |
         static_cast<uint32_t>(dtype.lanes);
}

int64_t PrimTypeAnyHash(const ffi::Any& src) {
  return static_cast<int64_t>(PackDataTypeKey(src.cast<PrimType>()->dtype));
}

bool PrimTypeAnyEqual(const ffi::Any& lhs, const ffi::Any& rhs) {
  return lhs.cast<PrimType>()->dtype == rhs.cast<PrimType>()->dtype;
}

ffi::ObjectPtr<PrimTypeNode> GetCachedPrimTypeNode(DLDataType dtype) {
  thread_local std::unordered_map<uint32_t, ffi::ObjectPtr<PrimTypeNode>> cache;
  uint32_t key = PackDataTypeKey(dtype);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

  ffi::ObjectPtr<PrimTypeNode> node = MakePrimTypeNode(dtype);
  return cache.emplace(key, std::move(node)).first->second;
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TypeNode::RegisterReflection();
  PrimTypeNode::RegisterReflection();
  refl::TypeAttrDef<PrimTypeNode>()
      .attr(refl::type_attr::kAnyHash, reinterpret_cast<void*>(&PrimTypeAnyHash))
      .attr(refl::type_attr::kAnyEqual, reinterpret_cast<void*>(&PrimTypeAnyEqual));
  PointerTypeNode::RegisterReflection();
  TupleTypeNode::RegisterReflection();
  FuncTypeNode::RegisterReflection();
  TensorMapTypeNode::RegisterReflection();
}

PrimType::PrimType(DLDataType dtype) { data_ = GetCachedPrimTypeNode(dtype); }

PrimType::PrimType(DLDataTypeCode code, int bits, int lanes)
    : PrimType(MakeDLDataType(code, bits, lanes)) {}

PrimType PrimType::Int(int bits, int lanes) {
  if (lanes == 1) {
    if (bits == 32) {
      thread_local PrimType i32_ty(MakeDLDataType(DLDataTypeCode::kDLInt, 32, 1));
      return i32_ty;
    }
    if (bits == 64) {
      thread_local PrimType i64_ty(MakeDLDataType(DLDataTypeCode::kDLInt, 64, 1));
      return i64_ty;
    }
  }
  return PrimType(MakeDLDataType(DLDataTypeCode::kDLInt, bits, lanes));
}

PrimType PrimType::Float(int bits, int lanes) {
  if (bits == 32 && lanes == 1) {
    thread_local PrimType f32_ty(MakeDLDataType(DLDataTypeCode::kDLFloat, 32, 1));
    return f32_ty;
  }
  return PrimType(MakeDLDataType(DLDataTypeCode::kDLFloat, bits, lanes));
}

PrimType PrimType::Bool() {
  thread_local PrimType bool_ty(MakeDLDataType(DLDataTypeCode::kDLBool, 8, 1));
  return bool_ty;
}

PrimType PrimType::ScalableVector(DLDataTypeCode code, int bits, int lanes) {
  return PrimType(MakeDLDataType(code, bits, lanes, /*is_scalable=*/true));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.PrimType", [](DLDataType dtype) { return PrimType(dtype); });
}

PointerType::PointerType(Type element_type, ffi::String storage_scope) {
  ffi::ObjectPtr<PointerTypeNode> n = ffi::make_object<PointerTypeNode>();
  if (storage_scope.empty()) {
    n->storage_scope = "global";
  } else {
    n->storage_scope = std::move(storage_scope);
  }
  n->element_type = std::move(element_type);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.PointerType", [](Type element_type, ffi::String storage_scope = "") {
    return PointerType(element_type, storage_scope);
  });
}

FuncType::FuncType(tvm::ffi::Array<Type> arg_types, Type ret_type, Span span) {
  ffi::ObjectPtr<FuncTypeNode> n = ffi::make_object<FuncTypeNode>();
  n->arg_types = std::move(arg_types);
  n->ret_type = std::move(ret_type);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.FuncType", [](tvm::ffi::Array<Type> arg_types, Type ret_type) {
    return FuncType(arg_types, ret_type);
  });
}

TupleType::TupleType(ffi::Array<Type> fields, Span span) {
  ffi::ObjectPtr<TupleTypeNode> n = ffi::make_object<TupleTypeNode>();
  n->fields = std::move(fields);
  n->span = std::move(span);
  data_ = std::move(n);
}

TupleType TupleType::Empty() { return TupleType(ffi::Array<Type>()); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ir.TupleType",
           [](ffi::Array<Type> fields, Span span) { return TupleType(fields, span); })
      .def("ir.TensorMapType", [](Span span) { return TensorMapType(span); });
}

TensorMapType::TensorMapType(Span span) {
  ffi::ObjectPtr<TensorMapTypeNode> n = ffi::make_object<TensorMapTypeNode>();
  n->span = std::move(span);
  data_ = std::move(n);
}

}  // namespace tvm
