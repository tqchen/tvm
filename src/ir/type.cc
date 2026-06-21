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

ffi::ObjectPtr<PrimTypeNode> MakePrimTypeNode(runtime::DataType dtype, Span span = Span()) {
  ffi::ObjectPtr<PrimTypeNode> n = ffi::make_object<PrimTypeNode>();
  n->dtype = dtype;
  n->span = std::move(span);
  return n;
}

uint32_t PackDataTypeKey(runtime::DataType dtype) {
  DLDataType raw = dtype;
  return (static_cast<uint32_t>(raw.code) << 24) | (static_cast<uint32_t>(raw.bits) << 16) |
         static_cast<uint32_t>(raw.lanes);
}

ffi::ObjectPtr<PrimTypeNode> GetCachedPrimTypeNode(runtime::DataType dtype) {
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
  TypeNode::RegisterReflection();
  PrimTypeNode::RegisterReflection();
  PointerTypeNode::RegisterReflection();
  TupleTypeNode::RegisterReflection();
  FuncTypeNode::RegisterReflection();
  TensorMapTypeNode::RegisterReflection();
}

PrimType::PrimType(runtime::DataType dtype, Span span) {
  if (!span.defined()) {
    if (auto cached = GetCachedPrimTypeNode(dtype)) {
      data_ = std::move(cached);
      return;
    }
  }
  data_ = MakePrimTypeNode(dtype, std::move(span));
}

PrimType PrimType::Int(int bits, int lanes) { return PrimType(DataType::Int(bits, lanes)); }

PrimType PrimType::UInt(int bits, int lanes, bool is_scalable) {
  return PrimType(DataType::UInt(bits, lanes, is_scalable));
}

PrimType PrimType::Float(int bits, int lanes) { return PrimType(DataType::Float(bits, lanes)); }

PrimType PrimType::BFloat(int bits, int lanes) { return PrimType(DataType::BFloat(bits, lanes)); }

PrimType PrimType::Bool(int lanes, bool is_scalable) {
  return PrimType(DataType::Bool(lanes, is_scalable));
}

PrimType PrimType::Handle(int bits, int lanes) { return PrimType(DataType::Handle(bits, lanes)); }

PrimType PrimType::Void() { return PrimType(DataType::Void()); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.PrimType", [](runtime::DataType dtype) { return PrimType(dtype); });
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
