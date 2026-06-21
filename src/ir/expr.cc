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
 * \file src/ir/expr.cc
 * \brief The expression AST nodes for the common IR infra.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/type.h>
#include <tvm/te/tensor.h>
#include <tvm/tirx/expr.h>

#include <cmath>

#include "../support/limits.h"

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  BaseExprNode::RegisterReflection();
  PrimExprNode::RegisterReflection();
  RelaxExprNode::RegisterReflection();
  BaseFuncNode::RegisterReflection();
  GlobalVarNode::RegisterReflection();
  IntImmNode::RegisterReflection();
  FloatImmNode::RegisterReflection();
  RangeNode::RegisterReflection();
}

PrimExpr::PrimExpr(int32_t value) : PrimExpr(IntImm::Int32(value)) {}

PrimExpr::PrimExpr(float value) : PrimExpr(FloatImm(DataType::Float(32), value)) {}

PrimExpr PrimExpr::ConvertFallbackValue(ffi::String value) { return tirx::StringImm(value); }

DataType PrimExprNode::dtype() const {
  if (const auto* prim_type = this->ty.as<PrimTypeNode>()) {
    return prim_type->dtype;
  }
  if (this->ty.as<PointerTypeNode>()) {
    return DataType::Handle();
  }
  if (this->ty.defined() && IsVoidType(this->ty)) {
    return DataType::Void();
  }
  TVM_FFI_ICHECK(this->ty.defined()) << "PrimExpr is missing its type";
  TVM_FFI_THROW(InternalError) << "Cannot derive runtime dtype from PrimExpr type " << this->ty;
}

void PrimExprNode::SetDType(DataType dtype) { this->SetType(PrimType(dtype)); }

void PrimExprNode::SetType(Type ty) {
  if (const auto* prim_type = ty.as<PrimTypeNode>()) {
    this->dtype_ = prim_type->dtype;
  } else if (ty.as<PointerTypeNode>()) {
    this->dtype_ = DataType::Handle();
  } else if (IsVoidType(ty)) {
    this->dtype_ = DataType::Void();
  } else {
    TVM_FFI_THROW(InternalError) << "Cannot derive runtime dtype from PrimExpr type " << ty;
  }
  this->ty = std::move(ty);
}

IntImm::IntImm(DataType dtype, int64_t value, Span span) : IntImm(PrimType(dtype), value, span) {}

IntImm::IntImm(PrimType dtype, int64_t value, Span span) {
  DataType runtime_dtype = dtype->dtype;
  TVM_FFI_CHECK(runtime_dtype.is_scalar(), ValueError)
      << "IntImm can only take scalar, but " << runtime_dtype << " was supplied.";
  TVM_FFI_CHECK(runtime_dtype.is_int() || runtime_dtype.is_uint() || runtime_dtype.is_bool(),
                ValueError)
      << "IntImm supports only int or uint or bool type, but " << runtime_dtype
      << " was supplied.";
  if (runtime_dtype.is_uint()) {
    TVM_FFI_CHECK_GE(value, 0U, ValueError)
        << "Literal value " << value << " is negative for unsigned integer type " << runtime_dtype;
    if (runtime_dtype.bits() < 64) {
      TVM_FFI_CHECK_LT(value, 1LL << runtime_dtype.bits(), ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    }
  } else if (runtime_dtype.bits() == 1 || runtime_dtype.is_bool()) {
    // int(1)
    TVM_FFI_CHECK(value == 0 || value == 1, ValueError)
        << value << " exceeds range of " << runtime_dtype;
  } else if (runtime_dtype.bits() < 64) {
    TVM_FFI_CHECK_GE(value, -(1LL << (runtime_dtype.bits() - 1)), ValueError)
        << "Literal value " << value << " exceeds minimum of " << runtime_dtype;
    TVM_FFI_CHECK_LT(value, 1LL << (runtime_dtype.bits() - 1), ValueError)
        << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
  }
  ffi::ObjectPtr<IntImmNode> node = ffi::make_object<IntImmNode>();
  node->SetType(std::move(dtype));
  node->value = value;
  node->span = span;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.IntImm", [](DataType dtype, int64_t value, Span span) {
    return IntImm(dtype, value, span);
  });
}

FloatImm::FloatImm(DataType dtype, double value, Span span)
    : FloatImm(PrimType(dtype), value, span) {}

FloatImm::FloatImm(PrimType dtype, double value, Span span) {
  DataType runtime_dtype = dtype->dtype;
  TVM_FFI_CHECK_EQ(runtime_dtype.lanes(), 1, ValueError) << "FloatImm can only take scalar.";

  TVM_FFI_CHECK(runtime_dtype.is_float() || runtime_dtype.is_bfloat16() ||
                    runtime_dtype.is_float8() || runtime_dtype.is_float6() ||
                    runtime_dtype.is_float4() || runtime_dtype.code() >= DataType::kCustomBegin,
                ValueError)
      << "FloatImm supports only float, but " << runtime_dtype << " was supplied.";

  // check range for float32 and float16 since they have specified range.
  if (!std::isinf(value) && !std::isnan(value)) {
    if (runtime_dtype.bits() == 32) {
      TVM_FFI_CHECK_GE(value, std::numeric_limits<float>::lowest(), ValueError)
          << "Literal value " << value << " exceeds minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, std::numeric_limits<float>::max(), ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    } else if (runtime_dtype.is_float16()) {
      TVM_FFI_CHECK_GE(value, -support::kMaxFloat16, ValueError)
          << "Literal value " << value << " exceeds minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, support::kMaxFloat16, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    } else if (runtime_dtype.is_bfloat16()) {
      TVM_FFI_CHECK_GE(value, -support::kMaxBFloat16, ValueError)
          << "Literal value " << value << " exceeds minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, support::kMaxBFloat16, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    } else if (runtime_dtype.is_float8_e3m4() || runtime_dtype.is_float8_e4m3() ||
               runtime_dtype.is_float8_e4m3b11fnuz() || runtime_dtype.is_float8_e4m3fn() ||
               runtime_dtype.is_float8_e4m3fnuz() || runtime_dtype.is_float8_e5m2() ||
               runtime_dtype.is_float8_e5m2fnuz() || runtime_dtype.is_float8_e8m0fnu()) {
      double bound = 0.0;
      bool nonneg = false;

      switch (runtime_dtype.code()) {
        case DataType::TypeCode::kFloat8_e3m4:
          bound = support::kMaxE3M4;
          break;
        case DataType::TypeCode::kFloat8_e4m3:
          bound = support::kMaxE4M3;
          break;
        case DataType::TypeCode::kFloat8_e4m3b11fnuz:
          bound = support::kMaxE4M3B11FNUZ;
          nonneg = true;
          break;
        case DataType::TypeCode::kFloat8_e4m3fn:
          bound = support::kMaxE4M3FN;
          break;
        case DataType::TypeCode::kFloat8_e4m3fnuz:
          bound = support::kMaxE4M3FNUZ;
          nonneg = true;
          break;
        case DataType::TypeCode::kFloat8_e5m2:
          bound = support::kMaxE5M2;
          break;
        case DataType::TypeCode::kFloat8_e5m2fnuz:
          bound = support::kMaxE5M2FNUZ;
          nonneg = true;
          break;
        case DataType::TypeCode::kFloat8_e8m0fnu:
          bound = support::kMaxE8M0FNU;
          nonneg = true;
          break;
        default:
          TVM_FFI_THROW(InternalError) << "Unhandled float8 type: " << runtime_dtype;
      }

      if (nonneg) {
        TVM_FFI_CHECK_GE(value, 0, ValueError)
            << "Literal value " << value << " below zero for unsigned " << runtime_dtype;
      } else {
        TVM_FFI_CHECK_GE(value, -bound, ValueError)
            << "Literal value " << value << " below minimum of " << runtime_dtype;
      }
      TVM_FFI_CHECK_LE(value, bound, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;

    } else if (runtime_dtype.is_float6_e2m3fn() || runtime_dtype.is_float6_e3m2fn()) {
      double bound = (runtime_dtype.code() == DataType::TypeCode::kFloat6_e2m3fn)
                         ? support::kMaxE2M3FN
                         : support::kMaxE3M2FN;
      TVM_FFI_CHECK_GE(value, -bound, ValueError)
          << "Literal value " << value << " below minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, bound, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;

    } else if (runtime_dtype.is_float4_e2m1fn()) {
      double bound = support::kMaxE2M1FN;
      TVM_FFI_CHECK_GE(value, -bound, ValueError)
          << "Literal value " << value << " below minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, bound, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    }
  }
  ffi::ObjectPtr<FloatImmNode> node = ffi::make_object<FloatImmNode>();
  node->SetType(std::move(dtype));
  node->value = value;
  node->span = span;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.FloatImm", [](DataType dtype, double value, Span span) {
    return FloatImm(dtype, value, span);
  });
}

Range::Range(PrimExpr begin, PrimExpr end, Span span)
    : Range(ffi::make_object<RangeNode>(begin, tirx::is_zero(begin) ? end : (end - begin), span)) {}

Range Range::FromMinExtent(PrimExpr min, PrimExpr extent, Span span) {
  return Range(ffi::make_object<RangeNode>(min, extent, span));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ir.Range_from_min_extent", Range::FromMinExtent)
      .def("ir.Range", [](PrimExpr begin, ffi::Optional<PrimExpr> end, Span span) -> Range {
        if (end.defined()) {
          return Range(begin, end.value(), span);
        } else {
          return Range(IntImm(begin.dtype(), 0), begin, span);
        }
      });
}

GlobalVar::GlobalVar(ffi::String name_hint, Span span) {
  ffi::ObjectPtr<GlobalVarNode> n = ffi::make_object<GlobalVarNode>();
  n->name_hint = std::move(name_hint);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ir.GlobalVar", [](ffi::String name) { return GlobalVar(name); })
      .def("ir.DebugPrint", [](ffi::ObjectRef ref) {
        std::stringstream ss;
        ss << ref;
        return ss.str();
      });
  // Note: kRepr for GlobalVarNode is registered in script/printer/ir/ir.cc
  // via TVM_REGISTER_SCRIPT_AS_REPR(GlobalVarNode, ReprPrintIR).
}

}  // namespace tvm
