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
 * \file tvm/ir/base_expr.h
 * \brief Base expression and primitive type nodes.
 */
#ifndef TVM_IR_BASE_EXPR_H_
#define TVM_IR_BASE_EXPR_H_

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/source_map.h>
#include <tvm/runtime/data_type.h>

#include <cstdint>

namespace tvm {

/*!
 * \brief Type is the base type of all types.
 *
 * TVM's type system contains following subclasses:
 *
 * - PrimType: type of primitive type values used in the low-level IR.
 * - FuncType: type of a function.
 * - TensorType: type of certain Tensor values in the expression.
 *
 * There are also advanced types to support generic(polymorphic types).
 * \sa Type
 */
class TypeNode : public ffi::Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    // span do not participate in structural equal and hash.
    refl::ObjectDef<TypeNode>().def_ro("span", &TypeNode::span, refl::DefaultValue(Span()),
                                       refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  static constexpr const uint32_t _type_child_slots = 14;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.Type", TypeNode, ffi::Object);
};

/*!
 * \brief Managed reference to TypeNode.
 * \sa TypeNode
 */
class Type : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Type, ffi::ObjectRef, TypeNode);
};

/*!
 * \brief Primitive data types used in the low-level IR.
 *
 * PrimType represents POD-values and handles that are
 * not automatically managed by the runtime.
 *
 * \sa PrimType
 */
class PrimTypeNode : public TypeNode {
 public:
  /*!
   * \brief The raw DLPack dtype represented by this primitive type.
   */
  DLDataType dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PrimTypeNode>().def_ro("dtype", &PrimTypeNode::dtype);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.PrimType", PrimTypeNode, TypeNode);
};

/*
 * \brief Managed reference to PrimTypeNode.
 * \sa PrimTypeNode
 */
class PrimType : public Type {
 public:
  /*!
   * \brief Construct from a raw DLPack dtype.
   * \param dtype The corresponding DLPack dtype.
   */
  TVM_DLL explicit PrimType(DLDataType dtype);

  /*!
   * \brief Construct from DLPack dtype fields.
   * \param code The DLPack dtype code.
   * \param bits The scalar bit width.
   * \param lanes The fixed lane count.
   */
  TVM_DLL PrimType(DLDataTypeCode code, int bits, int lanes = 1);

  // Fast constructors.
  TVM_DLL static PrimType Int(int bits, int lanes = 1);

  static PrimType UInt(int bits, int lanes = 1) {
    return PrimType(DLDataTypeCode::kDLUInt, bits, lanes);
  }

  TVM_DLL static PrimType Float(int bits, int lanes = 1);

  static PrimType BFloat(int bits, int lanes = 1) {
    return PrimType(DLDataTypeCode::kDLBfloat, bits, lanes);
  }

  TVM_DLL static PrimType Bool();

  static PrimType Bool(int lanes) {
    if (lanes == 1) return Bool();
    return PrimType(DLDataTypeCode::kDLBool, 8, lanes);
  }

  static PrimType Handle(int bits = 64, int lanes = 1) {
    return PrimType(DLDataTypeCode::kDLOpaqueHandle, bits, lanes);
  }

  static PrimType Void() { return PrimType(DLDataTypeCode::kDLOpaqueHandle, 0, 0); }

  TVM_DLL static PrimType ScalableVector(DLDataTypeCode code, int bits, int lanes);

  // Accessors.
  DLDataTypeCode code() const {
    return static_cast<DLDataTypeCode>(static_cast<int>(get()->dtype.code));
  }

  int32_t bits() const { return get()->dtype.bits; }

  int32_t lanes() const {
    int16_t encoded_lanes = static_cast<int16_t>(get()->dtype.lanes);
    if (encoded_lanes < 0) {
      TVM_FFI_THROW(InternalError)
          << "Can't fetch the lanes of a scalable vector at a compile time.";
    }
    return encoded_lanes;
  }

  DLDataType dtype() const { return get()->dtype; }

  // Quick checks.
  bool MatchesElementType(DLDataTypeCode code, int bits) const {
    DLDataType dtype = this->dtype();
    return dtype.code == static_cast<uint8_t>(code) && dtype.bits == bits;
  }

  bool IsVoid() const {
    DLDataType dtype = this->dtype();
    return dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLOpaqueHandle) && dtype.bits == 0 &&
           static_cast<int16_t>(dtype.lanes) == 0;
  }

  bool IsHandle() const {
    return this->code() == DLDataTypeCode::kDLOpaqueHandle && !this->IsVoid();
  }

  bool IsPredicate() const {
    DLDataType dtype = this->dtype();
    return dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLBool) ||
           (dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLUInt) && dtype.bits == 1);
  }

  bool IsScalableVector() const { return static_cast<int16_t>(get()->dtype.lanes) < -1; }

  bool IsFixedLengthVector() const { return static_cast<int16_t>(get()->dtype.lanes) > 1; }

  // Rewriters.
  TVM_DLL PrimType WithCode(DLDataTypeCode code) const;

  TVM_DLL PrimType WithBits(int bits) const;

  TVM_DLL PrimType WithLanes(int lanes) const;

  TVM_DLL int32_t VScaleFactor() const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimType, Type, PrimTypeNode);
};

/*!
 * \brief Base type of all the expressions.
 * \sa Expr
 */
class BaseExprNode : public ffi::Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  /*!
   * \brief The deduced or annotated type of the expression.
   *
   * This field is intentionally nullable because type information may
   * be populated by later analysis passes instead of expression
   * constructors.
   */
  mutable Type ty;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    // span and ty do not participate in structural equal and hash.
    refl::ObjectDef<BaseExprNode>()
        .def_ro("span", &BaseExprNode::span, refl::DefaultValue(Span()),
                refl::AttachFieldFlag::SEqHashIgnore())
        .def_ro("ty", &BaseExprNode::ty, refl::DefaultValue(Type()),
                refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  static constexpr const uint32_t _type_child_slots = 64;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.BaseExpr", BaseExprNode, ffi::Object);
};

/*!
 * \brief Managed reference to BaseExprNode.
 * \sa BaseExprNode
 */
class BaseExpr : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BaseExpr, ffi::ObjectRef, BaseExprNode);
};

namespace ffi {
template <>
inline constexpr bool use_default_type_traits_v<PrimType> = false;

template <>
struct TypeTraits<PrimType> : public ObjectRefWithFallbackTraitsBase<PrimType, runtime::DataType> {
  TVM_FFI_INLINE static PrimType ConvertFallbackValue(runtime::DataType dtype) {
    return PrimType(dtype.operator DLDataType());
  }
};
}  // namespace ffi

}  // namespace tvm

#endif  // TVM_IR_BASE_EXPR_H_
