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
   * \brief Constructor
   * \param dtype The corresponding DLPack dtype.
   * \param span The span
   */
  TVM_DLL explicit PrimType(DLDataType dtype, Span span = Span());

  /*! \brief Construct an int type. */
  TVM_DLL static PrimType Int(int bits, int lanes = 1);

  /*! \brief Construct an uint type. */
  TVM_DLL static PrimType UInt(int bits, int lanes = 1);

  /*! \brief Construct a float type. */
  TVM_DLL static PrimType Float(int bits, int lanes = 1);

  /*! \brief Construct a bfloat type. */
  TVM_DLL static PrimType BFloat(int bits, int lanes = 1);

  /*! \brief Construct float8 e3m4 type. */
  TVM_DLL static PrimType Float8E3M4(int lanes = 1);

  /*! \brief Construct float8 e4m3 type. */
  TVM_DLL static PrimType Float8E4M3(int lanes = 1);

  /*! \brief Construct float8 e4m3b11fnuz type. */
  TVM_DLL static PrimType Float8E4M3B11FNUZ(int lanes = 1);

  /*! \brief Construct float8 e4m3fn type. */
  TVM_DLL static PrimType Float8E4M3FN(int lanes = 1);

  /*! \brief Construct float8 e4m3fnuz type. */
  TVM_DLL static PrimType Float8E4M3FNUZ(int lanes = 1);

  /*! \brief Construct float8 e5m2 type. */
  TVM_DLL static PrimType Float8E5M2(int lanes = 1);

  /*! \brief Construct float8 e5m2fnuz type. */
  TVM_DLL static PrimType Float8E5M2FNUZ(int lanes = 1);

  /*! \brief Construct float8 e8m0fnu type. */
  TVM_DLL static PrimType Float8E8M0FNU(int lanes = 1);

  /*! \brief Construct float6 e2m3fn type. */
  TVM_DLL static PrimType Float6E2M3FN(int lanes = 1);

  /*! \brief Construct float6 e3m2fn type. */
  TVM_DLL static PrimType Float6E3M2FN(int lanes = 1);

  /*! \brief Construct float4 e2m1fn type. */
  TVM_DLL static PrimType Float4E2M1FN(int lanes = 1);

  /*! \brief Construct a bool type. */
  TVM_DLL static PrimType Bool();

  /*! \brief Construct a fixed-length bool vector type. */
  TVM_DLL static PrimType Bool(int lanes);

  /*! \brief Construct a handle type. */
  TVM_DLL static PrimType Handle(int bits = 64, int lanes = 1);

  /*! \brief Construct a void type. */
  TVM_DLL static PrimType Void();

  /*! \brief Construct a scalable vector type. */
  TVM_DLL static PrimType ScalableVector(DLDataTypeCode code, int bits, int lanes);

  /*! \brief Check whether the scalar element type matches code and bits. */
  TVM_DLL bool MatchesElementType(DLDataTypeCode code, int bits) const;

  /*! \brief Whether the type is void. */
  TVM_DLL bool IsVoid() const;

  /*! \brief Whether the type is an opaque handle. */
  TVM_DLL bool IsHandle() const;

  /*! \brief Whether the type can be used as a predicate. */
  TVM_DLL bool IsPredicate() const;

  /*! \brief Whether the type is a scalable vector. */
  TVM_DLL bool IsScalableVector() const;

  /*! \brief Whether the type is a fixed-length vector. */
  TVM_DLL bool IsFixedLengthVector() const;

  /*! \brief Return a type with the same code/lanes and new bit width. */
  TVM_DLL PrimType WithBits(int bits) const;

  /*! \brief Return a fixed-length type with the same code/bits and new lanes. */
  TVM_DLL PrimType WithLanes(int lanes) const;

  /*! \brief Return the DLPack code. */
  TVM_DLL DLDataTypeCode code() const;

  /*! \brief Return the scalar bit width. */
  TVM_DLL int32_t bits() const;

  /*! \brief Return fixed lanes, or fail for scalable vectors. */
  TVM_DLL int32_t lanes() const;

  /*! \brief Return the vscale multiplier for scalable vectors. */
  TVM_DLL int32_t VScaleFactor() const;

  /*! \brief Return the raw DLPack dtype. */
  TVM_DLL DLDataType dtype() const;

  /*! \brief Convert to the raw DLPack dtype. */
  TVM_DLL operator DLDataType() const;

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
    return PrimType(static_cast<DLDataType>(dtype));
  }
};
}  // namespace ffi

}  // namespace tvm

#endif  // TVM_IR_BASE_EXPR_H_
