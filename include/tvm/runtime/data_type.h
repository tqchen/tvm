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
/*
 * \file tvm/runtime/data_type.h
 * \brief Primitive runtime data type helpers.
 */
#ifndef TVM_RUNTIME_DATA_TYPE_H_
#define TVM_RUNTIME_DATA_TYPE_H_

#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/runtime/base.h>

#include <cstdint>
#include <type_traits>

namespace tvm {
namespace runtime {

inline DLDataType MakeDType(int code, int bits, int lanes, bool is_scalable = false) {
  DLDataType dtype;
  dtype.code = static_cast<uint8_t>(code);
  dtype.bits = static_cast<uint8_t>(bits);
  if (is_scalable) {
    TVM_FFI_ICHECK(lanes > 1) << "Invalid value for vscale factor" << lanes;
  }
  dtype.lanes = is_scalable ? static_cast<uint16_t>(-lanes) : static_cast<uint16_t>(lanes);
  if (code == kDLBfloat) {
    TVM_FFI_ICHECK_EQ(bits, 16);
  }
  if (code == kDLFloat8_e3m4 || code == kDLFloat8_e4m3 || code == kDLFloat8_e4m3b11fnuz ||
      code == kDLFloat8_e4m3fn || code == kDLFloat8_e4m3fnuz || code == kDLFloat8_e5m2 ||
      code == kDLFloat8_e5m2fnuz || code == kDLFloat8_e8m0fnu) {
    TVM_FFI_ICHECK_EQ(bits, 8);
  }
  if (code == kDLFloat6_e2m3fn || code == kDLFloat6_e3m2fn) {
    TVM_FFI_ICHECK_EQ(bits, 6);
  }
  if (code == kDLFloat4_e2m1fn) {
    TVM_FFI_ICHECK_EQ(bits, 4);
  }
  return dtype;
}

inline DLDataType IntDType(int bits, int lanes = 1) { return MakeDType(kDLInt, bits, lanes); }

inline DLDataType UIntDType(int bits, int lanes = 1, bool is_scalable = false) {
  return MakeDType(kDLUInt, bits, lanes, is_scalable);
}

inline DLDataType FloatDType(int bits, int lanes = 1) { return MakeDType(kDLFloat, bits, lanes); }

inline DLDataType BFloatDType(int bits, int lanes = 1) {
  return MakeDType(kDLBfloat, bits, lanes);
}

inline DLDataType Float8E3M4DType(int lanes = 1) {
  return MakeDType(kDLFloat8_e3m4, 8, lanes);
}

inline DLDataType Float8E4M3DType(int lanes = 1) {
  return MakeDType(kDLFloat8_e4m3, 8, lanes);
}

inline DLDataType Float8E4M3B11FNUZDType(int lanes = 1) {
  return MakeDType(kDLFloat8_e4m3b11fnuz, 8, lanes);
}

inline DLDataType Float8E4M3FNDType(int lanes = 1) {
  return MakeDType(kDLFloat8_e4m3fn, 8, lanes);
}

inline DLDataType Float8E4M3FNUZDType(int lanes = 1) {
  return MakeDType(kDLFloat8_e4m3fnuz, 8, lanes);
}

inline DLDataType Float8E5M2DType(int lanes = 1) {
  return MakeDType(kDLFloat8_e5m2, 8, lanes);
}

inline DLDataType Float8E5M2FNUZDType(int lanes = 1) {
  return MakeDType(kDLFloat8_e5m2fnuz, 8, lanes);
}

inline DLDataType Float8E8M0FNUDType(int lanes = 1) {
  return MakeDType(kDLFloat8_e8m0fnu, 8, lanes);
}

inline DLDataType Float6E2M3FNDType(int lanes = 1) {
  return MakeDType(kDLFloat6_e2m3fn, 6, lanes);
}

inline DLDataType Float6E3M2FNDType(int lanes = 1) {
  return MakeDType(kDLFloat6_e3m2fn, 6, lanes);
}

inline DLDataType Float4E2M1FNDType(int lanes = 1) {
  return MakeDType(kDLFloat4_e2m1fn, 4, lanes);
}

inline DLDataType BoolDType(int lanes = 1, bool is_scalable = false) {
  return MakeDType(kDLBool, 8, lanes, is_scalable);
}

inline DLDataType HandleDType(int bits = 64, int lanes = 1) {
  return MakeDType(kDLOpaqueHandle, bits, lanes);
}

inline DLDataType VoidDType() { return MakeDType(kDLOpaqueHandle, 0, 0); }

inline DLDataType ShapeIndexDType() {
  if (std::is_signed<ffi::Shape::index_type>::value) {
    return IntDType(sizeof(ffi::Shape::index_type) * 8);
  } else {
    return UIntDType(sizeof(ffi::Shape::index_type) * 8);
  }
}

inline int DTypeCode(DLDataType dtype) { return static_cast<int>(dtype.code); }

inline int DTypeBits(DLDataType dtype) { return static_cast<int>(dtype.bits); }

inline int DTypeBytes(DLDataType dtype) { return (DTypeBits(dtype) + 7) / 8; }

inline int DTypeLanes(DLDataType dtype) {
  int lanes = static_cast<int16_t>(dtype.lanes);
  if (lanes < 0) {
    TVM_FFI_THROW(InternalError)
        << "Can't fetch the lanes of a scalable vector at a compile time.";
  }
  return lanes;
}

inline int DTypeVScaleFactor(DLDataType dtype) {
  int lanes = static_cast<int16_t>(dtype.lanes);
  if (lanes >= -1) {
    TVM_FFI_THROW(InternalError) << "A fixed length vector doesn't have a vscale factor.";
  }
  return -lanes;
}

inline bool IsScalableVectorDType(DLDataType dtype) {
  return static_cast<int16_t>(dtype.lanes) < -1;
}

inline int DTypeLanesOrVScaleFactor(DLDataType dtype) {
  return IsScalableVectorDType(dtype) ? DTypeVScaleFactor(dtype) : DTypeLanes(dtype);
}

inline bool IsFixedLengthVectorDType(DLDataType dtype) {
  return static_cast<int16_t>(dtype.lanes) > 1;
}

inline bool IsVectorDType(DLDataType dtype) { return DTypeLanes(dtype) > 1; }

inline bool IsScalableOrFixedLengthVectorDType(DLDataType dtype) {
  int lanes = static_cast<int16_t>(dtype.lanes);
  return lanes < -1 || lanes > 1;
}

inline bool IsScalarDType(DLDataType dtype) {
  return !IsScalableVectorDType(dtype) && DTypeLanes(dtype) == 1;
}

inline bool IsVoidDType(DLDataType dtype) {
  return dtype.code == kDLOpaqueHandle && dtype.bits == 0 &&
         static_cast<int16_t>(dtype.lanes) == 0;
}

inline bool IsBoolDType(DLDataType dtype) { return dtype.code == kDLBool; }

inline bool IsUIntDType(DLDataType dtype) { return dtype.code == kDLUInt; }

inline bool IsIntDType(DLDataType dtype) { return dtype.code == kDLInt; }

inline bool IsFloatDType(DLDataType dtype) { return dtype.code == kDLFloat; }

inline bool IsBFloatDType(DLDataType dtype) { return dtype.code == kDLBfloat; }

inline bool IsHandleDType(DLDataType dtype) {
  return dtype.code == kDLOpaqueHandle && !IsVoidDType(dtype);
}

inline bool IsPredicateDType(DLDataType dtype) {
  return IsBoolDType(dtype) || (IsUIntDType(dtype) && dtype.bits == 1);
}

inline bool IsVectorBoolDType(DLDataType dtype) {
  return IsScalableOrFixedLengthVectorDType(dtype) && IsBoolDType(dtype);
}

inline bool IsFloat8DType(DLDataType dtype) {
  return dtype.bits == 8 &&
         (dtype.code == kDLFloat8_e3m4 || dtype.code == kDLFloat8_e4m3 ||
          dtype.code == kDLFloat8_e4m3b11fnuz || dtype.code == kDLFloat8_e4m3fn ||
          dtype.code == kDLFloat8_e4m3fnuz || dtype.code == kDLFloat8_e5m2 ||
          dtype.code == kDLFloat8_e5m2fnuz || dtype.code == kDLFloat8_e8m0fnu);
}

inline bool IsFloat6DType(DLDataType dtype) {
  return dtype.bits == 6 &&
         (dtype.code == kDLFloat6_e2m3fn || dtype.code == kDLFloat6_e3m2fn);
}

inline bool IsFloat4DType(DLDataType dtype) {
  return dtype.bits == 4 && dtype.code == kDLFloat4_e2m1fn;
}

inline bool IsFloat16DType(DLDataType dtype) {
  return IsFloatDType(dtype) && dtype.bits == 16;
}

inline bool IsBFloat16DType(DLDataType dtype) {
  return dtype.code == kDLBfloat && dtype.bits == 16;
}

inline DLDataType DTypeWithLanes(DLDataType dtype, int lanes) {
  dtype.lanes = static_cast<uint16_t>(lanes);
  return dtype;
}

inline DLDataType DTypeWithScalableVScaleFactor(DLDataType dtype, int vscale_factor) {
  TVM_FFI_ICHECK(vscale_factor > 1) << "Invalid value for vscale factor" << vscale_factor;
  dtype.lanes = static_cast<uint16_t>(-vscale_factor);
  return dtype;
}

inline DLDataType DTypeWithBits(DLDataType dtype, int bits) {
  dtype.bits = static_cast<uint8_t>(bits);
  return dtype;
}

inline DLDataType DTypeElementOf(DLDataType dtype) { return DTypeWithLanes(dtype, 1); }

/*!
 * \brief Get the number of bytes needed in a vector.
 * \param dtype The data type.
 * \return Number of bytes needed.
 */
inline int GetVectorBytes(DLDataType dtype) {
  int lanes = static_cast<int16_t>(dtype.lanes);
  if (lanes < 0) {
    TVM_FFI_THROW(InternalError)
        << "Can't fetch the bytes of a scalable vector at a compile time.";
  }
  int data_bits = dtype.bits * lanes;
  auto type_match = [](DLDataType type, int code, int bits, int lanes = 1) {
    return type.code == code && type.bits == bits && type.lanes == lanes;
  };
  // allow bool to exist
  if (type_match(dtype, kDLBool, 8) || type_match(dtype, kDLInt, 4) ||
      type_match(dtype, kDLUInt, 4) || type_match(dtype, kDLInt, 1) ||
      type_match(dtype, kDLFloat4_e2m1fn, 4) ||
      type_match(dtype, kDLFloat6_e2m3fn, 6) ||
      type_match(dtype, kDLFloat6_e3m2fn, 6)) {
    return 1;
  }
  TVM_FFI_ICHECK_EQ(data_bits % 8, 0U) << "Need to load/store by multiple of bytes";
  return data_bits / 8;
}

/*!
 * \brief Check whether type matches the given spec.
 * \param t The type
 * \param code The type code.
 * \param bits The number of bits to be matched.
 * \param lanes The number of lanes in the type.
 */
inline bool TypeMatch(DLDataType t, int code, int bits, int lanes = 1) {
  return t.code == code && t.bits == bits && t.lanes == lanes;
}

/*!
 * \brief Check whether two types are equal.
 * \param lhs The left operand.
 * \param rhs The right operand.
 */
inline bool TypeEqual(DLDataType lhs, DLDataType rhs) { return lhs == rhs; }

}  // namespace runtime
}  // namespace tvm

namespace std {
template <>
struct hash<DLDataType> {
  inline int cantor_pairing_function(int a, int b) const { return (a + b) * (a + b + 1) / 2 + b; }
  std::size_t operator()(DLDataType const& dtype) const {
    int a = dtype.code;
    int b = dtype.bits;
    int c = static_cast<int16_t>(dtype.lanes);
    int d = cantor_pairing_function(a, b);
    return cantor_pairing_function(c, d);
  }
};
}  // namespace std

#endif  // TVM_RUNTIME_DATA_TYPE_H_
