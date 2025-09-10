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

 #ifndef TVM_FFI_CYTHON_HELPERS_H_
 #define TVM_FFI_CYTHON_HELPERS_H_

#include <tvm/ffi/c_api.h>

namespace tvm_ffi {

}
/**
 * \brief Recycle temporary arguments
 * \param args The arguments to recycle
 * \param num_args The number of arguments
 */
inline void TVMFFICyRecycleTempArgs(
  TVMFFIAny* args, int32_t num_args, int64_t bitmask_temp_args) {
  if (bitmask_temp_args == 0)  return;
  for (int32_t i = 0; i < num_args; ++i) {
    if ((bitmask_temp_args >> i) & 1) {
      if (args[i].v_obj->deleter != nullptr) {
        args[i].v_obj->deleter(args[i].v_obj, kTVMFFIObjectDeleterFlagBitMaskBoth);
      }
    }
  }
}

inline void TVMFFICySetBitMaskTempArgs(int64_t* bitmask_temp_args, int32_t index) noexcept {
  *bitmask_temp_args |= 1 << index;
}

/*!
 * \brief Function pointer to speed convert a Tensor to a DLManagedTensor.
 * \param obj The Tensor to convert.
 * \param out The output DLManagedTensor.
 * \return 0 on success, nonzero on failure.
 */
typedef int (*TVMFFICyTensorToDLPackCallType)(void* py_obj, DLManagedTensor** out);
#endif
