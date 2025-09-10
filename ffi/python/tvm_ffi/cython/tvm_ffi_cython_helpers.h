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

#include <Python.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/base_details.h>
#include <unordered_map>
#include <exception>
/*!
 * \brief Function pointer to speed convert a Tensor to a DLManagedTensor.
 * \param obj The Tensor to convert.
 * \param out The output DLManagedTensor.
 * \return 0 on success, nonzero on failure.
 */
typedef int (*TVMFFICyTensorToDLPackCallType)(void* py_obj, DLManagedTensor** out);


struct TVMFFICyCallContext {
  /*! \brief the device type to be set during the call */
  int ctx_device_type = -1;
  /*! \brief the device id to be set during the call */
  int ctx_device_id = 0;
  /*! \brief the stream to be set during the call */
  void* ctx_stream = nullptr;
  /*! \brief the temporary arguments to be recycled */
  TVMFFIObjectHandle* temp_args = nullptr;
  /*! \brief the number of temporary arguments */
  int num_temp_args = 0;
};

/*!
 * \brief Function pointer to speed convert a Tensor to a DLManagedTensor.
 * \param obj The Tensor to convert.
 * \param out The output DLManagedTensor.
 * \return 0 on success, nonzero on failure.
 */
typedef int (*TVMFFICyTensorConverter)(PyObject* py_obj, DLManagedTensor** out);


/*! \brief the context for the argument setter */
struct TVMFFICyArgSetter {
  /*!
  * \brief Function pointer to set an argument
  * \param self The argument setter.
  * \param ctx The call context.
  * \param arg The argument to set.
  * \param py_arg The python argument.
  * \return 0 on success, nonzero on failure.
  */
  int (*func)(void* self, TVMFFICyCallContext* ctx,  PyObject* py_arg, TVMFFIAny* arg);
  /*!
   * \brief Optional tensor converter for setters that involves tensor conversion
   */
  TVMFFICyTensorConverter tensor_converter{nullptr};

  TVM_FFI_INLINE int operator()(TVMFFICyCallContext* ctx,  PyObject* py_arg, TVMFFIAny* arg) const {
    return (*func)(const_cast<TVMFFICyArgSetter*>(this), ctx, py_arg, arg);
  }
};

/*!
 * \brief Factory that returns the argument setter for a given python argument.
 * \param value The python argument.
 * \param out The output argument setter.
 * \return 0 on success, nonzero on failure.
 */
typedef int (*TVMFFICyArgSetterFactory)(PyObject* value, TVMFFICyArgSetter* out);

/*!
 * \brief thread-local context to help setup calls
 */
class TVMFFICyCallDispatcher {
 public:
 /*!
  * \brief Call a function with a variable number of arguments
  * \param chandle The handle of the function to call
  * \param py_args The arguments to the function
  * \param workspace_packed_args The workspace for the packed arguments
  * \param workspace_temp_args The workspace for the temporary arguments
  * \param num_args The number of arguments
  * \param c_api_ret_code The return code of the function
  * \return 0 on success, -1 on python error
  */
  TVM_FFI_INLINE int Call(void* chandle,
                          PyObject* py_arg_tuple,
                          TVMFFIAny* workspace_packed_args,
                          TVMFFIObjectHandle* workspace_temp_args,
                          int num_args,
                          TVMFFIAny* result,
                          int* c_api_ret_code) {
    try {
      TVMFFICyCallContext ctx;
      ctx.temp_args = workspace_temp_args;
      for (int i = 0; i < num_args; ++i) {
        PyObject* py_arg = PyTuple_GetItem(py_arg_tuple, i);
        TVMFFIAny* c_arg = workspace_packed_args + i;
        PyTypeObject* py_type = Py_TYPE(py_arg);
        auto it = dispatch_.find(py_type);
        if (it != dispatch_.end()) {
          TVMFFICyArgSetter setter = it->second;
          if (setter(&ctx, py_arg, c_arg) != 0) return -1;
        } else {
          // no dispatch found, query and create a new one.
          TVMFFICyArgSetter setter;
          // propagate python error back
          if (setter_factory_(py_arg, &setter) != 0) {
            return -1;
          }
          dispatch_.emplace(py_type, setter);
          if (setter(&ctx, py_arg, c_arg) != 0) return -1;
        }
      }
      c_api_ret_code[0] = TVMFFIFunctionCall(chandle, workspace_packed_args, num_args, result);
      // recycle the temporary arguments if any
      for (int i = 0; i < ctx.num_temp_args; ++i) {
        TVMFFIObject* obj = static_cast<TVMFFIObject*>(ctx.temp_args[i]);
        if (obj->deleter != nullptr) {
          obj->deleter(obj, kTVMFFIObjectDeleterFlagBitMaskBoth);
        }
      }
      return 0;
    } catch (const std::exception& err) {
      // this is very rare, but we would like to guard against possible c++ exceptions
      TVMFFIErrorSetRaisedFromCStr("RuntimeError", err.what());
      // this is not a python error, so we return by c_api_ret_code
      c_api_ret_code[0] = -1;
      return 0;
    }
  }

  static TVMFFICyCallDispatcher* ThreadLocal(TVMFFICyArgSetterFactory factory) {
   static thread_local TVMFFICyCallDispatcher inst(factory);
   return &inst;
  }

 private:
  explicit TVMFFICyCallDispatcher(TVMFFICyArgSetterFactory factory) : setter_factory_(factory) {}

  std::unordered_map<PyTypeObject*, TVMFFICyArgSetter> dispatch_;
  TVMFFICyArgSetterFactory setter_factory_;
};


/*!
 * \brief Call a function with a variable number of arguments
 * \param chandle The handle of the function to call
 * \param args The arguments to the function
 * \param num_args The number of arguments
 * \param result The result of the function
 * \param c_api_ret_code The return code of the function
 * \return 0 on success, nonzero on failure
 */
inline int TVMFFICyFuncCallImpl(void* chandle,
                                PyObject* py_args,
                                TVMFFIAny* packed_args,
                                int num_args,
                                TVMFFIAny* result,
                                int* c_api_ret_code) {


  return 0;
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

#endif
