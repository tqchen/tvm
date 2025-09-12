# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
This script is used to benchmark the API overhead of different
cpp extension based DLPack conversion overhead.
"""
import tvm_ffi
import tvm_ffi.cpp
import torch
import time
import sys


def print_speed(name, speed):
    print(f"{name:<60} {speed} sec/call")


def get_ffi_dlpack_bench():
    cpp_source = """
#include <tvm/ffi/container/tensor.h>
#include <iostream>
namespace ffi = tvm::ffi;

typedef int (*DLPackPyCExporter)(void* py_obj, DLManagedTensorVersioned** out, void** env_stream);


void dlpack_cpp_exporter_bench(tvm::ffi::Tensor input, int repeat) {
  TVMFFIObjectHandle chandle = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(input));
  DLManagedTensorVersioned* dlpack;
  for (int i = 0; i < repeat; i++) {
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITensorToDLPackVersioned(chandle, &dlpack));
    dlpack->deleter(dlpack);
  }
  TVMFFIObjectDecRef(chandle);
}

void dlpack_cpp_full_bench(tvm::ffi::Tensor input, int repeat) {
  TVMFFIObjectHandle chandle = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(input));
  TVMFFIObjectHandle temp;
  DLManagedTensorVersioned* dlpack;
  for (int i = 0; i < repeat; i++) {
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITensorToDLPackVersioned(chandle, &dlpack));
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITensorFromDLPackVersioned(dlpack, 0, 0, &temp));
    TVMFFIObjectDecRef(temp);
  }
  TVMFFIObjectDecRef(chandle);
}



"""
    module = tvm_ffi.cpp.load_inline(
        name="dlpack_bench",
        cpp_sources=cpp_source,
        functions=["dlpack_cpp_exporter_bench", "dlpack_cpp_full_bench"],
        extra_cflags=["-O3"],
    )
    return module

import gc


def run_dlpack_cpp_exporter_bench(name, x, func, repeat):
    func(x, 1)
    tstart = time.time()
    func(x, repeat)
    tend = time.time()
    print_speed(name, (tend - tstart) / repeat)


def main():
    repeat = 100000
    module = get_ffi_dlpack_bench()
    x = torch.arange(1, device="cuda")


    ffi_x = tvm_ffi.from_dlpack(torch.arange(1, device="cuda"))
    wrapper_x = tvm_ffi.core.DLTensorTestWrapper(ffi_x)
    x.temp = wrapper_x

    run_dlpack_cpp_exporter_bench("cpp-exporter-bench", ffi_x, module.dlpack_cpp_exporter_bench, repeat)
    run_dlpack_cpp_exporter_bench("cpp-full-bench", ffi_x, module.dlpack_cpp_full_bench, repeat)


if __name__ == "__main__":
    main()
