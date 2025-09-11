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

try:
  import hack_torch_dlpack
except ImportError:
  hack_torch_dlpack = None
  pass

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

void dlpack_py_c_exporter_bench(int64_t py_obj_ptr, int64_t dlpack_c_exporter, int repeat) {
  DLPackPyCExporter exporter = reinterpret_cast<DLPackPyCExporter>(dlpack_c_exporter);
  void* py_obj = reinterpret_cast<void*>(py_obj_ptr);
  for (int i = 0; i < repeat; i++) {
    DLManagedTensorVersioned* dlpack;
    (*exporter)(py_obj, &dlpack, nullptr);
    dlpack->deleter(dlpack);
  }
}
"""
    module = tvm_ffi.cpp.load_inline(
        name="dlpack_bench",
        cpp_sources=cpp_source,
        functions=["dlpack_cpp_exporter_bench", "dlpack_py_c_exporter_bench"],
        extra_cflags=["-O3"],
    )
    return module


def run_dlpack_cpp_exporter_bench(name, x, func, repeat):
    x = tvm_ffi.from_dlpack(torch.arange(1))
    func(x, 1)
    tstart = time.time()
    func(x, repeat)
    tend = time.time()
    print_speed(name, (tend - tstart) / repeat)


def run_dlpack_py_c_exporter_bench(name, x, func, repeat, cached=False):
    func_ptr = x.__dlpack_c_exporter_cached__ if cached else x.__dlpack_c_exporter__
    func(x.__pyobject_ptr__(), func_ptr, 1)
    tstart = time.time()
    func(x.__pyobject_ptr__(), func_ptr, repeat)
    tend = time.time()
    print_speed(f"{name}[cached={cached}]", (tend - tstart) / repeat)



def main():
    repeat = 10000
    module = get_ffi_dlpack_bench()
    x = torch.arange(1, device="cuda")
    ffi_x = tvm_ffi.from_dlpack(x)
    wrapper_x = tvm_ffi.core.DLTensorTestWrapper(ffi_x)
    run_dlpack_cpp_exporter_bench("cpp-exporter-bench", ffi_x, module.dlpack_cpp_exporter_bench, repeat)
    run_dlpack_py_c_exporter_bench("py-c-exporter-bench", wrapper_x, module.dlpack_py_c_exporter_bench, repeat, cached=False)
    run_dlpack_py_c_exporter_bench("py-c-exporter-bench", wrapper_x, module.dlpack_py_c_exporter_bench, repeat, cached=True)
    if hack_torch_dlpack is not None:
        torch_module = hack_torch_dlpack.load_to_dlpack()
        x.__dlpack_c_exporter__ = torch_module.TorchDLPackPyCExporterPtr(False)
        x.__dlpack_c_exporter_cached__ = torch_module.TorchDLPackPyCExporterPtr(True)
        run_dlpack_cpp_exporter_bench("torch-cpp-exporter-bench", x, module.dlpack_cpp_exporter_bench, repeat)
        # run_dlpack_py_c_exporter_bench("torch-py-c-exporter-bench", x, module.dlpack_py_c_exporter_bench, repeat, cached=False)

if __name__ == "__main__":
    main()
