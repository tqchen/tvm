import tvm_ffi
import tvm_ffi.cpp
import torch
import time

def get_ffi_dlpack_bench():
    cpp_source = """
#include <tvm/ffi/container/tensor.h>
namespace ffi = tvm::ffi;

void dlpack_bench(tvm::ffi::Tensor input, int repeat) {
  TVMFFIObjectHandle chandle = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(input));
  TVMFFIObjectHandle output;
  DLManagedTensor* dlpack;
  for (int i = 0; i < repeat; i++) {
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITensorToDLPack(chandle, &dlpack));
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITensorFromDLPack(dlpack, 0, 0, &output));
    TVMFFIObjectDecRef(output);
  }
  TVMFFIObjectDecRef(chandle);
}
"""

    module = tvm_ffi.cpp.load_inline(
        name="dlpack_bench",
        cpp_sources=cpp_source,
        functions="dlpack_bench",
        extra_cflags=["-O3"],
    )
    return module.dlpack_bench

def run_dlpack_bench(name, func, repeat):
    x = tvm_ffi.from_dlpack(torch.arange(1))
    func(x, 1)
    tstart = time.time()
    func(x, repeat)
    tend = time.time()
    print(f"Time taken: {(tend - tstart) / repeat} secs/call")


def test_get_dlpack_c_converter(repeat):
    x = tvm_ffi.from_dlpack(torch.arange(1))
    x = tvm_ffi.core.DLTensorTestWrapper(x)
    z = 0
    tstart = time.time()
    for i in range(repeat):
        if hasattr(x, "__dlpack_c_converter__"):
            z += x.__dlpack_c_converter__
    tend = time.time()
    print(f"DLPack C converter: {(tend - tstart) / repeat} secs/call")


def main():
    repeat = 10000
    run_dlpack_bench("ffi", get_ffi_dlpack_bench(), repeat)
    test_get_dlpack_c_converter(repeat)

if __name__ == "__main__":
    main()