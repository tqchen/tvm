import argparse
import torch
from torch.utils import cpp_extension
import time

import tvm_ffi
import tvm_ffi.cpp


def gen_torch_func():
    cpp_source = """
void fake_fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps) {
    // Do nothing.
}
"""
    module = cpp_extension.load_inline(
        name="torch_fake_fused_add_rmsnorm",
        cpp_sources=cpp_source,
        functions="fake_fused_add_rmsnorm",
        extra_cflags=["-O3"],
        verbose=True,
    )

    return module.fake_fused_add_rmsnorm


def gen_ffi_func():
    cpp_source = """
#include <tvm/ffi/container/tensor.h>
void fake_fused_add_rmsnorm(tvm::ffi::Tensor input, tvm::ffi::Tensor residual, tvm::ffi::Tensor weight, double eps) {
    // Do nothing.
}
"""

    module = tvm_ffi.cpp.load_inline(
        name="ffi_fake_fused_add_rmsnorm",
        cpp_sources=cpp_source,
        functions="fake_fused_add_rmsnorm",
        extra_cflags=["-O3"],
    )
    return tvm_ffi.get_global_func("testing.nop")
    return module.fake_fused_add_rmsnorm


def bench(use_cuda_graph, backend, fn, nrepeat=10000):
    if not use_cuda_graph:
        for _ in range(100):
            fn()

        torch.cuda.synchronize()
        tic = time.time()
        for _ in range(nrepeat):
            fn()

        torch.cuda.synchronize()
        toc = time.time()
        print(f"{backend} w/o CUDAGraph {(toc - tic)/nrepeat} sec/call")
    else:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(nrepeat):
                fn()

        # residual.zero_()
        torch.cuda.synchronize()
        tic = time.time()
        g.replay()
        torch.cuda.synchronize()
        toc = time.time()
        print(f"{backend} w/  CUDAGraph {(toc - tic)/nrepeat} sec/call")



def bench_to_dlpack(x, nrepeat=1000):
    import hack_dlpack



@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
    )
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()

    eps = 1e-6
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    dtype_str = args.dtype

    # Loop over each combination of batch_size, hidden_size, and dtype
    dtype = getattr(torch, dtype_str)

    # Define tensors with the correct dtype
    x = torch.randn((batch_size, hidden_size), dtype=dtype, device="cuda")
    residual = torch.zeros_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    torch_func = gen_torch_func()
    ffi_func = gen_ffi_func()

    bench_to_dlpack(x)
    bench(False, "torch", lambda: torch_func(x, residual, weight, eps))
    bench(False, "  ffi", lambda: ffi_func(x, residual, weight, eps))
    x0 = tvm_ffi.from_dlpack(x)
    residual0 = tvm_ffi.from_dlpack(residual)
    weight0 = tvm_ffi.from_dlpack(weight)
    bench(False, "  ffi-stol", lambda: ffi_func(x0, residual0, weight0, eps))

if __name__ == "__main__":
    main()