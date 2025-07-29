import json
import sys
import time
from tvm import ffi as tvm_ffi


def benchmark_json_parse(name, func, nrepeat=2):
    x = open(sys.argv[1], "r").read()
    func(x)
    start = time.time()
    for i in range(nrepeat):
        func(x)
    end = time.time()
    speed =  (end - start) / nrepeat
    print(f"{name}: {speed} sec/iter")


def run_benchmark():
    ffi_parse = tvm_ffi.get_global_func("ffi.json.Parse")
    picojson_parse = tvm_ffi.get_global_func("ffi.picojson.parse")
    ffi_string_copy = tvm_ffi.get_global_func("ffi.json.StringCopy")
    benchmark_json_parse("json.loads", json.loads)
    benchmark_json_parse("tvm_ffi.json.Parse", ffi_parse)
    benchmark_json_parse("tvm_ffi.picojson.parse", picojson_parse)
    benchmark_json_parse("tvm_ffi.json.StringCopy", ffi_string_copy)


if __name__ == "__main__":
    run_benchmark()
