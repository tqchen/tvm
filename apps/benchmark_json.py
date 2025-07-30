import json
import sys
import time
from tvm import ffi as tvm_ffi


def benchmark_json_parse(name, func, data, nrepeat=2):
    func(data)
    start = time.time()
    for i in range(nrepeat):
        func(data)
    end = time.time()
    speed =  (end - start) / nrepeat
    print(f"{name}: {speed} sec/iter")


def run_benchmark():
    ffi_parse = tvm_ffi.get_global_func("ffi.json.Parse")
    picojson_parse = tvm_ffi.get_global_func("ffi.picojson.parse")
    ffi_string_copy = tvm_ffi.get_global_func("ffi.json.StringCopy")
    ffi_new_parse = tvm_ffi.get_global_func("ffi.json.NewParse")
    data = open(sys.argv[1], "r").read()
    data = tvm_ffi.convert(data)
    benchmark_json_parse("json.loads", json.loads, data)
    benchmark_json_parse("tvm_ffi.json.Parse", ffi_parse, data)
    benchmark_json_parse("tvm_ffi.picojson.parse", picojson_parse, data)
    benchmark_json_parse("tvm_ffi.json.StringCopy", ffi_string_copy, data)
    benchmark_json_parse("tvm_ffi.json.NewParse", ffi_new_parse, data)


if __name__ == "__main__":
    run_benchmark()
