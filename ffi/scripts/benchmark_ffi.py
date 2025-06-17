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

from tvm import ffi as tvm_ffi
import time

def addone(x):
    return x + 1


def print_speed(name, speed):
    print(f"{name:<40} {speed} sec/call")


def benchmark_addone(repeat, addone):
    result = 0
    addone(0)
    start = time.time()
    for i in range(repeat):
        addone(result)
    end = time.time()
    speed = (end - start) / repeat
    name = str(addone.__module__) + "." + addone.__name__
    print_speed(name, speed)
    return result


def benchmark_loop(repeat):
    result = 0
    start = time.time()
    for i in range(repeat):
        result += 1
    end = time.time()
    speed = (end - start) / repeat
    name = "loop"
    print_speed(name, speed)
    return result



class MyClass:
    def __init__(self):
        self.value = 10

    @property
    def v_i64(self):
        return self.value + 1


class MyClassDirect:
    def __init__(self):
        self.v_i64 = 10


class MyClassSub(MyClass):
    pass

def benchmark_getter(repeat, obj):
    result = 0
    start = time.time()
    for i in range(repeat):
        obj.v_i64
    end = time.time()
    speed = (end - start) / repeat
    name = type(obj).__name__ + ".v_i64"
    print_speed(name, speed)
    return result


def benchmark_invoke_n_times(repeat, f):
    f = tvm_ffi.convert(f)
    f(0)
    invoke_n = tvm_ffi.get_global_func("testing.invoke_n_times")
    start = time.time()
    invoke_n(repeat, f)
    end = time.time()
    speed = (end - start) / repeat
    name = str(f.__module__) + "." + f.__name__
    print_speed(name, speed)



def benchmark_invoke_n_times_native(repeat, selector):
    invoke_n = tvm_ffi.get_global_func("testing.invoke_n_times_native")
    start = time.time()
    invoke_n(repeat, selector)
    end = time.time()
    speed = (end - start) / repeat
    name = "invoke_n_times_native"
    print_speed(name, speed)


def benchmark_invoke_n_times_any(repeat, val):
    invoke_n = tvm_ffi.get_global_func("testing.invoke_n_times_any")
    start = time.time()
    invoke_n(repeat, val)
    end = time.time()
    speed = (end - start) / repeat
    name = "invoke_n_times_any"
    print_speed(name, speed)

def benchmark_invoke_n_times_round(repeat, val):
    invoke_n = tvm_ffi.get_global_func("testing.invoke_n_times_round")
    start = time.time()
    invoke_n(repeat, val)
    end = time.time()
    speed = (end - start) / repeat
    name = "invoke_n_times_round"
    print_speed(name, speed)


def main():
    repeat = 100000
    print("-----------------------------")
    print("Benchmark addone(x) overhead")
    print("-----------------------------")
    benchmark_loop(repeat)
    benchmark_addone(repeat, lambda x: x + 1)
    benchmark_addone(repeat, tvm_ffi.core.TestAddOne)
    benchmark_addone(repeat, tvm_ffi.extra.add_one)
    print("-----------------------------")
    print("Benchmark x.v_i64 overhead")
    print("-----------------------------")
    benchmark_getter(repeat, MyClass())
    benchmark_getter(repeat, MyClassDirect())
    benchmark_getter(repeat, tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=10))
    benchmark_getter(repeat, MyClassSub())
    print("-----------------------------")
    print("Benchmark testing.invoke_n_times overhead")
    print("-----------------------------")
    benchmark_invoke_n_times(repeat, lambda x: x + 1)
    benchmark_invoke_n_times(repeat, tvm_ffi.get_global_func("testing.echo"))
    benchmark_invoke_n_times_native(repeat, selector=2)
    benchmark_invoke_n_times_any(repeat, tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=10))
    benchmark_invoke_n_times_round(repeat, "a")

if __name__ == "__main__":
    main()
