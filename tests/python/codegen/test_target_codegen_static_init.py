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
import tvm
from tvm import te
import ctypes
import numpy as np
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I_
from tvm.script.ir_builder import tir as T_


def test_static_callback():
    dtype = "int64"
    cp = te.thread_axis((0, 1), "cop")
    finit = tvm.tir.StringImm("TVMBackendRunOnce")
    with IRBuilder() as ib:
        with I_.ir_module():
            with T_.prim_func():
                T_.func_name("ramp")
                A_handle = T_.arg("A", T_.handle())
                n = tvm.tir.Var("n", "int32")
                A = T_.match_buffer(A_handle, (n,), dtype)
                with T_.attr(cp, "coproc_uop_scope", finit):
                    with T_.parallel(0, n) as i:
                        T_.buffer_store(A, A[i] + T_.int64(1), [i])
    mod = ib.get()
    f = tvm.driver.build(mod, target="llvm")
    a = tvm.runtime.tensor(np.zeros(10, dtype=dtype))
    f(a)
    f(a)
    np.testing.assert_equal(a.numpy(), np.ones(a.shape[0]))


def test_static_init():
    dtype = "int64"

    @tvm.register_global_func("test_static_callback")
    def test_cb(sh, A):
        assert isinstance(sh, ctypes.c_void_p)
        return sh

    with IRBuilder() as ib:
        with I_.ir_module():
            with T_.prim_func():
                T_.func_name("ramp")
                A_handle = T_.arg("A", T_.handle())
                n = tvm.tir.Var("n", "int32")
                A = T_.match_buffer(A_handle, (n,), dtype)
                handle = tvm.tir.call_intrin("handle", "tir.tvm_static_handle")
                T_.evaluate(tvm.tir.call_packed("test_static_callback", handle, A.data))
    mod = ib.get()
    f = tvm.driver.build(mod, target="llvm")
    a = tvm.runtime.tensor(np.zeros(10, dtype=dtype))
    f(a)


if __name__ == "__main__":
    test_static_callback()
    test_static_init()
