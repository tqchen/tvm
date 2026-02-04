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
import tvm.testing

from tvm import te
from tvm.script import tir as T, ir as I
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I_
from tvm.script.ir_builder import tir as T_


def test_double_buffer():
    n = 100
    m = 4
    tx = te.thread_axis("threadIdx.x")
    with IRBuilder() as ib:
        with I_.ir_module():
            with T_.prim_func():
                T_.func_name("db")
                A_handle = T_.arg("A", T_.handle())
                C_handle = T_.arg("C", T_.handle())
                A = T_.match_buffer(A_handle, (n * m,), "float32")
                C = T_.match_buffer(C_handle, (m,), "float32")
                with T_.attr(tx, "thread_extent", 1):
                    with T_.serial(0, n) as i:
                        with T_.allocate([m], "float32", scope="shared") as B_data:
                            B = T_.Buffer([m], "float32", data=B_data)
                            with T_.attr(B_data, "double_buffer_scope", 1):
                                with T_.serial(0, m) as j:
                                    T_.buffer_store(B, A[i * 4 + j], [j])
                            with T_.serial(0, m) as j:
                                T_.buffer_store(C, B[j] + T_.float32(1), [j])
    mod = ib.get()

    opt = tvm.transform.Sequential(
        [tvm.tir.transform.InjectDoubleBuffer(), tvm.tir.transform.Simplify()]
    )

    with tvm.transform.PassContext(config={"tir.InjectDoubleBuffer": {"split_loop": 2}}):
        mod = opt(mod)
    stmt = mod["db"].body

    # Find the Allocate node (may be at different depths depending on IR structure)
    alloc = stmt
    while not isinstance(alloc, tvm.tir.Allocate):
        alloc = alloc.body
    assert list(alloc.extents) == [m * 2]

    f = tvm.tir.transform.ThreadSync("shared")(mod)["db"]
    count = [0]

    def count_sync(op):
        if isinstance(op, tvm.tir.Call) and op.op.same_as(tvm.ir.Op.get("tir.tvm_storage_sync")):
            count[0] += 1

    tvm.tir.stmt_functor.post_order_visit(f.body, count_sync)
    assert count[0] == 4


def test_double_buffer_transform():
    transform = tvm.ir.transform.Sequential(
        [
            tvm.tir.transform.InjectDoubleBuffer(),
            tvm.tir.transform.Simplify(),
        ]
    )

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer([16, 32], "float32"), B: T.Buffer(16, "float32")):
            for i in range(16):
                cache_data = T.allocate([32], "float32")
                cache = T.Buffer(32, "float32", data=cache_data)

                T.attr(cache_data, "double_buffer_scope", 1)

                for j in range(32):
                    cache[j] = A[i, j]

                B[i] = 0.0
                for j in range(32):
                    B[i] = B[i] + cache[j]

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer((16, 32), "float32"), B: T.Buffer((16,), "float32")):
            cache_data = T.allocate([64], "float32", "global")
            cache = T.Buffer(64, data=cache_data)
            for j in range(32):
                cache[j] = A[0, j]

            B[0] = T.float32(0)
            for j in range(32):
                B[0] = B[0] + cache[j]

            for i_outer in range(15):
                T.attr(cache_data, "double_buffer_write", 1)
                for j in range(32):
                    cache[(i_outer + 1) % 2 * 32 + j] = A[i_outer + 1, j]
                B[i_outer + 1] = T.float32(0)
                for j in range(32):
                    B[i_outer + 1] = B[i_outer + 1] + cache[(i_outer + 1) % 2 * 32 + j]

    After = transform(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_double_buffer_with_decl_buffer():
    """Like test_double_buffer_transform, but with a declared buffer object"""

    transform = tvm.ir.transform.Sequential(
        [
            tvm.tir.transform.InjectDoubleBuffer(),
            tvm.tir.transform.Simplify(),
        ]
    )

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((16, 32), "float32"), B: T.Buffer(16, "float32")):
            for i in range(16):
                cache = T.decl_buffer(32, "float32")
                T.attr(cache.data, "double_buffer_scope", 1)

                for j in range(32):
                    cache[j] = A[i, j]

                B[i] = 0.0
                for j in range(32):
                    B[i] = B[i] + cache[j]

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer((16, 32), "float32"), B: T.Buffer(16, "float32")):
            cache = T.decl_buffer(64, "float32")
            for j in range(32):
                cache[j] = A[0, j]

            B[0] = T.float32(0)
            for j in range(32):
                B[0] = B[0] + cache[j]

            for i_outer in range(15):
                T.attr(cache.data, "double_buffer_write", 1)
                for j in range(32):
                    cache[(i_outer + 1) % 2 * 32 + j] = A[i_outer + 1, j]
                B[i_outer + 1] = T.float32(0)
                for j in range(32):
                    B[i_outer + 1] = B[i_outer + 1] + cache[(i_outer + 1) % 2 * 32 + j]

    After = transform(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
