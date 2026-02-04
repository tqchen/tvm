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
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import tir as T


def _build_mod_unroll_loop():
    """Build module with outer serial(n, n+2) and inner unroll(0, 8)."""
    with IRBuilder() as ib:
        with I.ir_module():
            with T.prim_func():
                T.func_name("main")
                n = T.arg("n", T.int32())
                A_handle = T.arg("A", T.handle())
                A = T.match_buffer(A_handle, (n,), "int64")
                with T.serial(n, n + 2) as i:
                    with T.unroll(0, 8) as j:
                        T.buffer_store(A, A[i] + T.int64(1), [j + 1])
    return ib.get()


def _build_mod_with_pragma():
    """Build module with pragma_auto_unroll_max_step then two copies of the loop."""
    with IRBuilder() as ib:
        with I.ir_module():
            with T.prim_func():
                T.func_name("main")
                n = T.arg("n", T.int32())
                A_handle = T.arg("A", T.handle())
                A = T.match_buffer(A_handle, (n,), "int64")
                with T.attr(T.int32(0), "pragma_auto_unroll_max_step", 16):
                    with T.serial(n, n + 2) as i:
                        with T.unroll(0, 8) as j:
                            T.buffer_store(A, A[i] + T.int64(1), [j + 1])
                with T.serial(n, n + 2) as i:
                    with T.unroll(0, 8) as j:
                        T.buffer_store(A, A[i] + T.int64(1), [j + 1])
    return ib.get()


def test_unroll_loop():
    Mod = _build_mod_unroll_loop()
    body = Mod["main"].body
    assert isinstance(body, tvm.tir.For)

    with tvm.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 16}}):
        ret = tvm.tir.transform.UnrollLoop()(Mod)["main"].body
        assert not isinstance(ret, tvm.tir.For)

    with tvm.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 15}}):
        ret = tvm.tir.transform.UnrollLoop()(Mod)["main"].body
        assert isinstance(ret, tvm.tir.For)

    with tvm.transform.PassContext(
        config={"tir.UnrollLoop": {"auto_max_step": 16, "explicit_unroll": False}}
    ):
        ret = tvm.tir.transform.UnrollLoop()(Mod)["main"].body
        assert isinstance(ret, tvm.tir.For)
        assert ret.kind == tvm.tir.ForKind.UNROLLED

    ModWithPragma = _build_mod_with_pragma()
    with tvm.transform.PassContext(
        config={"tir.UnrollLoop": {"auto_max_depth": 8, "explicit_unroll": False}}
    ):
        ret = tvm.tir.transform.UnrollLoop()(ModWithPragma)["main"].body
        assert isinstance(ret[0], tvm.tir.For)
        assert ret[0].kind == tvm.tir.ForKind.UNROLLED
        assert isinstance(ret[1], tvm.tir.For)
        assert ret[1].kind != tvm.tir.ForKind.UNROLLED


def _build_mod_unroll_fake_loop():
    with IRBuilder() as ib:
        with I.ir_module():
            with T.prim_func():
                T.func_name("main")
                n = T.arg("n", T.int32())
                A_handle = T.arg("A", T.handle())
                A = T.match_buffer(A_handle, (n,), "int32")
                with T.serial(0, 1) as i:
                    T.buffer_store(A, T.int32(3), [i * 2])
                    with T.serial(0, 10) as j:
                        T.buffer_store(A, A[i] + T.int32(1), [j + 1])
    return ib.get()


def test_unroll_fake_loop():
    Mod = _build_mod_unroll_fake_loop()
    with tvm.transform.PassContext(
        config={
            "tir.UnrollLoop": {"auto_max_depth": 8, "auto_max_extent": 1, "explicit_unroll": False}
        }
    ):
        ret = tvm.tir.transform.UnrollLoop()(Mod)["main"].body
        assert isinstance(ret[0], tvm.tir.BufferStore)


def _build_mod_unroll_allocations_before():
    with IRBuilder() as ib:
        with I.ir_module():
            with T.prim_func():
                T.func_name("main")
                with T.unroll(0, 2) as i:
                    with T.decl_buffer([16], "float32") as buf:
                        T.buffer_store(buf, T.float32(0.0), [0])
    return ib.get()


def _build_mod_unroll_allocations_expected():
    with IRBuilder() as ib:
        with I.ir_module():
            with T.prim_func():
                T.func_name("main")
                with T.decl_buffer([16], "float32") as buf1:
                    T.buffer_store(buf1, T.float32(0.0), [0])
                with T.decl_buffer([16], "float32") as buf2:
                    T.buffer_store(buf2, T.float32(0.0), [0])
    return ib.get()


def test_unroll_allocations():
    Before = _build_mod_unroll_allocations_before()
    Expected = _build_mod_unroll_allocations_expected()
    after = tvm.tir.transform.UnrollLoop()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def _build_mod_unroll_local_access_before():
    with IRBuilder() as ib:
        with I.ir_module():
            with T.prim_func():
                T.func_name("main")
                T.arg("B", T.Buffer((64,), "float32"))
                with T.thread_binding(4, thread="blockIdx.x") as bx:
                    with T.thread_binding(4, thread="threadIdx.x") as tx:
                        with T.allocate([4], "float32", scope="local") as A_local_data:
                            A_local = T.Buffer([4], dtype="float32", data=A_local_data)
                            with T.serial(4) as i:
                                T.buffer_store(A_local, T.float32(i), [i])
    return ib.get()


def _build_mod_unroll_local_access_expected():
    with IRBuilder() as ib:
        with I.ir_module():
            with T.prim_func():
                T.func_name("main")
                T.arg("B", T.Buffer((64,), "float32"))
                with T.thread_binding(4, thread="blockIdx.x") as bx:
                    with T.thread_binding(4, thread="threadIdx.x") as tx:
                        with T.allocate([4], "float32", scope="local") as A_local_data:
                            A_local = T.Buffer([4], dtype="float32", data=A_local_data)
                            T.buffer_store(A_local, T.float32(0), [0])
                            T.buffer_store(A_local, T.float32(1), [1])
                            T.buffer_store(A_local, T.float32(2), [2])
                            T.buffer_store(A_local, T.float32(3), [3])
    return ib.get()


def test_unroll_local_access():
    Before = _build_mod_unroll_local_access_before()
    Expected = _build_mod_unroll_local_access_expected()
    with tvm.transform.PassContext(
        config={
            "tir.UnrollLoop": {
                "auto_max_depth": 0,
                "auto_max_extent": 1,
                "explicit_unroll": True,
                "unroll_local_access": True,
            }
        }
    ):
        after = tvm.tir.transform.UnrollLoop()(Before)
        after = tvm.tir.transform.Simplify()(after)

    tvm.ir.assert_structural_equal(after, Expected)


if __name__ == "__main__":
    test_unroll_local_access()
    test_unroll_loop()
    test_unroll_fake_loop()
    test_unroll_allocations()
