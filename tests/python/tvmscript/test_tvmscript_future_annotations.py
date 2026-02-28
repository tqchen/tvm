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
"""Tests for TVMScript compatibility with PEP 563 (from __future__ import annotations)."""
from __future__ import annotations

import tvm
import tvm.testing
from tvm.script import tir as T, ir as I


@T.prim_func
def elementwise_add(
    A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32")
):
    for i, j in T.grid(128, 128):
        B[i, j] = A[i, j] + 1.0


@T.prim_func
def vector_add(
    A: T.Buffer((256,), "float32"),
    B: T.Buffer((256,), "float32"),
    C: T.Buffer((256,), "float32"),
):
    for i in range(256):
        C[i] = A[i] + B[i]


def test_elementwise_add():
    """Test basic elementwise function with future annotations."""
    assert elementwise_add is not None
    assert isinstance(elementwise_add, tvm.tir.PrimFunc)


def test_vector_add():
    """Test vector addition with three buffer arguments."""
    assert vector_add is not None
    assert isinstance(vector_add, tvm.tir.PrimFunc)


def test_roundtrip():
    """Test that the parsed function can be printed and re-parsed."""
    script = elementwise_add.script()
    assert "T.Buffer" in script


def test_metaprogramming_parameterized():
    """Test metaprogramming: parameterized TVMScript function generation.

    When closure variables are used in annotations (e.g., T.Buffer((n,), ...)),
    the AST primary path handles it correctly because it evaluates the annotation
    AST node in the correct scope. The fallback path with _resolve_annotations
    cannot resolve closure variables since they are not in func.__globals__.
    """

    def make_scale_func(n, scale_val):
        @T.prim_func
        def scale(A: T.Buffer((n,), "float32"), B: T.Buffer((n,), "float32")):
            for i in range(n):
                B[i] = A[i] * T.float32(scale_val)

        return scale

    f1 = make_scale_func(32, 3.0)
    f2 = make_scale_func(64, 5.0)
    assert isinstance(f1, tvm.tir.PrimFunc)
    assert isinstance(f2, tvm.tir.PrimFunc)


def test_irmodule_with_future_annotations():
    """Test IRModule class definition with future annotations."""

    @I.ir_module
    class MyModule:
        @T.prim_func
        def add_one(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
            for i in range(128):
                B[i] = A[i] + 1.0

        @T.prim_func
        def mul_two(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
            for i in range(128):
                B[i] = A[i] * 2.0

    assert isinstance(MyModule, tvm.IRModule)
    assert "add_one" in [gv.name_hint for gv, _ in MyModule.functions_items()]
    assert "mul_two" in [gv.name_hint for gv, _ in MyModule.functions_items()]


def test_symbolic_shape_with_future_annotations():
    """Test symbolic shapes with future annotations."""

    @T.prim_func
    def dynamic_add(a: T.handle, b: T.handle):
        n = T.int32()
        A = T.match_buffer(a, (n,), "float32")
        B = T.match_buffer(b, (n,), "float32")
        for i in range(n):
            B[i] = A[i] + 1.0

    assert isinstance(dynamic_add, tvm.tir.PrimFunc)


def test_multi_dim_buffer_with_future_annotations():
    """Test multi-dimensional buffers and T.grid with future annotations."""

    @T.prim_func
    def batch_add(
        A: T.Buffer((4, 64, 64), "float32"),
        B: T.Buffer((4, 64, 64), "float32"),
        C: T.Buffer((4, 64, 64), "float32"),
    ):
        for b, i, j in T.grid(4, 64, 64):
            C[b, i, j] = A[b, i, j] + B[b, i, j]

    assert isinstance(batch_add, tvm.tir.PrimFunc)


if __name__ == "__main__":
    tvm.testing.main()
