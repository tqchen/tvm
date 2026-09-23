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
"""Native buffer stride declarations preserve dtype and symbol identity."""

import pytest

import tvm
from tvm.script import parser


@pytest.mark.parametrize("constructor", ["Buffer", "match_buffer"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_implicit_buffer_strides_use_symbolic_default(constructor, index_dtype):
    if constructor == "Buffer":
        source = """
@T.prim_func
def main(A: T.Buffer((16, 16), "float32", strides=("s0", "s1"))):
    T.evaluate(A[0, 0])
"""
    else:
        source = """
@T.prim_func
def main(a: T.handle):
    A = T.match_buffer(a, (16, 16), "float32", strides=("s0", "s1"))
    T.evaluate(A[0, 0])
"""
    source = source.replace("(16, 16)", f"(T.{index_dtype}(16), T.{index_dtype}(16))")
    function = parser.parse(source)
    assert [str(value.ty.dtype) for value in function.params[0].strides] == ["int64", "int64"]


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_explicit_stride_declaration_keeps_its_dtype_and_identity(dtype):
    function = parser.parse(f"""
@T.prim_func
def main(A: T.Buffer((16, 16), "float32", strides=("s0", "s1")), s0: T.{dtype}):
    s1 = T.int64()
    T.evaluate(A[0, 0] + s0 + s1)
""")
    buffer, stride = function.params
    assert buffer.strides[0].same_as(stride)
    assert [str(value.ty.dtype) for value in buffer.strides] == [dtype, "int64"]


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_match_buffer_reuses_prior_explicit_stride(dtype):
    function = parser.parse(f"""
@T.prim_func
def main(a: T.handle):
    s0 = T.{dtype}()
    A = T.match_buffer(a, (16,), "float32", strides=("s0",))
    T.evaluate(s0)
""")
    buffer = function.params[0]
    stride_use = function.body.value
    assert buffer.strides[0].same_as(stride_use)
    assert str(buffer.strides[0].ty.dtype) == dtype


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_decl_buffer_stride_roundtrip_preserves_dtype(dtype):
    # Standalone buffer declarations can contain free stride symbols.
    function = parser.parse(
        f"""
@T.prim_func
def main(data: T.handle("float32")):
    s0 = T.{dtype}()
    A = T.decl_buffer((1,), "float32", data=data, strides=(s0,))
    T.evaluate(A[0])
""",
        check_well_formed=False,
    )
    printed = function.script()
    if dtype == "int64":
        assert 'strides=("s0",)' in printed
    else:
        assert "s0 = T.int32()" in printed
    tvm.ir.assert_structural_equal(
        function, parser.parse(printed, check_well_formed=False), map_free_vars=True
    )
