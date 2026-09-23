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

"""Retained annotations construct complete Relax types, dataflow, and symbol identities."""

from __future__ import annotations

import tvm
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T


def test_local_annotation_retains_names_absent_from_signature():
    def build(shape, dtype):
        @R.function
        def function(x: R.Tensor((2,), "float32")):
            y: R.Tensor(shape, dtype) = R.add(x, x)
            return y

        return function

    @R.function
    def expected(x: R.Tensor((2,), "float32")):
        y: R.Tensor((2,), "float32") = R.add(x, x)
        return y

    actual = build((2,), "float32")
    tvm.ir.assert_structural_equal(
        actual.with_attr("global_symbol", ""), expected.with_attr("global_symbol", "")
    )


def test_local_annotation_and_signature_share_retained_context():
    def build(shape, dtype):
        @I.ir_module
        class Module:
            @R.function
            def main(x: R.Tensor(shape, dtype)):
                with R.dataflow():
                    y: R.Tensor(shape, dtype) = R.add(x, x)
                    R.output(y)
                return y

        return Module

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2,), "float32")):
            with R.dataflow():
                y: R.Tensor((2,), "float32") = R.add(x, x)
                R.output(y)
            return y

    tvm.ir.assert_structural_equal(build((2,), "float32"), Expected)


def test_local_annotation_preserves_explicit_parameter_and_d3_symbols():
    dtype = "float32"

    @R.function
    def function(n: R.Prim("int64"), x: R.Tensor((n, "m"), "float32")):
        m = T.int64()
        y: R.Tensor((n, m), dtype) = R.add(x, x)
        return y

    n, x = function.params
    binding = function.body.blocks[0].bindings[0]
    assert binding.var.ty.shape[0].same_as(n)
    assert binding.var.ty.shape[1].same_as(x.ty.shape[1])
    assert function.ret_ty.shape[0].same_as(n)
    assert function.ret_ty.shape[1].same_as(x.ty.shape[1])
