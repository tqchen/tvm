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
"""Source emission uses the dialect public builder contract."""

import pytest

from tvm import ir, tirx
from tvm.relax.script import builder as R
from tvm.relax.script.builder import ir as native_R
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script import builder as tir_builder


def test_explicit_relax_emit_retains_source_binding_api():
    source = """
@R.function
def main():
    x = R.emit(R.const(1))
    return x
"""
    with IRBuilder() as context:
        with R.function():
            R.func_name("main")
            value = native_R.emit(R.const(1))
            bound = native_R.emit(value)
            R.func_ret_value(bound)
    ir.assert_structural_equal(context.get(), parser.parse(source))


@pytest.mark.parametrize("dialect", ["relax"])
def test_generated_statements_use_dialect_emit_protocol(monkeypatch, dialect):
    builder = tir_builder if dialect == "tirx" else R
    original = builder.emit_
    calls = []

    def consume(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(builder, "emit_", consume)
    source = (
        "@T.prim_func\ndef main():\n    T.evaluate(1)\n"
        if dialect == "tirx"
        else (
            "@R.function(pure=False)\ndef main(x: R.Tensor((4,), 'float32')):\n"
            "    R.print(x)\n    return x\n"
        )
    )
    function = parser.parse(source)
    assert len(calls) == 1
    if dialect == "tirx":
        assert isinstance(function.body, tirx.Evaluate)
        assert function.body.span.line == 3
    else:
        binding = function.body.blocks[0].bindings[0]
        assert binding.value.op.name == "relax.print"
        assert binding.value.span.line == 3
