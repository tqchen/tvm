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
"""TIRx comparison conversion, shared effectful operands, and typed PTX predicates."""

import pytest

import tvm
from tvm import ir, tirx
from tvm.script import parser
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder

OPERATORS = [("<", "LT"), ("<=", "LE"), (">", "GT"), (">=", "GE"), ("==", "EQ"), ("!=", "NE")]


@pytest.mark.parametrize("operator,kind", OPERATORS)
def test_constexpr_preserves_host_comparison_with_iterator(operator, kind):
    class Result:
        def __bool__(self):
            raise AssertionError("custom comparison result must not be truth-tested")

        def asobject(self):
            raise AssertionError("custom comparison result must not be converted")

    result = Result()
    calls = []

    class Host:
        pass

    def compare(self, other):
        calls.append(other)
        return result

    setattr(Host, "__" + kind.lower() + "__", compare)

    def consume(value):
        assert value is result
        return 0

    right = tirx.IterVar(None, "axis", tirx.IterVar.DataPar)
    parser.parse(
        "@T.prim_func\ndef main(x: T.int32):\n"
        f"    T.evaluate(consume(T.constexpr(left {operator} right)))\n",
        extra_vars={"left": Host(), "right": right, "consume": consume},
    )
    assert len(calls) == 1
    assert calls[0] is right


def test_chain_binds_effectful_middle_operand_once():
    actual = parser.parse("""
@T.prim_func
def main():
    T.evaluate(0 < T.call_extern("int32", "middle") < 10)
""")
    value = actual.body.value
    assert isinstance(value, tirx.Let)
    assert isinstance(value.value, ir.Call)
    expected = tirx.And(tirx.LT(0, value.var), tirx.LT(value.var, 10))
    ir.assert_structural_equal(value.body, expected)


@pytest.mark.parametrize("operator,kind", OPERATORS)
def test_iterator_comparison_keeps_written_operand_order(operator, kind):
    calls = []

    def axis(value):
        calls.append(value)
        return tirx.IterVar(None, value, tirx.IterVar.DataPar)

    actual = parser.parse(
        f"@T.prim_func\ndef main(x: T.int32):\n    T.evaluate(0 {operator} axis(x))\n",
        extra_vars={"axis": axis},
    )
    assert len(calls) == 1
    assert calls[0].same_as(actual.params[0])
    ir.assert_structural_equal(actual.body.value, getattr(tirx, kind)(0, actual.params[0]))


@pytest.mark.parametrize("operator,kind", OPERATORS)
def test_unmarked_host_comparison_never_calls_python_overload(operator, kind):
    calls = []

    class Host:
        pass

    def comparison(self, other):
        calls.append(other)
        return True

    setattr(Host, "__" + kind.lower() + "__", comparison)
    decorator = "T.prim_func"
    statement = "T.evaluate(left " + operator + " right)"
    with pytest.raises(TypeError, match="(PrimExpr|primitive|convert|type)"):
        parser.parse(
            f"@{decorator}\ndef main():\n    {statement}\n",
            extra_vars={"left": Host(), "right": Host()},
        )
    assert calls == []


@pytest.mark.parametrize(
    "kind,source",
    [
        (
            "store",
            """
@T.prim_func
def main(dst: T.Buffer((2,), "uint32")):
    T.device_entry()
    tx = T.thread_id([32])
    T.ptx.st.global_.v2.b32(
        T.ptx.addr(dst.data, 0), T.uint32(1), T.uint32(2), pred=tx == 0
    )
""",
        ),
        (
            "mma",
            """
@T.prim_func
def main():
    T.device_entry()
    tx = T.thread_id([32])
    tmem = T.local_scalar("uint32")
    desc = T.local_scalar("uint64")
    idesc = T.local_scalar("uint32")
    T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
        tmem, desc, desc, idesc, 0, 0, 0, 0, tx == 0
    )
""",
        ),
    ],
)
def test_backend_predicate_materializes_equality(kind, source):
    with IRBuilder() as builder:
        with T.function():
            T.func_name("main")
            if kind == "store":
                dst = T.arg("dst", T.Buffer((2,), "uint32"))
            T.device_entry()
            tx = T.thread_id([32])
            predicate = tirx.EQ(tx, 0)
            if kind == "store":
                T.emit_(
                    T.ptx.st.global_.v2.b32(
                        T.ptx.addr(dst.data, 0), T.uint32(1), T.uint32(2), pred=predicate
                    )
                )
            else:
                tmem = T.local_scalar("uint32")
                desc = T.local_scalar("uint64")
                idesc = T.local_scalar("uint32")
                T.emit_(
                    T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
                        tmem, desc, desc, idesc, 0, 0, 0, 0, predicate
                    )
                )
    expected = builder.get()
    actual = parser.parse(source)
    tvm.ir.assert_structural_equal(expected, actual)
