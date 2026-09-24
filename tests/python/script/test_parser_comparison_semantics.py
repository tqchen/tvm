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
"""Source comparison semantics using recording frames and common primitive IR nodes."""

from functools import reduce

import pytest
import tvm_ffi

from tvm import ir
from tvm.ir import prim
from tvm.ir._overload_prim_expr import EqualOp
from tvm.script.ir_builder import base
from tvm.script.parser import entry

OPERATORS = [("<", "LT"), ("<=", "LE"), (">", "GT"), (">=", "GE"), ("==", "EQ"), ("!=", "NE")]


@pytest.fixture
def language(language, monkeypatch):
    def span_entry(span):
        language.events.append(("span_entry", span))
        return base.SpanEntry(span)

    monkeypatch.setattr(entry, "SpanEntry", span_entry)
    language.I.at_ = base.at
    language.I.with_at_group_ = base.with_at_group_
    for _, kind in OPERATORS:
        constructor = getattr(prim._ffi_api, "_Op" + kind)
        setattr(
            language.X, kind.lower() + "_", lambda lhs, rhs, make=constructor: make(lhs, rhs, None)
        )

    def conjunction(*conditions):
        return reduce(lambda rhs, lhs: prim.And(lhs, rhs), reversed(conditions))

    language.X.and_ = conjunction
    return language


def parse(language, source, *, track_span=True, **captures):
    return entry.parse(
        source,
        extra_vars={"X": language.X, **captures},
        filename="comparison.py",
        track_span=track_span,
        root_builder=language.X,
    )


@pytest.mark.parametrize("operator,kind", OPERATORS)
@pytest.mark.parametrize("literal_left", [True, False])
@pytest.mark.parametrize("track_span", [True, False])
def test_written_comparison_order(language, operator, kind, literal_left, track_span):
    # Before: 0 < x (and each written operator/operand order).
    # Expected builder program: X.emit_(X.lt_(0, x), span=_S[i]).
    x = ir.Var("x", "int32")
    expression = f"0 {operator} x" if literal_left else f"x {operator} 0"
    actual = parse(
        language,
        f"@X.script\ndef main():\n    {expression}\n",
        x=x,
        track_span=track_span,
    ).body[0][1]
    operands = (0, x) if literal_left else (x, 0)
    ir.assert_structural_equal(actual, getattr(prim, kind)(*operands))
    entries = [event[1] for event in language.events if event[0] == "span_entry"]
    if track_span:
        assert actual.span is not None
        assert entries
        # The written comparison owns the expression range, not either name/literal.
        body_entries = [span for span in entries if span.line == span.end_line == 3]
        assert body_entries
        assert all(
            (span.column, span.end_column) == (5, 5 + len(expression)) for span in body_entries
        )
    else:
        assert actual.span is None
        assert entries == []


@pytest.mark.parametrize("dtype", ["int64", "uint32", "float32", "int32x4", "float32x4"])
def test_literal_uses_ir_operand_type_and_lanes(language, dtype):
    # Before: 0 < x, where x carries the scalar or vector dtype.
    # Expected builder program: X.lt_(0, x); native IR performs literal promotion/broadcast.
    x = ir.Var("x", dtype)
    value = parse(language, "@X.script\ndef main():\n    0 < x\n", x=x).body[0][1]
    assert isinstance(value, prim.LT)
    assert str(value.a.ty) == str(value.b.ty) == dtype
    assert value.b.same_as(x)


@pytest.mark.parametrize("operator,kind", OPERATORS)
@pytest.mark.parametrize("other_kind", ["host", "primitive"])
def test_custom_host_comparison_result_is_preserved(language, operator, kind, other_kind):
    # Before: consume(X.constexpr(left < right)).
    # Expected builder program: consume(left < right), preserving the Python overload result.
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

    right = ir.Var("x", "int32") if other_kind == "primitive" else object()
    parse(
        language,
        f"@X.script\ndef main():\n    consume(X.constexpr(left {operator} right))\n",
        left=Host(),
        right=right,
        consume=consume,
    )
    assert len(calls) == 1
    if other_kind == "primitive":
        assert calls[0].same_as(right)
    else:
        assert calls[0] is right


def test_numeric_subclass_keeps_custom_comparison(language):
    seen = []
    result = object()

    class HostInt(int):
        def __lt__(self, other):
            seen.append(other)
            return result

    def consume(value):
        assert value is result
        return 0

    parse(
        language,
        "@X.script\ndef main():\n    consume(X.constexpr(left < x))\n",
        x=ir.Var("x", "int32"),
        left=HostInt(0),
        consume=consume,
    )
    assert len(seen) == 1


def test_host_ordering_does_not_materialize_an_equality_result(language):
    result = ir.Var("host_result", "int32") == 0

    class Host:
        def __lt__(self, other):
            return result

    def consume(value):
        assert value is result
        return 0

    parse(
        language,
        "@X.script\ndef main():\n    consume(X.constexpr(left < right))\n",
        left=Host(),
        right=Host(),
        consume=consume,
    )


def test_simple_chain_and_complex_operand_rejection(language):
    # Simple adjacent comparisons lower directly; complex chains fail before
    # operand evaluation. The explicit constexpr boundary stays ordinary Python.
    x, y = ir.Var("x", "int32"), ir.Var("y", "int32")
    actual = parse(
        language,
        "@X.script\ndef main():\n    -1 < x <= y != +3\n",
        x=x,
        y=y,
    ).body[0][1]
    expected = prim.And(prim.LT(-1, x), prim.And(prim.LE(x, y), prim.NE(y, 3)))
    ir.assert_structural_equal(actual, expected)
    assert actual.span.line == 3

    def operand():
        pytest.fail("unsupported chain evaluated its operand")

    class Holder:
        @property
        def value(self):
            return operand()

        def __getitem__(self, index):
            return operand()

    for expression in (
        "operand() < x < y",
        "x < holder.value < y",
        "x < holder[0] < y",
        "x < x + 1 < y",
    ):
        with pytest.raises(SyntaxError, match="chain") as caught:
            parse(
                language,
                f"@X.script\ndef main():\n    {expression}\n",
                operand=operand,
                holder=Holder(),
                x=x,
                y=y,
            )
        assert caught.value.filename == "comparison.py" and caught.value.lineno == 3
        assert caught.value.offset >= 5


def test_constexpr_keeps_python_comparison_and_chain_short_circuit(language):
    # Before: if X.constexpr(a() < b() < invalid()): ...
    # Expected builder program: ordinary Python if/chain; the last operand stays lazy.
    seen = []

    def operand(index, value):
        seen.append(index)
        return value

    def invalid():
        raise AssertionError("constexpr comparison chain must short-circuit")

    actual = parse(
        language,
        """
@X.script
def main():
    if X.constexpr(operand(0, 2) < operand(1, 1) < invalid()):
        0
    elif X.constexpr(x == x):
        1
""",
        operand=operand,
        invalid=invalid,
        x=ir.Var("x", "int32"),
    )
    assert seen == [0, 1]
    assert actual.body == [("emit", 1)]


@pytest.mark.parametrize("operator,kind", OPERATORS)
@pytest.mark.parametrize("dtype", ["int64", "float32", "int64x4"])
def test_comparisons_use_native_promotion_and_broadcast(language, operator, kind, dtype):
    # Before: x < y (and each comparison), with distinct primitive types.
    # Expected builder program: X.lt_(x, y); native IR owns the cast and broadcast.
    x, y = ir.Var("x", "int32"), ir.Var("y", dtype)
    actual = parse(
        language,
        f"@X.script\ndef main():\n    x {operator} y\n",
        x=x,
        y=y,
    ).body[0][1]
    lhs = prim.Broadcast(prim.Cast("int64", x), 4) if dtype.endswith("x4") else prim.Cast(dtype, x)
    ir.assert_structural_equal(actual, getattr(prim, kind)(lhs, y))


@pytest.mark.parametrize("operator,kind", [("==", prim.EQ), ("!=", prim.NE)])
@pytest.mark.parametrize("track_span", [True, False])
def test_symbolic_equality_reaches_typed_consumer(language, operator, kind, track_span):
    # Before: consume(x == 0), with source tracking enabled or disabled.
    # Expected builder program: consume(X.eq_(x, 0)), with the original comparison location.
    seen = []

    def consume(value):
        assert isinstance(value, kind)
        assert str(value.ty) == "bool"
        seen.append(value)
        return value

    x = ir.Var("x", "int32")
    parse(
        language,
        f"@X.script\ndef main():\n    consume(x {operator} 0)\n",
        consume=consume,
        x=x,
        track_span=track_span,
    )
    assert len(seen) == 1
    assert seen[0].a.same_as(x)
    if track_span:
        assert seen[0].span is not None
    else:
        assert seen[0].span is None


def test_python_comparisons_remain_python_booleans(language):
    seen = []

    def consume(value):
        assert type(value) is bool
        seen.append(value)
        return int(value)

    parse(
        language,
        """
@X.script
def main():
    consume(X.constexpr([1, 2] == [1, 2]))
    consume(X.constexpr("x" != "y"))
""",
        consume=consume,
    )
    assert seen == [True, True]


def test_host_equality_result_is_not_arbitrarily_converted(language):
    class HostResult(tvm_ffi.ObjectConvertible):
        def asobject(self):
            raise AssertionError("host comparison result must not be converted")

    result = HostResult()

    class Host:
        def __eq__(self, other):
            return result

    seen = []

    def consume(value):
        assert value is result
        seen.append(value)
        return 0

    source = "@X.script\ndef main():\n    consume(X.constexpr(left == right))\n"
    parse(language, source, left=Host(), right=Host(), consume=consume)
    assert seen == [result]
    assert seen[0] is result

    class CustomEquality(EqualOp):
        def asobject(self):
            raise AssertionError("custom equality subclasses are host values")

    custom = CustomEquality(ir.Var("x", "int32"), 0)
    result = custom
    parse(language, source, left=Host(), right=Host(), consume=consume)
    assert seen[-1] is custom


def test_comparison_operands_evaluate_once_in_order(language):
    # Before: consume(a() == b()).
    # Expected builder program: evaluate operands once, then pass concrete EQ/And IR to consume.
    seen = []

    def operand(index, value):
        seen.append(index)
        return value

    def consume(value):
        assert isinstance(value, ir.Expr)
        assert str(value.ty) == "bool"
        return value

    expression = "operand(0, value0) == operand(1, value1)"
    parse(
        language,
        "@X.script\ndef main():\n" + f"    consume({expression})\n",
        operand=operand,
        consume=consume,
        **{f"value{i}": ir.Var(f"value{i}", "int32") for i in range(2)},
    )
    assert seen == [0, 1]
