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
"""Shared semantic boundaries for the explicit parser/builder contract."""

from __future__ import annotations

import ast

import pytest

from tvm.script import ir as I
from tvm.script.parser import entry
from tvm.script.parser import protocol_registry as registry


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_validation_hook_runs_after_complete_construction(language, module, enabled):
    # Before: a standalone function or two-member module, with validation enabled/disabled.
    # Expected builder program: complete all bodies/frames, then X/I.check_well_formed_.
    seen = []

    def check(result):
        assert language.stack == []
        if module:
            assert set(result) == {"first", "second"}
            assert result["first"].body == [("emit", 1)]
            assert result["second"].body == [("emit", 2)]
        else:
            assert result.body == [("emit", 1)]
        seen.append(result)

    language.I.check_well_formed_ = check if module else lambda result: pytest.fail("module hook")
    language.X.check_well_formed_ = (
        check if not module else lambda result: pytest.fail("function hook")
    )
    source = (
        "@I.ir_module\nclass M:\n    @X.script\n    def first():\n        X.record(1)\n"
        "    @X.script\n    def second():\n        X.record(2)\n"
        if module
        else "@X.script\ndef first():\n    X.record(1)\n"
    )
    result = entry.parse(source, {"X": language.X}, check_well_formed=enabled)
    assert seen == ([result] if enabled else [])


def test_pyfunc_collection_preserves_original_callable_and_closure(language):
    # Before: I.pyfunc marks an ordinary closure in a class.
    # Expected builder program: attach the original callable to __pyfuncs__; no IR declaration.
    factor = 7
    calls = []

    class Source:
        @I.pyfunc
        def multiply(value):
            calls.append(value)
            return value * factor

    original = Source.multiply
    result = entry.ir_module(Source)
    assert result.__pyfuncs__ == {"multiply": original}
    assert result == {}
    assert calls == []
    assert result.__pyfuncs__["multiply"](3) == 21
    assert calls == [3]
    assert not any(event[0] in ("arg", "name") for event in language.events)


@pytest.mark.parametrize(
    "source, introduction, offending",
    [
        ("@X.script\ndef main(x: X.tensor((n,))):\n    n = 2\n", 2, 3),
        ("@X.script\ndef main():\n    n = X.symbol()\n    n = 2\n", 3, 4),
        (
            "@X.script\ndef main():\n    n = X.symbol()\n"
            "    if X.constexpr(True):\n        n = 2\n",
            3,
            5,
        ),
        ("@X.script\ndef main():\n    n = X.symbol()\n    n += 1\n", 3, 4),
    ],
)
def test_symbol_reassignment_reports_introduction_and_exact_write(
    language, source, introduction, offending
):
    # Before: annotation-free/body-declared n, followed by an ordinary write.
    # Expected diagnostic: n, its introduction line, and the actual offending source target.
    with pytest.raises(SyntaxError) as caught:
        language.parse(source, n=4)
    message = str(caught.value)
    assert "Symbolic variable 'n' cannot be reassigned" in message
    assert f"introduced at line {introduction}" in message
    assert caught.value.filename == "dummy.py"
    cause = caught.value
    assert isinstance(cause, SyntaxError)
    assert (cause.lineno, cause.end_lineno) == (offending, offending)
    assert cause.end_offset - cause.offset == 1


def test_repeated_symbol_declarations_reuse_identity(language):
    # Before: annotation n; n = X.symbol(); n = X.symbol().
    # Expected builder program: both declarations resolve the same function-owned symbol.
    result = language.parse(
        "@X.script\ndef main(x: X.tensor(('n',))):\n"
        "    n = X.symbol()\n    X.record(n)\n    n = X.symbol()\n    X.record(n)\n"
    )
    symbol = result.params[0].args[0].args[0][0]
    assert result.body == [("emit", symbol), ("emit", symbol)]
    assert result.body[0][1] is result.body[1][1]


def test_mutable_loop_and_ordinary_names_remain_assignable(language):
    # Before: cell declaration/update, loop-variable rebinding, ordinary rebinding.
    # Expected builder program: set_mutable_var_ for cell, ordinary bind_ for i/value.
    result = language.parse(
        "@X.script\ndef main():\n    cell = X.cell(0)\n    cell = 1\n"
        "    value = 2\n    value = 3\n    for i in X.grid(4):\n"
        "        i = 5\n        X.record(i + value)\n"
    )
    assert result.body == [("emit", 8)]
    assert len([event for event in language.events if event[0] == "set"]) == 1


def test_nested_scope_does_not_reassign_outer_symbol(language):
    # Before: outer n is symbolic; nested function has an independent ordinary n.
    # Expected builder program: each source function has its own declaration/alias context.
    result = language.parse(
        "@X.script\ndef main():\n    n = X.symbol()\n"
        "    @X.script\n    def nested():\n        n = 2\n        X.record(n)\n    X.record(n)\n"
    )
    assert result.body[0][1].op == "symbol"
    assert language.functions["nested"].body == [("emit", 2)]


@pytest.mark.parametrize("count", [1, 2])
def test_direct_and_scope_declaration_calls_preserve_identity_once(language, count):
    # Before: variable(s) = declared(); direct(value); variable = direct(value).
    # Expected builder program: query/declaration naming only, direct calls bypass bind/emit.
    values = tuple(object() for _ in range(count))
    calls = []

    @registry.register_scope_var_query_or_decl
    def declared():
        calls.append("declared")
        return values[0] if count == 1 else values

    @registry.direct_call
    def direct(value):
        calls.append("direct")
        return value

    language.X.unpack = lambda value: value
    target = "first" if count == 1 else "first, second"
    result = language.parse(
        f"@X.script\ndef main():\n    {target} = declared()\n"
        "    kept = direct(first)\n    direct(kept)\n    X.record(kept)\n",
        declared=declared,
        direct=direct,
    )
    assert calls == ["declared", "direct", "direct"]
    assert result.body == [("emit", values[0])]
    assert not any(event[0] == "bind" for event in language.events)


@pytest.mark.parametrize("category", ["direct", "scope"])
def test_registered_global_callee_does_not_override_lexical_shadow(language, category):
    # Before: a local callable shadows a registered global of the same spelling.
    # Expected builder program: ordinary local call/bind semantics and one evaluation.
    calls = []

    def ordinary():
        calls.append("ordinary")
        return 7

    def operation():
        pytest.fail("shadowed global callable executed")

    if category == "direct":
        registry.direct_call(operation)
    else:
        registry.register_scope_var_query_or_decl(operation)
    result = language.parse(
        "@X.script\ndef main():\n    operation = ordinary\n"
        "    value = operation()\n    X.record(value)\n",
        operation=operation,
        ordinary=ordinary,
    )
    assert calls == ["ordinary"]
    assert result.body == [("emit", 7)]
    assert any(event[:2] == ("bind", "value") for event in language.events)


def test_direct_call_keeps_nested_expression_rewrites_and_call_scopes(language):
    # Before: direct(left() < right()); only the outer call has direct policy.
    # Expected builder program: normal comparison lowering and one scope per nested call.
    calls = []

    def operand(value):
        calls.append((value, len(language.source_stack)))
        return value

    @registry.direct_call
    def direct(value):
        calls.append(("direct", len(language.source_stack)))
        return value

    result = language.parse(
        "@X.script\ndef main():\n    kept = direct(operand(1) < operand(2))\n    X.record(kept)\n",
        operand=operand,
        direct=direct,
    )
    assert calls == [(1, 1), (2, 1), ("direct", 0)]
    comparison = result.body[0][1]
    assert comparison.op == "lt" and comparison.args == (1, 2)
    assert not any(event[:2] == ("bind", "kept") for event in language.events)


def test_body_annotation_reads_a_preceding_ordinary_local(language):
    # Before: shape is constructed in the function before a local annotation uses it.
    # Expected builder program: the annotation reads that local shape, not an absent capture.
    bind = language.X.bind_
    annotations = []

    def observe(value, **kwargs):
        if "ty" in kwargs:
            annotations.append(kwargs["ty"])
        return bind(value, **kwargs)

    language.X.bind_ = observe
    result = language.parse(
        "@X.script\ndef main():\n    shape = (4,)\n"
        "    value: X.tensor(shape) = 1\n    X.record(value)\n"
        "    shape = (8,)\n    X.record(shape)\n"
    )
    assert len(annotations) == 1 and annotations[0].args[0] == (4,)
    assert result.body == [("emit", 1), ("emit", (8,))]


@pytest.mark.parametrize("ordinary_write", [False, True])
def test_scope_declaration_reuses_symbol_spelling_but_plain_write_is_rejected(
    language, ordinary_write
):
    # Before: an explicit scope declaration reuses an existing symbolic spelling.
    # Expected: declaration precedence applies; a subsequent ordinary write remains illegal.
    value = object()

    @registry.register_scope_var_query_or_decl
    def declared():
        return value

    source = "@X.script\ndef main():\n    n = X.symbol()\n    n = declared()\n"
    if ordinary_write:
        source += "    n = 3\n"
        with pytest.raises(SyntaxError, match="Symbolic variable 'n' cannot be reassigned"):
            language.parse(source, declared=declared)
    else:
        result = language.parse(source + "    X.record(n)\n", declared=declared)
        assert result.body == [("emit", value)]
        assert not any(event[:2] == ("bind", "n") for event in language.events)


@pytest.mark.parametrize(
    "source, node_type, message",
    [
        ("@X.script\ndef main(value):\n    pass\n", ast.arg, "requires an annotation"),
        (
            '@X.script\ndef main(value: "invalid +"):\n    pass\n',
            ast.Constant,
            "Invalid annotation expression",
        ),
        (
            "@X.script\ndef main():\n    X.record((value :=\n        1))\n",
            ast.NamedExpr,
            "Unsupported expression: NamedExpr",
        ),
    ],
)
def test_transpilation_restrictions_keep_original_source_ranges(
    language, source, node_type, message
):
    # Before: invalid signature, quoted annotation or multiline expression syntax.
    # Expected: direct SyntaxError points at the original node before builder execution.
    expected = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, node_type))
    with pytest.raises(SyntaxError, match=message) as caught:
        entry.parse(source, extra_vars={"X": language.X}, filename="restriction.py")
    error = caught.value
    assert type(error) is SyntaxError
    assert (error.filename, error.lineno, error.offset, error.end_lineno, error.end_offset) == (
        "restriction.py",
        expected.lineno,
        expected.col_offset + 1,
        expected.end_lineno,
        expected.end_col_offset + 1,
    )
    assert not language.events
