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
def test_direct_producers_preserve_identity_once_and_override_mutable_targets(language, count):
    # Before: cell = X.cell(); cell, other = producer(); direct(cell).
    # Builder: ordinary Python assignment/unpacking of the direct producer's value,
    # without storing into the old cell or changing producer-owned names/spans.
    from dummy_builder import Value

    values = tuple(Value("native", name=f"producer_{i}", span=("producer",)) for i in range(count))
    calls = []

    @registry.direct_call
    def declared():
        calls.append("declared")
        return values[0] if count == 1 else values

    @registry.direct_call
    def direct(value):
        calls.append("direct")
        assert value is values[0]
        assert value.name == "producer_0" and value.span == ("producer",)
        return value

    language.X.unpack = lambda value: value
    target = "cell" if count == 1 else "cell, second"
    result = language.parse(
        f"@X.script\ndef main():\n    cell = X.cell()\n    {target} = declared()\n"
        "    kept = direct(cell)\n    direct(kept)\n",
        declared=declared,
        direct=direct,
    )
    assert calls == ["declared", "direct", "direct"]
    assert result.body == []
    assert not any(event[0] in ("bind", "set") for event in language.events)


def test_registered_global_callee_does_not_override_lexical_shadow(language):
    # Before: a local callable shadows a registered global of the same spelling.
    # Expected builder program: ordinary local call/bind semantics and one evaluation.
    calls = []

    def ordinary():
        calls.append("ordinary")
        return 7

    def operation():
        pytest.fail("shadowed global callable executed")

    registry.direct_call(operation)
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
def test_direct_producer_reuses_symbol_spelling_but_plain_write_is_rejected(
    language, ordinary_write
):
    # Before: a direct producer reuses an existing symbolic spelling.
    # Expected: direct-call precedence applies; a subsequent ordinary write remains illegal.
    value = object()

    @registry.direct_call
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


def test_result_span_contract_keeps_opaque_context_and_direct_call_exemption(language, monkeypatch):
    # Before: opaque(arg()), known_result(arg()), and kept = direct(arg()).
    # Expected builder program: only the opaque call needs .ctx; each argument
    # still has its own call context, and direct_call wins over result_span.
    from dummy_builder import RecordingSpanEntry, Value

    seen = []
    kept = Value("kept", span=("producer",))

    class Opaque:
        @property
        def __tvm_direct_call__(self):
            pytest.fail("classification must not evaluate metadata descriptors")

        @property
        def __tvm_result_span__(self):
            pytest.fail("classification must not evaluate metadata descriptors")

        def __call__(self, value, *, span=None):
            seen.append(("opaque", len(language.source_stack)))
            return Value("opaque", (value,))

    class Factory:
        @registry.result_span
        def make(self, value):
            seen.append(("result", len(language.source_stack)))
            return Value("result", (value,))

    @registry.direct_call
    @registry.result_span
    def direct(value):
        seen.append(("direct", len(language.source_stack)))
        assert value == 3
        return kept

    def argument(value):
        seen.append(("argument", value, len(language.source_stack)))
        return value

    def annotation():
        pytest.fail("a direct-call assignment does not evaluate its annotation")

    recompose = entry._recompose_builder

    def capture(program, **kwargs):
        tables = [
            (name, value)
            for name, value in kwargs["environment"].items()
            if isinstance(value, list) and value and isinstance(value[0], RecordingSpanEntry)
        ]
        assert len(tables) == 1
        name, table = tables[0]
        used = {
            node.slice.value
            for node in ast.walk(program)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == name
        }
        assert used == set(range(len(table)))
        return recompose(program, **kwargs)

    monkeypatch.setattr(entry, "_recompose_builder", capture)
    function = language.parse(
        "@X.script\ndef main():\n    opaque(argument(1))\n"
        "    known(argument(2))\n    kept: annotation() = direct(argument(3))\n",
        opaque=Opaque(),
        known=Factory().make,
        direct=direct,
        argument=argument,
        annotation=annotation,
    )
    assert seen == [
        ("argument", 1, 2),
        ("opaque", 1),
        ("argument", 2, 1),
        ("result", 0),
        ("argument", 3, 1),
        ("direct", 0),
    ]
    assert [value.op for _, value in function.body] == ["opaque", "result"]
    assert [value.span[-1][1] for _, value in function.body] == [3, 4]
    assert kept.span == ("producer",)
    assert not any(event[:2] == ("bind", "kept") for event in language.events)


def test_native_bind_direct_call_keeps_returned_and_stored_variable_identity():
    # renamed = T.bind(argument(x), var=produced): the native producer alone
    # emits its binding, and assignment preserves its explicit variable/name/span.
    from tvm import ir, tirx

    span = ir.Span(ir.SourceName("producer.py"), 7, 7, 2, 19)
    produced = ir.Var("producer_name", "int32", span)
    calls, observed = [], []

    def argument(value):
        calls.append(value)
        return value

    @registry.direct_call
    def observe(value):
        observed.append(value)
        assert value.same_as(produced) and value.span.same_as(span)
        assert value.name == "producer_name"

    function = entry.parse(
        "@T.prim_func\ndef main(x: T.int32):\n"
        "    renamed = T.bind(argument(x), var=produced)\n"
        "    observe(renamed)\n    T.evaluate(renamed)\n",
        extra_vars={"produced": produced, "argument": argument, "observe": observe},
    )
    binding, use = function.body.seq
    assert isinstance(binding, tirx.Bind) and isinstance(use, tirx.Evaluate)
    assert binding.var.same_as(produced) and use.value.same_as(produced)
    assert len(observed) == len(calls) == 1 and calls[0].same_as(function.params[0])


def test_native_view_keeps_producer_identity_name_and_span(monkeypatch):
    # renamed = captured.view(mark()); assignment must not rename, attach a
    # consumer span to, or bind the direct native producer's result.
    from functools import wraps

    from tvm import ir, tirx
    from tvm.script.ir_builder import base
    from tvm.tirx.script import builder as T

    captured = T.Buffer((4, 4), "float32")
    original = type(captured).view
    seen, produced, observed = [], [], []
    span = base.source_span(("producer.py", 7, 7, 2, 19))

    @registry.direct_call
    @wraps(original)
    def view(buffer, *args):
        seen.append("view")
        value = base.at_(span, original(buffer, *args))
        produced.append((value, value.name))
        return value

    def mark():
        seen.append("argument")
        return 16

    @registry.direct_call
    def observe(value):
        observed.append((value, value.span))

    monkeypatch.setattr(type(captured), "view", view)
    function = entry.parse(
        "@T.prim_func\ndef main(A: captured):\n"
        "    renamed = captured.view(mark())\n    observe(renamed)\n    renamed[0] = 0\n",
        extra_vars={"captured": captured, "mark": mark, "observe": observe},
    )
    assert seen == ["argument", "view"]
    value, name = produced[0]
    assert len(produced) == len(observed) == 1
    assert name != "renamed" and value.name == name
    assert observed[0][0].same_as(value) and observed[0][1].same_as(span)
    nodes = list(function.body.seq)
    assert len(nodes) == 2 and not any(isinstance(node, tirx.Bind) for node in nodes)
    assert nodes[0].buffer.same_as(value) and nodes[1].buffer.same_as(value)
    ir.assert_structural_equal(nodes[0].data, captured.data)


def test_native_concise_scopes_unwind_with_their_parent():
    # bx = launch_thread(...); tx = launch_thread(...); evaluate(bx + tx).
    # bind_ enters native children; parent callbacks close nested scopes in order.
    from tvm import tirx
    from tvm.script.ir_builder import IRBuilder
    from tvm.tirx.script import builder as T

    with IRBuilder() as builder:
        with T.function() as parent:
            T.func_name("main")
            block = T.launch_thread("blockIdx.x", 2)
            assert builder.frames[-1].same_as(parent)
            bx = T.bind_(block, name="bx")
            thread = T.launch_thread("threadIdx.x", 32)
            tx = T.bind_(thread, name="tx")
            assert builder.frames[-1].same_as(thread)
            T.evaluate(bx + tx)
        assert not builder.frames
        # Callback storage is native and not exposed as Python frame state;
        # empty active frames and the nested result establish actual unwinding.
    body = builder.get().body
    assert isinstance(body, tirx.AttrStmt) and isinstance(body.body, tirx.AttrStmt)
    assert body.node.var.same_as(bx) and body.body.node.var.same_as(tx)
    assert body.body.body.value.a.same_as(bx) and body.body.body.value.b.same_as(tx)


@pytest.mark.parametrize("control", ["break", "continue"])
def test_native_loop_control_is_checked_only_after_construction(monkeypatch, control):
    # Before: break/continue outside a loop, with validation disabled or enabled.
    # Expected builder program: construct the same native IR in both cases;
    # the final native hook rejects invalid placement after the function frame exits.
    from tvm import ir, tirx
    from tvm.script.ir_builder import IRBuilder
    from tvm.tirx.script import builder as T

    check = T.check_well_formed_
    checked = []

    def check_completed(function):
        assert not IRBuilder.is_in_scope()
        checked.append(function)
        return check(function)

    monkeypatch.setattr(T, "check_well_formed_", check_completed)
    source = f"@T.prim_func\ndef main():\n    {control}\n"
    invalid = entry.parse(source, check_well_formed=False)
    assert isinstance(invalid.body, tirx.Evaluate)
    assert invalid.body.value.op.same_as(getattr(tirx, control + "_loop")().op)
    assert checked == []
    with pytest.raises(ValueError, match="requires an enclosing loop"):
        entry.parse(source)
    assert len(checked) == 1
    ir.assert_structural_equal(checked[0], invalid)
    valid = entry.parse(f"@T.prim_func\ndef main():\n    for i in range(2):\n        {control}\n")
    assert isinstance(valid.body, tirx.For)
    assert valid.body.body.value.op.same_as(invalid.body.value.op)
    assert len(checked) == 2 and checked[-1] is valid
