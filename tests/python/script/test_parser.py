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
"""Observable source-to-builder contracts, exercised through the real parser."""

from __future__ import annotations

import ast

import pytest
from dummy_builder import Value

from tvm.script.parser import entry
from tvm.script.parser import protocol_registry as registry


@pytest.mark.parametrize(
    "arguments",
    [
        '("n + 1", "n"), "float32", "cuda:1"',
        'shape=("n + 1", "n"), dtype="float32", device="cuda:1"',
        'shape=["n + 1", "n"], dtype="float32", device="cuda:1"',
    ],
)
@pytest.mark.parametrize("quoted", [False, True])
def test_argument_policies_reuse_symbols_and_resolve_only_marked_literals(
    language, arguments, quoted
):
    # Before: x: X.tensor(("n + 1", "n"), "float32", "cuda:1")
    # Expected builder program:
    # X.arg("x", X.tensor((X.resolve_type_var_("n") + 1, X.resolve_type_var_("n")),
    #                     "float32", I.resolve_global_info_("cuda:1")))
    device = object()
    language.global_infos["cuda:1"] = device
    annotation = f"X.tensor({arguments})"
    if quoted:
        annotation = repr(annotation)
    result = language.parse(f"""
@X.script
def main(x: {annotation}):
    X.record(x)
""")
    annotation = result.params[0].args[0]
    shape, dtype, resolved, placement = annotation.args
    assert shape[0].op == "add" and shape[0].args[0] is shape[1]
    assert shape[0].args[1] == 1
    assert (dtype, placement) == ("float32", "S[0]")
    assert resolved is device
    assert [item for item in language.events if item[0] == "global_info"] == [
        ("global_info", "cuda:1")
    ]
    assert ("record", result.params[0]) in language.events


def test_policies_leave_computed_arguments_and_shorthand_strings_alone(language):
    # Before: X.tensor(shape(), device=device()); X.tensor("float32")
    # Expected builder program: X.tensor(shape(), device=device()); X.tensor("float32")
    # Policies never inspect a computed argument to reinterpret its strings.
    seen = []
    concrete = object()

    def value(label, result):
        seen.append(label)
        return result

    result = language.parse(
        """
@X.script
def main():
    X.tensor(value("shape", (4,)), dtype=value("dtype", "float32"),
             device=value("device", concrete))
    X.tensor("float32", placement="literal[0]")
""",
        value=value,
        concrete=concrete,
    )
    assert seen == ["shape", "dtype", "device"]
    assert not any(item[0] == "global_info" for item in language.events)
    assert result.body[0][1].args == ((4,), "float32", concrete, "S[0]")
    assert result.body[1][1].args == ("float32", "float32", None, "literal[0]")


def test_nested_policy_and_starred_calls_are_evaluated_once(language):
    # Before: outer(X.tensor(("n",), device="mesh[0]")); collect(*values, other=3)
    # Expected builder program: outer(X.tensor((X.resolve_type_var_("n"),),
    #                             device=I.resolve_global_info_("mesh[0]")))
    seen = []
    mesh = object()
    language.global_infos["mesh[0]"] = mesh

    def outer(value):
        seen.append(value)
        return value

    def collect(*values, other):
        return values, other

    result = language.parse(
        """
@X.script
def main():
    outer(X.tensor(("n",), device="mesh[0]"))
    collect(*values, other=3)
""",
        outer=outer,
        collect=collect,
        values=(1, 2),
    )
    assert len(seen) == 1 and seen[0].args[2] is mesh
    assert result.body[1][1] == ((1, 2), 3)
    assert [item for item in language.events if item[0] == "global_info"] == [
        ("global_info", "mesh[0]")
    ]


@pytest.mark.parametrize("marker", ["I.constexpr", "X.constexpr"])
def test_constexpr_is_lazy_and_executes_in_parent_scope(language, marker):
    # Before: if I.constexpr(choose()): x = 7; X.record(x)
    # Expected builder program: if choose(): x = X.bind_(7, name="x"); X.emit_(X.record(x))
    seen = []

    def choose():
        seen.append("condition")
        return True

    result = language.parse(
        f"""
@X.script
def main():
    if {marker}(choose()):
        x = 7
    else:
        invalid()
    X.record(x)
    X.record(1 if {marker}(True) else invalid())
    X.record({marker}(0) and invalid())
    X.record({marker}(4) or invalid())
""",
        choose=choose,
    )
    assert seen == ["condition"]
    assert [value for kind, value in result.body] == [7, 1, 0, 4]


def test_unmarked_expressions_build_both_arms_in_source_order(language):
    # Before: first() if condition else second(); condition and last()
    # Expected builder program:
    # X.if_then_else_(condition, first(), second()); X.and_(condition, last())
    seen = []

    def operand(value):
        seen.append(value)
        return value

    condition = Value("condition")
    result = language.parse(
        """
@X.script
def main():
    operand(1) if condition else operand(2)
    condition and operand(3)
    condition or operand(4)
""",
        condition=condition,
        operand=operand,
    )
    assert seen == [1, 2, 3, 4]
    assert [value.op for _, value in result.body] == ["select", "and", "or"]
    assert all(value.args[0] is condition for _, value in result.body)


def test_callee_arguments_and_keywords_evaluate_once_with_caller_context(language):
    # Before: callee()(operand(1), b=operand(2))
    # Expected builder program:
    # X.emit_(_S[i].ctx(lambda: callee()(operand(1), b=operand(2))), span=_S[i])
    seen = []
    result_value = Value("returned")

    def callee():
        seen.append("callee")

        def function(a, *, b):
            seen.append((a, b))
            assert language.source_stack
            return result_value

        return function

    def operand(value):
        seen.append(value)
        return value

    result = language.parse(
        """
@X.script
def main():
    callee()(operand(1), b=operand(2))
""",
        callee=callee,
        operand=operand,
    )
    assert seen == ["callee", 1, 2, (1, 2)]
    assert result.body[0][1] is result_value
    assert result_value.span[-1][1] == 4
    assert language.source_stack == []


def test_binding_and_mutation_have_distinct_operations(language):
    # Before: n = X.symbol(); x = helper(); cell = X.cell(); cell = x
    # Expected builder program:
    # n = X.resolve_type_var_("n", "int64"); x = X.bind_(helper(), name="x")
    # cell = X.decl_mutable_cell_(X.cell(), name="cell"); X.set_mutable_cell_(cell, x)
    from dummy_builder import Value

    marker = Value("producer", span=("producer",))
    binds_with_spans, calls = [], []
    bind = language.X.bind_

    def observe_binding(value, *, span, value_span, **kwargs):
        assert value is marker and value.span == ("producer",)
        binds_with_spans.append((span.span, value_span.span))
        return bind(value, **kwargs)

    def helper():
        calls.append(True)
        assert len(language.source_stack) == 1
        assert language.source_stack[-1][1:] == (5, 5, 9, 17)
        return marker

    language.X.bind_ = observe_binding
    result = language.parse(
        """
@X.script
def main():
    n = X.symbol()
    x = helper()
    cell = X.cell()
    cell = x
    X.record(x)
""",
        helper=helper,
    )
    assert result.body[-1][1] is marker
    binds = [event for event in language.events if event[0] == "bind"]
    assert [(event[1], event[2]) for event in binds] == [("x", marker)]
    declaration = next(event for event in language.events if event[0] == "declare")
    mutation = next(event for event in language.events if event[0] == "set")
    assert declaration[1] == "cell" and mutation[1] is declaration[2]
    assert mutation[2] is marker
    assert calls == [True] and len(binds_with_spans) == 1
    target, rhs = binds_with_spans[0]
    assert target.line == rhs.line == 5
    assert (target.column, target.end_column) == (5, 6)
    assert (rhs.column, rhs.end_column) == (9, 17)


def test_ordinary_tuple_and_outer_branch_assignments_still_store(language):
    # Before: a = X.cell(); b = X.cell(); a, b = values(); if cond: a = first
    # Expected builder program: declare a/b; unpack values once; set a/b;
    # with X.Then(): def branch(): X.set_mutable_cell_(a, first); branch()
    # Named += uses bind_ for ordinary x and set_mutable_cell_ for declared b.
    first, second = object(), object()
    calls = []

    def values():
        calls.append("values")
        return first, second

    language.X.unpack = lambda value: value
    result = language.parse(
        """
@X.script
def main():
    a = X.cell()
    b = X.cell()
    a, b = values()
    if X.value():
        a = first
    else:
        a = second
    x = 1
    x += 2
    b += 3
    X.record(x)
""",
        values=values,
        first=first,
        second=second,
    )
    declarations = {event[1]: event[2] for event in language.events if event[0] == "declare"}
    stores = [(event[1], event[2]) for event in language.events if event[0] == "set"]
    assert calls == ["values"]
    assert stores[:-1] == [
        (declarations["a"], first),
        (declarations["b"], second),
        (declarations["a"], first),
        (declarations["a"], second),
    ]

    assert [event[2] for event in language.events if event[:2] == ("bind", "x")] == [1, 3]
    assert result.body[-1] == ("emit", 3)
    target, addition = stores[-1]
    assert target is declarations["b"]
    assert addition.op == "add" and addition.args == (target, 3)


@pytest.mark.parametrize("scope", ["local", "parameter", "enclosing"])
def test_ordinary_callable_aliases_update_mutable_targets(language, scope):
    # Before: axis_alias = ordinary; cell = X.cell(); cell = axis_alias()
    # Expected builder program: X.set_mutable_cell_(cell, axis_alias()).
    # Local, parameter and enclosing Python callables all preserve ordinary stores.
    marker, calls = object(), []

    def axis_alias():
        raise AssertionError("The shadowed ambient callable must not run")

    def ordinary():
        calls.append("ordinary")
        return marker

    body = "cell = X.cell()\n    cell = axis_alias()\n    X.record(cell)"
    if scope == "local":
        source = "@X.script\ndef main():\n    axis_alias = ordinary\n    " + body
    elif scope == "parameter":
        # A callable parameter is an opaque value supplied by the fake frame.
        def arg(name, annotation, **kwargs):
            frame = language.frame()
            frame.params.append(ordinary)
            frame.function.params.append(ordinary)
            return ordinary

        language.X.arg = arg
        source = "@X.script\ndef main(axis_alias: X.tensor(())):\n    " + body
    else:
        source = (
            "@X.script\ndef main():\n    axis_alias = ordinary\n"
            "    @X.script\n    def inner():\n        " + body.replace("\n", "\n    ")
        )
    language.parse(source, axis_alias=axis_alias, ordinary=ordinary)
    declaration = next(event[2] for event in language.events if event[0] == "declare")
    stores = [event for event in language.events if event[0] == "set"]
    assert calls == ["ordinary"]
    assert len(stores) == 1 and stores[0][1] is declaration and stores[0][2] is marker
    assert next(event[1] for event in language.events if event[0] == "record") is declaration


def test_quoted_symbols_share_identity_without_introducing_python_bindings(language):
    # Before: def main(x: X.tensor(("n", "n"))): X.record(n)
    # Expected builder program: X.tensor((X.resolve_type_var_("n"), X.resolve_type_var_("n")))
    # The unquoted body name n raises NameError until explicitly declared.
    with pytest.raises(NameError) as error:
        language.parse("""
@X.script
def main(x: X.tensor(("n", "n"))):
    X.record(n)
""")
    assert isinstance(error.value, NameError)


@pytest.mark.parametrize(
    "binding", ["X = 1", "for X in range(2):\n        pass", "with context() as X:\n        pass"]
)
def test_script_namespace_cannot_be_rebound(language, binding):
    # Before: X = 1  (also loop/with targets)
    # Expected builder program: reject a binding that shadows the fixed script namespace X.
    with pytest.raises(SyntaxError, match="namespace|shadow|rebind"):
        language.parse(f"@X.script\ndef main():\n    {binding}\n")


def test_generated_names_do_not_capture_source_bindings(language):
    # Before: _builder = 4; _frame = 5; X.record(_builder + _frame)
    # Expected builder program: generated I/X/frame aliases avoid all source identifiers.
    result = language.parse("""
@X.script
def main():
    _builder = 4
    _frame = 5
    _tvm_0 = 6
    X.record(_builder + _frame + _tvm_0)
""")
    assert result.body[0][1] == 15


@pytest.mark.parametrize(
    "target, bounds, arguments, names",
    [
        ("i", "4", "i", ["i"]),
        ("(i,)", "4", "i", ["i"]),
        ("[i]", "4", "i", ["i"]),
        ("i, j", "4, 5", "i, j", ["i", "j"]),
        ("i, *tail", "4, 5, 6", "i, *tail", ["i", "tail_0", "tail_1"]),
    ],
)
def test_loop_targets_configure_the_entered_frame(language, target, bounds, arguments, names):
    # Scalar: with X.for_(...) as i. Sequence: loop = X.for_(...);
    # with loop: original_target = loop.vars. Construct once and unpack stable vars.
    grid = language.X.grid
    frames, observed = [], []

    def construct(*extents):
        frame = grid(*extents)
        frames.append((frame, tuple(frame.vars)))
        return frame

    def observe(*values):
        frame, original = frames[-1]
        assert language.stack[-1] is frame
        assert len(values) == len(original)
        assert all(value is old for value, old in zip(values, original))
        assert [value.name for value in values] == names
        observed.append(values)

    language.X.grid = construct
    source = (
        f"@X.script\ndef main():\n    for {target} in X.grid({bounds}):\n"
        f"        observe({arguments})\n"
    )
    language.parse(source, observe=observe)
    assert len(frames) == len(observed) == 1
    assert language.stack == [] and language.source_stack == []
    if target == "i":
        # A bare target collects the original multi-dimensional entry sequence.
        names = ["iters_0", "iters_1"]
        language.parse(
            "@X.script\ndef main():\n    for iters in X.grid(4, 5):\n        observe(*iters)\n",
            observe=observe,
        )
        assert len(frames) == len(observed) == 2
        assert language.stack == [] and language.source_stack == []
    if target == "(i,)":
        failure = ValueError("loop body failure")

        def fail(*values):
            observe(*values)
            raise failure

        with pytest.raises(ValueError) as caught:
            language.parse(source, observe=fail)
        assert caught.value is failure
        assert len(frames) == len(observed) == 2
        assert language.stack == [] and language.source_stack == []


def test_actual_decorator_preserves_annotation_definition_and_body_scopes(language):
    # Before: def outer(extent): @X.script def f(x: X.tensor((extent,))): local_extent = 2
    # Expected builder program:
    # X.arg("x", X.tensor((definition_extent,))); local_extent = X.bind_(2, name="local_extent")
    X = language.X
    calls = []

    def ordinary_decorator(function):
        calls.append("decorate")
        return function

    X.ordinary = ordinary_decorator

    def outer(extent):
        @X.script
        def main(x: X.tensor((extent,))):
            local_extent = 2

            @X.ordinary
            def helper():
                calls.append("call")
                return local_extent

            X.record(helper())

        return main

    result = outer(7)
    assert result.params[0].args[0].args[0] == (7,)
    assert result.body[0][1] == 2
    assert calls == ["decorate", "call"]
    assert [event[1] for event in language.events if event[0] == "name"] == ["main"]


def test_module_declarations_precede_bodies_and_reuse_frames(language):
    # Before: @X.script def first(x: ...): second(x); @X.script def second(y: ...): ...
    # Expected builder program:
    # with X.function_(decl=True) as first_fn: X.arg("x", ...)
    # with X.function_(decl=True) as second_fn: X.arg("y", ...)
    # with first_fn: X.emit_(X.call_global_var_(second_fn.global_var, [first_fn.params[0]]))
    result = language.parse("""
@I.ir_module
class Module:
    @X.script
    def first(x: X.tensor((4,))):
        second(x)
    @X.script
    def second(y: X.tensor((4,))):
        X.record(y)
""")
    entries = [event for event in language.events if event[:2] == ("enter", "function")]
    assert [event[2] for event in entries] == [True, True, False, False]
    assert entries[0][3] is entries[2][3]
    assert entries[1][3] is entries[3][3]
    assert len([event for event in language.events if event[0] == "arg"]) == 2
    call = result["first"].body[0][1]
    assert call.op == "call"
    assert call.args[0] is language.references["second"]
    assert call.args[1] is result["first"].params[0]
    assert result["second"].body[0][1] is result["second"].params[0]


def test_conditional_outputs_share_one_native_frame_result(language):
    # Before: if condition: y = left; else: y = right; X.record(y)
    # Expected builder program:
    # with X.If(condition) as frame:
    #     with X.Then():
    #         def then(): y = X.bind_(left, name="y")
    #         then()
    #     with X.Else():
    #         def otherwise(): y = X.bind_(right, name="y")
    #         otherwise()
    # y = frame.var; X.emit_(X.record(y))
    language.X.__tvm_value_if__ = True
    condition, left, right = Value("condition"), Value("left"), Value("right")
    result = language.parse(
        """
@X.script
def main():
    if condition:
        y = left
    else:
        y = right
    X.record(y)
""",
        condition=condition,
        left=left,
        right=right,
    )
    output = result.body[0][1]
    assert output.op == "if" and output.args == (condition, left, right)
    frame = next(event[3] for event in language.events if event[:2] == ("enter", "if"))
    assert output is frame.var


@pytest.mark.parametrize("other", ["z = right", "X.record(right)"])
def test_conditional_branches_require_matching_output_names(language, other):
    # Before: if condition: y = left; else: z = right
    # Expected builder program: reject branches without the same named terminal output.
    language.X.__tvm_value_if__ = True
    with pytest.raises(SyntaxError, match="same named output"):
        language.parse(
            "@X.script\ndef main():\n    if condition:\n        y = left\n"
            f"    else:\n        {other}\n"
        )


def test_repeated_parses_acquire_independent_symbol_trees(language, monkeypatch):
    # Before: parse the same quoted symbolic annotation twice.
    # Expected builder program: each parse owns fresh syntax and its own symbol map.
    source = '@X.script\ndef main(x: X.tensor(("n + 1",))):\n    X.record(x)\n'
    acquire = entry.acquire_source
    acquired = []
    original = []

    def capture(*args, **kwargs):
        tree, filename, flags = acquire(*args, **kwargs)
        acquired.append(tree)
        original.append(ast.dump(tree, include_attributes=True))
        return tree, filename, flags

    monkeypatch.setattr(entry, "acquire_source", capture)
    first = language.parse(source)
    second = language.parse(source)
    assert len(acquired) == 2 and acquired[0] is not acquired[1]
    assert original[0] == original[1]
    assert first.params[0] is not second.params[0]
    first_shape = first.params[0].args[0].args[0][0]
    second_shape = second.params[0].args[0].args[0][0]
    assert first_shape.op == second_shape.op == "add"
    assert first_shape.args[0] is not second_shape.args[0]
    assert first_shape.args[1] == second_shape.args[1] == 1
    assert first.body[0][1] is first.params[0]
    assert second.body[0][1] is second.params[0]


def test_bare_callable_alias_does_not_acquire_constexpr_syntax(language):
    # Before: marker = I.constexpr; if marker(True): ...
    # Expected builder program: an ordinary call to marker raises its runtime syntax-marker error.
    with pytest.raises(TypeError, match="syntax marker"):
        language.parse(
            """
@X.script
def main():
    if marker(True):
        X.record(1)
""",
            marker=registry.constexpr,
        )


@pytest.mark.parametrize("recursive", [False, True])
def test_source_function_uses_declaration_only_when_reference_is_needed(language, recursive):
    # Before: @X.script def main(x: ...): X.record(x)  (or main(x))
    # Expected builder program:
    # ordinary: with X.function_(): x = X.arg(...); X.emit_(X.record(x))
    # recursive: with X.function_(decl=True) as fn: X.arg(...)
    #            with fn: X.emit_(X.call_global_var_(fn.global_var, [fn.params[0]]))
    statement = "main(x)" if recursive else "X.record(x)"
    result = language.parse(f"@X.script\ndef main(x: X.tensor((4,))):\n    {statement}\n")
    entries = [event for event in language.events if event[:2] == ("enter", "function")]
    assert [event[2] for event in entries] == ([True, False] if recursive else [False])
    assert len([event for event in language.events if event[0] == "arg"]) == 1
    if recursive:
        assert entries[0][3] is entries[1][3]
        call = result.body[0][1]
        assert call.op == "call" and call.args[0] is language.references["main"]
        assert call.args[1] is result.params[0]
    else:
        assert result.body[0][1] is result.params[0]


@pytest.mark.parametrize("alias", [False, True])
def test_module_alias_keeps_frame_identity_and_caller_dialect(language, alias):
    # Before: cls = Module; cls.callee(x)
    # Expected builder program:
    # with I.ir_module() as Module: ...
    # cls = Module; X.emit_(X.call_global_var_(cls.callee, [x]))
    setup, owner = ("cls = Module", "cls") if alias else ("pass", "Module")
    result = language.parse(f"""
@I.ir_module
class Module:
    @X.script
    def caller(x: X.tensor((4,))):
        {setup}
        {owner}.callee(x)
    @X.script
    def callee(y: X.tensor((4,))):
        X.record(y)
""")
    call = result["caller"].body[0][1]
    assert call.op == "call"
    assert call.args == (language.references["callee"], result["caller"].params[0])
    modules = [event[3] for event in language.events if event[:2] == ("enter", "module")]
    assert len(modules) == 1
    if alias:
        assert not any(event[:2] == ("bind", "cls") for event in language.events)
        assert modules[0].callee is language.references["callee"]


@pytest.mark.parametrize(
    "body, line, column, end_column, message",
    [
        ("    X = 1\n", 3, 5, 6, "namespace"),
        ("    for X in range(2):\n        pass\n", 3, 9, 10, "namespace"),
        (
            "    if condition:\n        y = left\n    else:\n        z = right\n",
            6,
            9,
            18,
            "same named output",
        ),
    ],
)
def test_syntax_diagnostics_point_to_the_offending_binding(
    language, body, line, column, end_column, message
):
    # Before: X = 1; or if condition: y = left; else: z = right
    # Expected builder program: reject the offending namespace/output binding
    # at its original filename, line and column, before running the builder.
    language.X.__tvm_value_if__ = True
    with pytest.raises(SyntaxError, match=message) as error:
        language.parse("@X.script\ndef main():\n" + body)
    cause = error.value
    assert isinstance(cause, SyntaxError)
    assert (cause.filename, cause.lineno, cause.offset) == ("dummy.py", line, column)
    assert cause.end_lineno == line
    assert cause.end_offset == end_column
    assert not language.events


def test_void_branch_statements_need_no_synthetic_named_output(language):
    # Before: if condition: X.record(None); else: X.record(None)
    # Expected builder program:
    # with X.If(condition):
    #     with X.Then(): X.emit_(X.record(None))
    #     with X.Else(): X.emit_(X.record(None))
    language.X.__tvm_value_if__ = True
    result = language.parse(
        """
@X.script
def main():
    if condition:
        X.record(None)
    else:
        X.record(None)
    X.record(9)
""",
        condition=Value("condition"),
    )
    assert [value for kind, value in result.body] == [None, None, 9]
    assert not any(event[0] == "bind" for event in language.events)


def test_lexical_range_binding_calls_the_custom_iterator_once(language):
    # Before: range = custom_range; for i in range(4): X.record(i)
    # Expected builder program: range = X.bind_(custom_range, name="range")
    # with X.for_(range(4), names="i") as i: X.emit_(X.record(i))
    calls = []

    def custom_range(extent):
        calls.append(extent)
        return language.X.grid(2)

    result = language.parse(
        "@X.script\ndef main():\n    range = custom_range\n"
        "    for i in range(4):\n        X.record(i)\n",
        custom_range=custom_range,
    )
    assert calls == [4]
    variable = result.body[0][1]
    assert variable.op == "loop" and variable.args == (2,) and variable.name == "i"


@pytest.mark.parametrize("expression", ["value", "holder.item", "items[0]", "7"])
def test_non_call_expression_reads_keep_their_source_range(language, expression):
    # Before: value; holder.item; items[0]; 7
    # Expected builder program: X.emit_(expression, span=_S[i]).
    # Names/literals need no expression wrapper; emission still owns their location.
    # Native expression identity survives; host constants stay ordinary values.
    value, reads, located = Value("read"), [], []

    class Holder:
        @property
        def item(self):
            reads.append("attribute")
            return value

        def __getitem__(self, index):
            reads.append(index)
            return value

    def at(location, result):
        located.append(((str(location[0].name), *location[1:]), result))
        return language.at(location, result)

    language.I.at_ = at
    holder = Holder()
    result = language.parse(
        f"@X.script\ndef main():\n    {expression}\n", value=value, holder=holder, items=holder
    )
    expected = 7 if expression == "7" else value
    location = ("dummy.py", 3, 3, 5, 5 + len(expression))
    assert result.body[0][1] is expected
    assert [(loc, result) for loc, result in located if loc == location] == [(location, expected)]
    assert reads == {"holder.item": ["attribute"], "items[0]": [0]}.get(expression, [])


@pytest.mark.parametrize("dtype", [None, "int32", "int64"])
def test_argument_policy_preserves_expression_dtype(language, dtype):
    @registry.args_policy("X.shape", {"values": "expr_str"}, dtype=dtype)
    def shape(values):
        return values

    language.X.shape = shape
    function = language.parse("""
@X.script
def main():
    X.shape(("n", "n + 1"))
""")
    n, increment = function.body[0][1]
    assert n.args == (dtype,)
    assert increment.op == "add"
    assert increment.args[0] is n
    assert increment.args[1] == 1
