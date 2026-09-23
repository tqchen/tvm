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
from types import SimpleNamespace

import pytest
from dummy_builder import Language, Value

from tvm.script.parser import entry, protocol


@pytest.fixture
def language(monkeypatch):
    language = Language()
    monkeypatch.setattr(entry, "builder_ir", language.I)
    return language


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
    #                     "float32", I.resolve_global_info("cuda:1")))
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
    #                             device=I.resolve_global_info("mesh[0]")))
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


def test_policy_registration_is_immutable_and_aliases_keep_identity():
    # Before: alias = constructor; alias(shape=("n",))
    # Expected builder program: alias(shape=(X.resolve_type_var_("n"),))
    fields = {"shape": "expr_str", "device": "global_info"}
    decorate = protocol.args_policy(fields)
    fields["shape"] = "global_info"

    def constructor(shape, device):
        return shape, device

    alias = decorate(constructor)
    policy = protocol.get_args_policy(alias)
    assert policy is protocol.get_args_policy(constructor)
    assert dict(policy.fields) == {"shape": "expr_str", "device": "global_info"}
    with pytest.raises(TypeError):
        policy.fields["shape"] = "global_info"
    assert protocol.get_args_policy([]) is None
    assert protocol.get_args_policy(None) is None

    @protocol.expr_str_args("values", scalar_strings=False)
    def shorthand(values):
        return values

    assert dict(protocol.get_args_policy(shorthand).fields) == {"values": "expr_str"}
    assert protocol.expr_str_policy(shorthand).fields == ("values",)


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"shape": "vdevice"}, "Unknown argument policies"),
        ({"missing": "expr_str"}, "Unknown argument policy fields"),
    ],
)
def test_invalid_policy_registration_is_rejected(fields, message):
    # Before: register an unknown policy or a nonexistent constructor parameter.
    # Expected builder program: registration raises ValueError before parsing.
    with pytest.raises(ValueError, match=message):
        protocol.args_policy(fields)(lambda shape: shape)


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


def test_comparison_chain_evaluates_middle_once_and_preserves_written_order(language):
    # Before: operand(0) < operand(1) <= operand(2)
    # Expected builder program:
    # a, b, c = operand(0), operand(1), operand(2)
    # X.and_(X.lt(a, b), X.le(b, c))
    seen = []
    values = [Value("operand", (index,)) for index in range(3)]

    def operand(index):
        seen.append(index)
        return values[index]

    result = language.parse(
        """
@X.script
def main():
    operand(0) < operand(1) <= operand(2)
""",
        operand=operand,
    )
    assert seen == [0, 1, 2]
    comparison = result.body[0][1]
    assert comparison.op == "and"
    first, second = comparison.args
    assert first.args == tuple(values[:2]) and second.args == tuple(values[1:])


def test_callee_arguments_and_keywords_evaluate_once_with_caller_context(language):
    # Before: callee()(operand(1), b=operand(2))
    # Expected builder program:
    # X.emit_(I.with_at_group_(loc, lambda: callee()(operand(1), b=operand(2))))
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
    # cell = X.decl_mutable_var_(X.cell(), name="cell"); X.set_mutable_var_(cell, x)
    marker = object()
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
        helper=lambda: marker,
    )
    assert result.body[-1][1] is marker
    binds = [event for event in language.events if event[0] == "bind"]
    assert [(event[1], event[2]) for event in binds] == [("x", marker)]
    declaration = next(event for event in language.events if event[0] == "declare")
    mutation = next(event for event in language.events if event[0] == "set")
    assert declaration[1] == "cell" and mutation[1] is declaration[2]
    assert mutation[2] is marker


def test_quoted_symbols_share_identity_without_introducing_python_bindings(language):
    # Before: def main(x: X.tensor(("n", "n"))): X.record(n)
    # Expected builder program: X.tensor((X.resolve_type_var_("n"), X.resolve_type_var_("n")))
    # The unquoted body name n raises NameError until explicitly declared.
    with pytest.raises(Exception) as error:
        language.parse("""
@X.script
def main(x: X.tensor(("n", "n"))):
    X.record(n)
""")
    assert isinstance(error.value.__cause__, NameError)


def test_explicit_symbol_declaration_reuses_annotation_identity(language):
    # Before: def main(x: X.tensor(("n",))): n = X.symbol(); X.record(n)
    # Expected builder program: n = X.resolve_type_var_("n", "int64"); X.emit_(X.record(n))
    result = language.parse("""
@X.script
def main(x: X.tensor(("n",))):
    n = X.symbol()
    X.record(n)
""")
    symbol = result.params[0].args[0].args[0][0]
    assert result.body[0][1] is symbol


@pytest.mark.parametrize(
    "binding", ["X = 1", "for X in range(2):\n        pass", "with context() as X:\n        pass"]
)
def test_script_namespace_cannot_be_rebound(language, binding):
    # Before: X = 1  (also loop/with targets)
    # Expected builder program: reject a binding that shadows the fixed script namespace X.
    with pytest.raises(Exception, match="namespace|shadow|rebind"):
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


@pytest.mark.parametrize("target, bounds, names", [("i", "4", "i"), ("i, j", "4, 5", ("i", "j"))])
def test_loop_targets_configure_the_entered_frame(language, target, bounds, names):
    # Before: for i, j in X.grid(4, 5): X.record(i)
    # Expected builder program:
    # with X.for_(X.grid(4, 5), names=("i", "j")) as (i, j):
    #     X.emit_(X.record(i))
    result = language.parse(f"""
@X.script
def main():
    for {target} in X.grid({bounds}):
        X.record(i)
""")
    assert ("loop_names", names) in language.events
    assert result.body[0][1].name == "i"


def test_actual_decorator_preserves_annotation_definition_and_body_scopes(language):
    # Before: def outer(extent): @X.script def f(x: X.tensor((extent,))): extent = 2
    # Expected builder program:
    # X.arg("x", X.tensor((definition_extent,))); extent = X.bind_(2, name="extent")
    X = language.X

    def outer(extent):
        @X.script
        def main(x: X.tensor((extent,))):
            extent = 2
            X.record(extent)

        return main

    result = outer(7)
    assert result.params[0].args[0].args[0] == (7,)
    assert result.body[0][1] == 2


def test_policy_resolution_preserves_instance_and_namespace_callables(language):
    # Before: namespace.constructor(("n",)); instance.constructor(("n",))
    # Expected builder program: each registered callable receives (X.resolve_type_var_("n"),).
    seen = []

    class Owner:
        @protocol.args_policy({"shape": "expr_str"})
        def constructor(self, shape):
            seen.append(shape)
            return shape

    owner = Owner()
    namespace = SimpleNamespace(constructor=Owner.constructor)
    language.parse(
        """
@X.script
def main():
    namespace.constructor(owner, ("n",))
    owner.constructor(("n",))
""",
        namespace=namespace,
        owner=owner,
    )
    assert len(seen) == 2 and seen[0][0] is seen[1][0]


def test_module_declarations_precede_bodies_and_reuse_frames(language):
    # Before: @X.script def first(x: ...): second(x); @X.script def second(y: ...): ...
    # Expected builder program:
    # with X.function(decl=True) as first_fn: X.arg("x", ...)
    # with X.function(decl=True) as second_fn: X.arg("y", ...)
    # with first_fn: X.emit_(X.call_global_var_(second_fn.reference, [first_fn.params[0]]))
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
    with pytest.raises(Exception, match="same named output"):
        language.parse(
            "@X.script\ndef main():\n    if condition:\n        y = left\n"
            f"    else:\n        {other}\n"
        )


def test_entry_keeps_reusable_input_tree_unchanged(language, monkeypatch):
    # Before: x: X.tensor(("n + 1",)); X.record(x)
    # Expected builder program: transform one owned copy; the caller's source AST is unchanged.
    source = '@X.script\ndef main(x: X.tensor(("n + 1",))):\n    X.record(x)\n'
    acquired = entry.acquire_source(source, "dummy.py")
    before = ast.dump(acquired[0], include_attributes=True)
    monkeypatch.setattr(entry, "acquire_source", lambda *args, **kwargs: acquired)
    first = language.parse(source)
    second = language.parse(source)
    assert ast.dump(acquired[0], include_attributes=True) == before
    assert first.params[0] is not second.params[0]
    assert first.params[0].args[0].args[0][0].op == "add"
    assert second.params[0].args[0].args[0][0].op == "add"


def test_bare_callable_alias_does_not_acquire_constexpr_syntax(language):
    # Before: marker = I.constexpr; if marker(True): ...
    # Expected builder program: an ordinary call to marker raises its runtime syntax-marker error.
    with pytest.raises(Exception, match="syntax marker"):
        language.parse(
            """
@X.script
def main():
    if marker(True):
        X.record(1)
""",
            marker=protocol.constexpr,
        )
