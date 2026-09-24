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

"""Frontend dispatch, lexical capture, and real shared IR source locations."""

import pytest
from dummy_builder import Function

from tvm import ir
from tvm.script import ir as I
from tvm.script.parser import entry, protocol_registry
from tvm.script.parser.inspect_source import Source
from tvm.tirx.script.jit import make_jit

VALUE = 1
MACRO_VALUE = 2


def test_ordinary_class_member_constructs_function(language):
    # Before: ordinary class with a decorated member.
    # Expected builder: enter X.function immediately and emit the member body.
    X = language.X

    class Holder:
        @X.script
        def function():
            X.record(7)

    assert isinstance(Holder.function, Function)
    assert Holder.function.body == [("emit", 7)]


def test_ir_module_defers_members_until_forward_signatures_exist(language):
    # Before: main calls Module.identity, declared later in the class.
    # Expected builder: declare both signatures, then call the reserved identity global.
    X = language.X

    @I.ir_module
    class Module:
        @X.script
        def main(x: X.tensor((2,), "float32")):
            return Module.identity(x)

        @X.script
        def identity(x: X.tensor((2,), "float32")):
            return x

    call = Module["main"].body[0][1]
    assert call.op == "call"
    assert call.args[0] is language.references["identity"]
    assert call.args[1] is Module["main"].params[0]
    assert Module["identity"].body[0][1] is Module["identity"].params[0]
    entries = [event[1:3] for event in language.events if event[0] == "enter"]
    assert entries == [
        ("module", False),
        ("function", True),
        ("function", True),
        ("function", False),
        ("function", False),
    ]


@pytest.fixture
def spanned_language(spanned_language):
    language = spanned_language
    language.X.evaluate = lambda value: value
    language.X.call_extern = lambda dtype, name: ir.Call(ir.GlobalVar(name), [], ret_ty=dtype)
    return language


def _position(span):
    return span.line, span.column, span.end_line, span.end_column


@pytest.mark.parametrize("entrypoint", ["source", "function"])
def test_source_and_ir_call_spans_use_one_based_columns(spanned_language, entrypoint):
    # Before: evaluate(call_extern(...)) through source and callable entry points.
    # Expected builder: _S[i] attaches the fixed call range with one-based columns.
    language = spanned_language
    X = language.X

    def direct():
        X.evaluate(X.call_extern("int32", "direct"))

    if entrypoint == "source":
        program = '@X.script\ndef direct():\n    X.evaluate(X.call_extern("int32", "direct"))\n'
        source = Source(program)
        function = entry.parse(program, extra_vars={"X": X}, root_builder=spanned_language.X)
    else:
        source = Source(direct)
        function = X.script(direct)
    call = source.as_ast().body[0].body[0].value.args[0]
    expected = source.to_span(call)
    assert expected.column == call.col_offset + source.start_column + 1
    assert expected.end_column == call.end_col_offset + source.start_column + 1
    assert _position(function.body[0][1].span) == _position(expected)
    assert function.body[0][1].span.source_name.name == source.source_name


def test_nested_helper_spans_preserve_caller_and_definition_columns(spanned_language):
    # Before: main -> outer inline helper -> inner inline helper -> call.
    # Expected builder: production span stack retains all three call coordinates in order.
    X = spanned_language.X
    text = """@X.inline
def inner():
    X.evaluate(X.call_extern("int32", "nested"))
@X.inline
def outer():
    inner()
@X.script
def main():
    outer()
"""
    source = Source(text)
    inner, outer, main = source.as_ast().body
    calls = [main.body[0].value, outer.body[0].value, inner.body[0].value.args[0]]
    expected = [_position(source.to_span(call)) for call in calls]
    assert [position[1] for position in expected] == [5, 5, 16]
    function = entry.parse(text, extra_vars={"X": X}, root_builder=spanned_language.X)
    span = function.body[0][1].span
    assert isinstance(span, ir.SequentialSpan)
    assert [_position(item) for item in span.spans] == expected
    assert all(item.source_name.name == "<str>" for item in span.spans)


def test_annotation_only_enclosing_binding_is_captured(language):
    # Before: a quoted annotation uses an enclosing shape absent from body bytecode.
    # Expected builder: X.arg sees (3,), never the unrelated caller's (7,).
    X = language.X

    def build():
        shape = (3,)

        @X.script
        def function(A: "X.tensor(shape, 'int32')"):
            X.record(1)

        return function

    shape = (7,)  # noqa: F841
    function = build()
    assert function.params[0].args[0].args[:2] == ((3,), "int32")
    assert function.body == [("emit", 1)]


def _build_symbolic_functions(X):
    @I.ir_module
    class Module:
        @X.script
        def first(x: X.tensor(("n",), "float32")):
            n = X.symbol()  # noqa: F841
            return x

        @X.script
        def second(x: X.tensor(("n",), "float32")):
            n = X.symbol()  # noqa: F841
            return x

    return Module


def test_dynamic_caller_symbol_does_not_join_function_declarations(language):
    # Before: two function signatures declare "n" while a caller has its own n.
    # Expected builder: each function frame resolves its own n, independent of the caller.
    n = ir.Var("n", "int64")
    module = _build_symbolic_functions(language.X)
    first = module["first"].params[0].args[0].args[0][0]
    second = module["second"].params[0].args[0].args[0][0]
    assert first is not second
    assert first is not n
    assert second is not n
    declared = [event[2] for event in language.events if event[:2] == ("symbol", "n")]
    assert all(symbol is first or symbol is second for symbol in declared)
    assert sum(symbol is first for symbol in declared) >= 2
    assert sum(symbol is second for symbol in declared) >= 2


@pytest.mark.parametrize("hygienic,expected", [(True, 2), (False, 1)])
def test_macro_capture_policy_remains_explicit(language, hygienic, expected, monkeypatch):
    # Before: macro captures MACRO_VALUE=2; the global changes to 1 before invocation.
    # Expected builder: hygienic macro records 2; caller-environment macro records 1.
    X = language.X
    X.macro = protocol_registry.declaration_kind("X.macro", "helper")(entry.make_macro_decorator(X))
    monkeypatch.setitem(globals(), "X", X)
    monkeypatch.setitem(globals(), "MACRO_VALUE", 2)

    @X.macro(hygienic=hygienic)
    def write():
        X.record(MACRO_VALUE)

    monkeypatch.setitem(globals(), "MACRO_VALUE", 1)

    @X.script
    def function():
        write()

    assert function.body[0] == ("emit", expected)
    assert [event for event in language.events if event[0] == "record"] == [("record", expected)]


def _define_delayed_annotation(shape, X):
    @X.jit
    def function(output: "X.tensor(shape, 'int32')"):
        X.record(VALUE)

    return function


def test_delayed_jit_retains_annotation_only_definition_binding(language):
    # Before: delayed signature captures shape=(3,), body reads lexical global VALUE.
    # Expected builder: after scope exits X.arg sees (3,), X.record sees 1.
    language.X.jit = make_jit(language.X)
    pending = _define_delayed_annotation((3,), language.X)
    shape = (7,)  # noqa: F841
    VALUE = 2  # noqa: F841
    function = pending.specialize()
    assert function.params[0].args[0].args[:2] == ((3,), "int32")
    assert function.body == [("emit", 1)]
