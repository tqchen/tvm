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

"""Deferred native results own spans and loop variables before frame entry."""

import pytest
import tvm_ffi

from tvm import ir, tirx
from tvm.error import InternalError
from tvm.script.ir_builder import IRBuilder, base
from tvm.tirx.script.builder import _ffi_api
from tvm.tirx.script.builder import ir as native


def location(line, filename="frame.py"):
    return (filename, line, line, 1, 20)


def positions(span):
    if span is None:
        return []
    spans = span.spans if isinstance(span, ir.SequentialSpan) else [span]
    return [(item.source_name.name, item.line) for item in spans]


def _captured(factory):
    return base.with_at_group_(
        location(3, "caller.py"), lambda: base.with_at_group_(location(7), factory)
    )


@pytest.mark.parametrize(
    "kind", ["serial", "parallel", "grid", "attr", "while", "assert", "hint", "decl_buffer"]
)
@pytest.mark.parametrize("capture", [True, False], ids=["stored_span", "no_span"])
def test_native_finalizers_ignore_unrelated_exit_context(kind, capture):
    factories = {
        "serial": lambda: native.serial(2),
        "parallel": lambda: native.parallel(2),
        "grid": lambda: native.grid(2, 3),
        "attr": lambda: native.attr(0, "stored_span", 1),
        "while": lambda: native.While(tirx.IntImm("bool", True)),
        "assert": lambda: native.Assert(tirx.IntImm("bool", True), "stored span"),
        "hint": lambda: native.hint("stored span"),
        "decl_buffer": lambda: _ffi_api.DeclBuffer(
            [2], ir.PrimType("float32"), "buffer", None, [], None, "global", 0, 0, None, None
        ),
    }
    with IRBuilder() as builder:
        frame = _captured(factories[kind]) if capture else factories[kind]()
        expected = [("caller.py", 3), ("frame.py", 7)] if capture else []
        assert positions(frame.source_span) == expected
        frame.__enter__()
        receipt = base.at_(location(11, "body.py"), native.evaluate(9))
        with builder.with_source_span(base.source_span(location(90, "unrelated.py"))):
            frame.__exit__(None, None, None)
    result = builder.get()
    assert positions(result.span) == expected
    statements = []
    tvm_ffi.structural_visit(result, [(tirx.Evaluate, lambda node, _: statements.append(node))])
    assert len(statements) == 1 and statements[0].same_as(receipt.value)
    assert positions(statements[0].span) == [("body.py", 11)]
    if kind == "grid":
        assert positions(result.body.span) == positions(result.span)
    if kind in ("assert", "decl_buffer"):
        assert positions(result.seq[0].span) == positions(result.span)


def test_function_and_launch_frame_finalize_with_their_own_spans():
    with IRBuilder() as builder:
        function = _captured(native.prim_func)
        function.__enter__()
        native.func_name("main")
        launch = base.at_(location(12), native.launch_thread("threadIdx.x", 32))
        launch.__enter__()
        receipt = base.at_(location(13), native.evaluate(4))
        with builder.with_source_span(base.source_span(location(90, "unrelated.py"))):
            launch.__exit__(None, None, None)
            function.__exit__(None, None, None)
    result = builder.get()
    assert positions(result.span) == [("caller.py", 3), ("frame.py", 7)]
    assert positions(result.body.span) == [("frame.py", 12)]
    assert result.body.body.same_as(receipt.value)
    assert positions(result.body.body.span) == [("frame.py", 13)]


def test_if_frame_owns_result_while_branches_keep_statement_locations():
    with IRBuilder() as builder:
        frame = _captured(lambda: native.If(tirx.IntImm("bool", True)))
        frame.__enter__()
        with base.at_(location(10), native.Then()):
            first = base.at_(location(11), native.evaluate(1))
        with base.at_(location(20), native.Else()):
            second = base.at_(location(21), native.evaluate(2))
        with builder.with_source_span(base.source_span(location(90, "unrelated.py"))):
            frame.__exit__(None, None, None)
    result = builder.get()
    assert positions(result.span) == [("caller.py", 3), ("frame.py", 7)]
    assert result.then_case.same_as(first.value)
    assert result.else_case.same_as(second.value)
    assert positions(result.then_case.span) == [("frame.py", 11)]
    assert positions(result.else_case.span) == [("frame.py", 21)]


def test_eager_receipt_preserves_python_and_stored_node_identity_once():
    calls = []
    with IRBuilder() as builder:

        def emit():
            calls.append("emit")
            return native.evaluate(6)

        receipt = _captured(emit)
        assert isinstance(receipt, base.AlreadyEmitted)
        assert base.at_(location(3, "caller.py"), receipt) is receipt
        from tvm.tirx.script import builder as T

        T.emit_(receipt)
    assert calls == ["emit"]
    assert builder.get().same_as(receipt.value)
    assert positions(receipt.value.span) == [("caller.py", 3), ("frame.py", 7)]
    assert not hasattr(receipt, "__dict__")


@pytest.mark.parametrize("container", ["list", "tuple"])
def test_source_call_preserves_cyclic_containers_and_native_member_spans(container):
    native_value = ir.Var("value", "int32")
    recursive = []
    recursive.append(recursive)
    value = [native_value, recursive, recursive]
    if container == "tuple":
        value = tuple(value)
    else:
        value.append(value)
    calls = []

    def helper():
        calls.append("helper")
        return value

    with IRBuilder():
        assert _captured(helper) is value
        assert base.at_(location(3, "caller.py"), value) is value
        assert native.serial(2).source_span is None
    assert calls == ["helper"]
    assert value[1] is value[2] is recursive and recursive[0] is recursive
    if container == "list":
        assert value[-1] is value
    assert positions(native_value.span) == [("caller.py", 3), ("frame.py", 7)]


@pytest.mark.parametrize("failure", ["source_call", "frame_exit"])
def test_native_diagnostics_keep_definition_location_and_restore_source_context(failure):
    with IRBuilder() as builder:
        if failure == "source_call":

            def fail():
                raise ValueError("native helper failed")

            with pytest.raises(ValueError, match="native helper failed") as error:
                _captured(fail)
        else:
            frame = _captured(lambda: native.If(tirx.IntImm("bool", True)))
            frame.__enter__()
            with builder.with_source_span(base.source_span(location(90, "unrelated.py"))):
                with pytest.raises(InternalError, match="then") as error:
                    frame.__exit__(None, None, None)
        assert error.value.__tvm_script_location__ == location(7)
        following = native.serial(2)
        assert following.source_span is None
        with following:
            receipt = base.at_(location(30), native.evaluate(1))
    assert builder.get().span is None
    assert positions(receipt.value.span) == [("frame.py", 30)]


@pytest.mark.parametrize(
    "kind", ["serial", "parallel", "vectorized", "unroll", "thread_binding", "grid"]
)
def test_native_loop_entry_always_returns_its_variable_sequence(kind, monkeypatch):
    with IRBuilder() as builder:
        frame = (
            native.thread_binding(2, thread="threadIdx.x")
            if kind == "thread_binding"
            else getattr(native, kind)(2)
        )
        original = frame.vars[0]
        assert original.name
        frame.set_names(("index",))
        assert frame.vars[0].same_as(original) and original.name == "index"
        assert not hasattr(frame, "names")
        monkeypatch.setattr(IRBuilder, "name", lambda *args: pytest.fail("entry must not rename"))
        with frame as variables:
            assert len(variables) == 1 and variables[0].same_as(original)
            native.evaluate(variables[0])
    assert builder.get().loop_var.same_as(original)
    assert builder.get().body.value.same_as(original)


@pytest.mark.parametrize(
    "names,expected",
    [
        (None, ["v0", "v1", "v2"]),
        (("i", "j", "k"), ["i", "j", "k"]),
        (["i", "*rest"], ["i", "rest_0", "rest_1"]),
        (("*prefix", "k"), ["prefix_0", "prefix_1", "k"]),
        ("indices", ["indices_0", "indices_1", "indices_2"]),
    ],
)
def test_grid_names_are_final_before_entry_and_unpack_normally(names, expected):
    with IRBuilder() as builder:
        frame = native.grid(2, 3, 4)
        before = list(frame.vars)
        frame.set_names(names)
        assert [var.name for var in frame.vars] == expected
        assert all(var.same_as(old) for var, old in zip(frame.vars, before))
        with frame as (first, *remaining):
            assert first.same_as(before[0])
            assert len(remaining) == 2
            assert all(var.same_as(old) for var, old in zip(remaining, before[1:]))
            native.evaluate(first + remaining[0] + remaining[1])
    loop = builder.get()
    for var in before:
        assert loop.loop_var.same_as(var)
        loop = loop.body


@pytest.mark.parametrize("names", [("one",), ("*a", "*b"), ("", "second")])
def test_loop_name_validation_precedes_entry_without_partial_rename(names):
    with IRBuilder():
        frame = native.grid(2, 3)
        before = [var.name for var in frame.vars]
        with pytest.raises(ValueError, match="target|names"):
            frame.set_names(names)
        assert [var.name for var in frame.vars] == before


def test_loop_names_cannot_be_changed_inside_entered_frame():
    with IRBuilder(), native.serial(2) as (index,):
        frame = IRBuilder.current().frames[-1]
        with pytest.raises(ValueError, match="before entering"):
            frame.set_names(("late",))
        assert index.name == "v"
        native.evaluate(index)


def test_direct_bind_and_scope_helpers_return_exact_native_variables():
    with IRBuilder() as builder, native.prim_func():
        native.func_name("main")
        native.device_entry()
        scope_vars = native.cta_id([2, 3])
        assert isinstance(scope_vars, tuple) and len(scope_vars) == 2
        assert all(isinstance(var, ir.Var) for var in scope_vars)
        bound = native.bind(scope_vars[0] + scope_vars[1])
        assert isinstance(bound, ir.Var)
        native.evaluate(bound)
    declaration, binding, evaluation = builder.get().body.body.seq
    declared = getattr(declaration, "def").def_ids
    assert all(var.same_as(original) for var, original in zip(declared, scope_vars))
    assert binding.var.same_as(bound)
    assert evaluation.value.same_as(bound)
