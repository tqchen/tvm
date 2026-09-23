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
"""Scope IDs retain the native variables defined by their emitted statements."""

import pytest

import tvm
from tvm.script import parser
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I


@pytest.mark.parametrize(
    "constructor, target, extents, names",
    [
        ("thread_id", "tx", "[32]", ["tx"]),
        ("cta_id", "bx, by", "[2, 3]", ["bx", "by"]),
        ("warp_id", "wx", "[4]", ["wx"]),
    ],
)
def test_scope_id_declarations_and_uses_share_identity(constructor, target, extents, names):
    # Tuple dimensions are each used, not just the leading scope variable.
    evaluations = "\n".join(f"        T.evaluate({name})" for name in names)
    source = f"""
@T.prim_func
def main():
    T.device_entry()
    {target} = T.{constructor}({extents})
    if {names[0]} == 0:
{evaluations}
"""
    with IRBuilder() as builder:
        with T.function():
            T.func_name("main")
            T.device_entry()
            declaration = getattr(T, constructor)(eval(extents))
            variables = declaration if isinstance(declaration, tuple) else (declaration,)
            with T.If(tvm.tirx.EQ(variables[0], 0)):
                with T.Then():
                    for variable in variables:
                        T.evaluate(variable)
    expected = builder.get()
    actual = parser.parse(source)
    tvm.ir.assert_structural_equal(expected, actual)
    declaration, branch = actual.body.body.seq
    variables = getattr(declaration, "def").def_ids
    assert variables[0].same_as(branch.condition.a)
    statements = (
        branch.then_case.seq
        if isinstance(branch.then_case, tvm.tirx.SeqStmt)
        else [branch.then_case]
    )
    for variable, statement in zip(variables, statements):
        assert variable.same_as(statement.value)


def test_scope_owned_variable_does_not_replace_signature_symbol():
    with IRBuilder():
        with T.function() as symbols:
            signature_symbol = symbols.resolve_type_var("tx", "int32")
            T.device_entry()
            variable = T.thread_id([32])
            bound = T.scope_var_query_or_decl_(variable, name="tx")
            assert bound is variable
            assert not bound.same_as(signature_symbol)
            assert symbols.resolve_type_var("tx").same_as(signature_symbol)
            T.evaluate(bound)


def test_unowned_anonymous_declarations_still_reuse_function_symbols():
    with IRBuilder():
        with T.function() as symbols:
            signature_symbol = symbols.resolve_type_var("n", "int32")
            value = T.int32()
            assert not value.same_as(signature_symbol)
            assert T.resolve_type_var_("n", "int32").same_as(signature_symbol)
            T.evaluate(0)


@pytest.mark.parametrize("dialect", ["tirx", "relax"])
def test_meta_var_keeps_python_payload_identity_in_each_dialect(dialect):
    from importlib import import_module

    namespace = import_module(f"tvm.{dialect}.script")
    value = object()
    # The shared identity call also works without a builder context.
    assert I.meta_var(value) is value
    observed = []

    def observe(result):
        observed.append(result)

    decorator = "prim_func" if dialect == "tirx" else "function"
    tail = "X.evaluate(0)" if dialect == "tirx" else "return X.const(0)"
    parser.parse(
        f"""
@X.{decorator}
def main():
    kept = I.meta_var(payload)
    I.meta_var(observe(kept))
    {tail}
""",
        extra_vars={"X": namespace, "I": I, "payload": value, "observe": observe},
    )
    assert len(observed) == 1
    assert observed[0] is value


def test_scope_tuple_assignment_returns_native_values():
    with IRBuilder():
        with T.function():
            T.device_entry()
            result = T.cta_id([2, 3])
            assert isinstance(result, tuple)
            values = T.scope_var_query_or_decl_(result, name="ids")
            assert values is result
            unpacked = T.unpack(result)
            assert unpacked is result
            for index, (variable, item) in enumerate(zip(values, unpacked)):
                assert isinstance(item, tvm.ir.Var)
                assert T.scope_var_query_or_decl_(item, name=f"axis_{index}") is variable
                T.evaluate(variable)


@pytest.mark.parametrize("track_span", [True, False])
def test_scope_ids_in_single_tuple_target(track_span):
    source = """
@T.prim_func
def main():
    T.device_entry()
    ids = T.cta_id([2, 3])
    T.evaluate(ids[0])
    T.evaluate(ids[1])
"""
    actual = parser.parse(source, track_span=track_span)
    declaration, first, second = actual.body.body.seq
    variables = getattr(declaration, "def").def_ids
    assert variables[0].same_as(first.value)
    assert variables[1].same_as(second.value)
    if track_span:
        assert variables[0].span is not None
        assert variables[1].span is not None


@pytest.mark.parametrize(
    "constructor,args",
    [
        ("scope_id", ([32], "cta", "thread")),
        ("cluster_id", ([2],)),
        ("cta_id", ([2],)),
        ("cta_id_in_cluster", ([2],)),
        ("cta_id_in_pair", ()),
        ("warpgroup_id", ([2],)),
        ("warp_id", ([4],)),
        ("warp_id_in_wg", ([4],)),
        ("lane_id", ([32],)),
        ("thread_id", ([32],)),
        ("thread_id_in_wg", ([128],)),
    ],
)
def test_all_scope_id_helpers_return_the_native_declared_variable(constructor, args):
    with IRBuilder() as builder:
        with T.function():
            T.device_entry()
            result = getattr(T, constructor)(*args)
            assert isinstance(result, tvm.ir.Var)
            value = T.scope_var_query_or_decl_(result, name="id")
            assert value is result
            T.evaluate(value)
        function = builder.get()
    declaration, use = function.body.body.seq
    assert getattr(declaration, "def").def_ids[0].same_as(use.value)


@pytest.mark.parametrize("extents", [[32], [2, 3]])
@pytest.mark.parametrize("emitter", [T.emit, T.emit_])
def test_standalone_scope_ids_direct(extents, emitter):
    with IRBuilder() as builder:
        with T.function():
            T.device_entry()
            result = T.cta_id(extents)
            emitter(result)
        function = builder.get()
    declaration = function.body.body
    assert isinstance(declaration, tvm.tirx.ScopeIdDefStmt)
    values = result if isinstance(result, tuple) else (result,)
    for declared, value in zip(getattr(declaration, "def").def_ids, values):
        assert declared.same_as(value)


@pytest.mark.parametrize("extents", [[32], [2, 3]])
@pytest.mark.parametrize("track_span", [True, False])
def test_standalone_scope_ids_source(extents, track_span):
    source = f"""
@T.prim_func
def main():
    T.device_entry()
    T.cta_id({extents!r})
"""
    with IRBuilder() as builder:
        with T.function():
            T.func_name("main")
            T.device_entry()
            T.cta_id(extents)
    expected = builder.get()
    actual = parser.parse(source, track_span=track_span)
    tvm.ir.assert_structural_equal(expected, actual)
    assert isinstance(actual.body.body, tvm.tirx.ScopeIdDefStmt)


@pytest.mark.parametrize("emitter", [T.emit, T.emit_])
@pytest.mark.parametrize("tuple_value", [False, True])
def test_scalar_and_tuple_values_emit_without_binding_wrappers(emitter, tuple_value):
    with IRBuilder() as builder:
        with T.function():
            values = (tvm.tirx.IntImm("int32", 7), tvm.tirx.IntImm("int32", 9))
            emitter(values if tuple_value else values[0])
        function = builder.get()
    statements = function.body.seq if tuple_value else [function.body]
    assert [statement.value.value for statement in statements] == ([7, 9] if tuple_value else [7])
