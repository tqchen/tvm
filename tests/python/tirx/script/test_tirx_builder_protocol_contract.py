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
"""Native TIRx obligations of the builder parser contract."""

import re
import textwrap

import pytest

import tvm
from tvm import ir, tirx
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script.ir_builder import AlreadyEmitted, IRBuilder
from tvm.script.ir_builder import ir as IB
from tvm.script.ir_builder.ir import parser_protocol as P
from tvm.script.parser import protocol_registry as registry
from tvm.tirx.script import builder as TB
from tvm.tirx.script.builder import parser_protocol as TP


def test_documented_two_function_program_builds_the_same_ir():
    # The contract's complete source and builder examples must stay executable.
    blocks = re.findall(r"\.\. code:: python\n\n((?:    .*\n|\n)+)", P.__doc__)
    source_blocks = [block for block in blocks if "@I.ir_module" in block]
    builder_blocks = [block for block in blocks if "with IRBuilder()" in block]
    assert len(source_blocks) == len(builder_blocks) == 1
    source = {"I": I, "T": T}
    # Text parse supplies retrievable source coordinates for this documentation.
    module = tvm.script.from_source(textwrap.dedent(source_blocks[0]), extra_vars=source)
    generated = {"I": IB, "X": TB, "IRBuilder": IRBuilder}
    exec(textwrap.dedent(builder_blocks[0]), generated)
    tvm.ir.assert_structural_equal(module, generated["result"], map_free_vars=True)
    assert not module["first"].params[0].ty.shape[0].same_as(module["second"].params[0].ty.shape[0])


def test_scope_category_keeps_explicit_bind_and_sequence_identity():
    with IRBuilder() as builder:
        with TB.function():
            TB.func_name("f")
            value = TB.bind(7)
            original_name = value.name
            assert TB.scope_var_query_or_decl_(value, name="x") is value
            values = (value,)
            assert TB.scope_var_query_or_decl_(values, name="aggregate") is values
            assert value.name == original_name
            TB.emit_(values)
    function = builder.get()
    assert isinstance(function.body, tirx.Bind)
    assert function.body.var.same_as(value)
    assert registry.is_scope_var_query_or_decl(TB.bind)


def test_scope_category_preserves_pointer_bind_identity_and_ir():
    with IRBuilder() as builder:
        with TB.function():
            TB.func_name("f")
            buffer = TB.arg("A", TB.Buffer((4,), "float32"))
            address = TB.address_of(buffer[0])
            variable = TB.bind(address, TB.handle("float32"))
            original_name = variable.name
            assert isinstance(variable.ty, ir.PointerType)
            assert TB.scope_var_query_or_decl_(variable, name="pointer") is variable
            values = (variable,)
            assert TB.scope_var_query_or_decl_(values, name="aggregate") is values
            assert variable.name == original_name
    # The category keeps the producer's one native Bind, without binding its result again.
    statement = builder.get().body
    assert isinstance(statement, tirx.Bind)
    assert statement.var.same_as(variable)
    tvm.ir.assert_structural_equal(statement, tirx.Bind(variable, address))


@pytest.mark.parametrize(
    "value", [7, tirx.IntImm("int32", 7), object()], ids=["int", "expr", "object"]
)
def test_scope_category_rejects_nonvariable_results(value):
    with pytest.raises(TypeError, match="must return a .*variable"):
        TB.scope_var_query_or_decl_(value, name="invalid")


def test_already_emitted_receipt_keeps_statement_identity_and_single_emission():
    span = ir.Span(ir.SourceName("statement.py"), 5, 5, 1, 12)
    with IRBuilder() as builder:
        with TB.function():
            TB.func_name("f")
            receipt = TB.evaluate(7)
            assert isinstance(receipt, AlreadyEmitted)
            assert IB.at_(span, receipt) is receipt
            TB.emit_(receipt)
    statement = builder.get().body
    assert statement.same_as(receipt.value)
    assert statement.span.same_as(span)


@pytest.mark.parametrize("validator", [TP.check_well_formed_, IB.check_well_formed_])
def test_native_primitive_validation_rejects_an_unbound_variable(validator):
    function = tirx.PrimFunc([], tirx.Evaluate(ir.Var("undefined", "int32")))
    value = ir.IRModule({"bad": function}) if validator is IB.check_well_formed_ else function
    with pytest.raises(ValueError, match="Program is not well-formed"):
        validator(value)


def test_direct_object_producers_keep_constructor_names_and_identities():
    # Source direct calls retain original producer naming; they do not bind IR.
    produced = []
    objects = []

    @registry.direct_call
    def record(value):
        objects.append(value)
        return value

    @T.meta_class
    class Resources:
        def __init__(self):
            self.buffer = T.alloc_buffer((1,), "float32")
            IRBuilder.name("owned", self.buffer)
            produced.append(self)

    @T.prim_func
    def function():
        layout = T.TileLayout(T.S[1])
        iv = T.iter_var("explicit", I.Range(0, 1), "DataPar", "")
        scratch = Resources()
        record(layout)
        record(iv)
        T.evaluate(scratch.buffer[0])

    assert len(produced) == 1
    assert len(objects) == 2
    assert objects[1].var.name == "explicit"
    assert produced[0].buffer.name == "owned"
    assert registry.is_direct_call(Resources)
    assert registry.is_direct_call(TB.TileLayout)
    assert registry.is_direct_call(TB.iter_var)
    assert registry.is_direct_call(TB.TileLayout(TB.S[1]).canonicalize)
    assert function is not None


def test_source_meta_var_keeps_existing_expression_span_without_a_binding():
    span = ir.Span(ir.SourceName("producer.py"), 8, 8, 3, 9)
    existing = tirx.IntImm("int32", 7, span=span)
    observed = []

    @registry.direct_call
    def observe(value):
        observed.append(value.span)

    @T.prim_func
    def function():
        kept = I.meta_var(existing)
        observe(kept)
        T.evaluate(kept)

    assert isinstance(function.body, tirx.Evaluate)
    assert function.body.value.same_as(existing)
    assert len(observed) == 1 and observed[0].same_as(span)
    # A later ordinary read still receives its normal source attribution.
    assert not existing.span.same_as(span)


def test_concise_thread_scope_closes_at_parent_exit():
    with IRBuilder() as builder:
        with TB.function() as parent:
            TB.func_name("f")
            child = TB.launch_thread("threadIdx.x", 32)
            assert IRBuilder.current().frames[-1].same_as(parent)
            tid = TB.bind_(child, name="tid")
            assert IRBuilder.current().frames[-1].same_as(child)
            TB.emit_(TB.evaluate(tid))
    function = builder.get()
    assert isinstance(function.body, tirx.AttrStmt)
    assert function.body.attr_key == "thread_extent"
    assert function.body.node.var.same_as(tid)
    assert isinstance(function.body.body, tirx.Evaluate)
    assert function.body.body.value.same_as(tid)


def test_result_member_registration_uses_real_callable_and_descriptor_identities():
    from tvm.tirx import _buffer_view
    from tvm.tirx.buffer import _BufferMethods

    assert registry.get_result_members(TB.Buffer) is _BufferMethods
    assert registry.get_result_members(TB.alloc_buffer) is _BufferMethods
    assert registry.get_result_members(_BufferMethods.view) is _BufferMethods
    assert registry.get_result_members(_BufferMethods.sub) is _buffer_view.SubIndexer
    assert registry.get_result_members(_BufferMethods.tile) is _buffer_view.TileIndexer
    for indexer in (_buffer_view.SubIndexer, _buffer_view.TileIndexer, _buffer_view.ChunkIndexer):
        assert registry.is_direct_call(indexer.__getitem__)
        assert registry.get_result_members(indexer.__getitem__) is _BufferMethods

    class Unrelated:
        def view(self):
            return self

    assert registry.get_result_members(Unrelated.view) is None
    assert not registry.is_direct_call(Unrelated.view)
