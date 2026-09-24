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
"""Production source-span composition on common IR nodes."""

import pytest

from tvm import ir
from tvm.ir import prim
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I


def loc(line, name="source.py", column=1, end_column=20):
    return (name, line, line, column, end_column)


def locations(value):
    span = value.span
    spans = span.spans if isinstance(span, ir.SequentialSpan) else [span]
    return [(str(s.source_name.name), s.line, s.column, s.end_column) for s in spans]


def test_at_preserves_native_expression_identity():
    with IRBuilder():
        value = prim.IntImm("int32", 2)
        assert I.at_(loc(4), value) is value
        assert locations(value) == [("source.py", 4, 1, 20)]


def test_enclosing_locations_collapse_without_losing_definition():
    with IRBuilder():
        value = prim.IntImm("int32", 1)
        I.at_(loc(10, column=5, end_column=10), value)
        I.at_(loc(10), value)
        assert locations(value) == [("source.py", 10, 5, 10)]
        I.at_(loc(10, column=6, end_column=8), value)
        assert locations(value) == [("source.py", 10, 6, 8)]


def test_composition_merges_shared_prefix_of_distinct_definition_chains():
    with IRBuilder():
        caller = ir.Span(ir.SourceName("caller.py"), 1, 1, 1, 20)
        original = ir.Span(ir.SourceName("definition.py"), 2, 2, 1, 20)
        value = prim.IntImm("int32", 1)
        I.at_(ir.SequentialSpan([caller, original]), value)
        with IRBuilder.current().with_source_span(caller):
            I.at_(loc(3, "definition.py"), value)
    assert locations(value) == [
        ("caller.py", 1, 1, 20),
        ("definition.py", 3, 1, 20),
        ("definition.py", 2, 1, 20),
    ]


def test_span_entry_reuses_fixed_metadata_with_each_dynamic_caller():
    # Before: two callers invoke one inline helper at a fixed definition location.
    # Expected builder program: caller.ctx(lambda: definition(value)) composes
    # the current caller each time; no caller is frozen into the definition entry.
    from tvm.script.ir_builder import base

    definition_span = base.source_span(loc(8, "definition.py"))
    definition = base.SpanEntry(definition_span)
    assert definition.span is definition_span
    with IRBuilder():
        for line in (2, 5):
            caller = base.SpanEntry(base.source_span(loc(line, "caller.py")))
            value = prim.IntImm("int32", line)
            assert caller.ctx(lambda: definition(value)) is value
            assert locations(value) == [("caller.py", line, 1, 20), ("definition.py", 8, 1, 20)]
        marker = object()
        assert definition(marker) is marker
        assert definition.ctx(lambda: marker) is marker
        fresh = definition(prim.IntImm("int32", 9))
        assert locations(fresh) == [("definition.py", 8, 1, 20)]
    assert definition.span is definition_span

    # An assignment keeps construction context without restamping its result.
    from tvm.tirx.script import builder as T

    producer_span = base.source_span(loc(12, "producer.py"))
    kept = ir.Var("producer", "int32", producer_span)
    emitted = []
    with IRBuilder() as builder:

        def construct():
            emitted.append(T.evaluate(definition(prim.IntImm("int32", 4))).value)
            return kept

        assert caller.ctx(construct, attach_result=False) is kept
    assert kept.span.same_as(producer_span) and kept.name == "producer"
    assert len(emitted) == 1 and builder.get().same_as(emitted[0])
    assert locations(emitted[0]) == [("caller.py", 5, 1, 20)]
    assert locations(emitted[0].value) == [
        ("caller.py", 5, 1, 20),
        ("definition.py", 8, 1, 20),
    ]


@pytest.mark.parametrize("existing_definition", [False, True])
def test_explicit_emission_keeps_receipt_identity_and_normalized_span(existing_definition):
    # emit_ adds the definition to either a scalar caller or a chain with the
    # same prefix; the already-stored native statement is never emitted twice.
    from tvm.script.ir_builder import base
    from tvm.tirx.script import builder as T

    caller = base.SpanEntry(base.source_span(loc(3, "caller.py")))
    definition = base.SpanEntry(base.source_span(loc(7, "definition.py")))
    with IRBuilder() as builder:

        def construct():
            receipt = T.evaluate(6)
            assert isinstance(receipt, base.AlreadyEmitted)
            if existing_definition:
                assert definition(receipt) is receipt
            T.emit_(receipt, span=definition)
            return receipt

        receipt = caller.ctx(construct)
    assert builder.get().same_as(receipt.value)
    assert locations(receipt.value) == [("caller.py", 3, 1, 20), ("definition.py", 7, 1, 20)]
    # Conversion of a plain emitted literal must also annotate the native value.
    with IRBuilder() as builder:
        T.emit_(4, span=definition)
    assert locations(builder.get()) == [("definition.py", 7, 1, 20)]
    assert locations(builder.get().value) == [("definition.py", 7, 1, 20)]


@pytest.mark.parametrize(
    "producer, dimensions, names",
    [
        ("serial", 1, None),
        ("parallel", 1, "i"),
        ("vectorized", 1, None),
        ("unroll", 1, "i"),
        ("grid", 1, ("i",)),
        ("grid", 2, ("i", "*tail")),
    ],
)
def test_native_loop_keeps_entered_variables_and_stored_span_at_exit(producer, dimensions, names):
    # Ordinary Python scalar entry and multi-loop entry preserve the native vars,
    # independent of names spelling. Stored spans win over unrelated exit context.
    from tvm import tirx
    from tvm.script.ir_builder import base
    from tvm.tirx.script import builder as T

    caller = base.SpanEntry(base.source_span(loc(3, "caller.py")))
    definition = base.SpanEntry(base.source_span(loc(7, "definition.py")))
    created = []
    with IRBuilder() as builder:

        def construct():
            frame = getattr(T, producer)(*(2, 3)[:dimensions])
            created.append((frame, tuple(frame.vars), [value.name for value in frame.vars]))
            return (
                definition(frame) if names is None else T.for_(frame, names=names, span=definition)
            )

        frame = caller.ctx(construct, attach_result=False)
        assert len(created) == 1
        variables = frame.vars
        expected_names = created[0][2] if names is None else ["i", "tail_0"][:dimensions]
        assert all(value.name for value in variables)
        assert [value.name for value in variables] == expected_names
        entered = frame.__enter__()
        assert entered.same_as(variables[0] if dimensions == 1 else variables)
        assert all(value.same_as(old) for value, old in zip(variables, created[0][1]))
        assert [value.name for value in variables] == expected_names
        value = variables[0] if dimensions == 1 else variables[0] + variables[1]
        receipt = T.evaluate(value)
        with builder.with_source_span(base.source_span(loc(90, "unrelated.py"))):
            frame.__exit__(None, None, None)
    node = builder.get()
    expected = [("caller.py", 3, 1, 20), ("definition.py", 7, 1, 20)]
    for variable in variables:
        assert isinstance(node, tirx.For)
        assert locations(node) == expected and node.loop_var.same_as(variable)
        node = node.body
    assert node.same_as(receipt.value)


def test_span_entry_restores_context_after_original_exception():
    # Before: a nested opaque call raises while constructing its result.
    # Expected builder program: nested .ctx calls propagate that exact exception
    # and restore the caller; later operations see only their own source location.
    from tvm.script.ir_builder import base

    caller = base.SpanEntry(base.source_span(loc(2, "caller.py")))
    definition = base.SpanEntry(base.source_span(loc(8, "definition.py")))
    failure = ValueError("original failure")
    calls = []

    def fail():
        calls.append(True)
        raise failure

    with IRBuilder():
        with pytest.raises(ValueError) as caught:
            caller.ctx(lambda: definition.ctx(fail, attach_result=False))
        assert caught.value is failure
        assert calls == [True]
        fresh = definition(prim.IntImm("int32", 1))
        assert locations(fresh) == [("definition.py", 8, 1, 20)]
