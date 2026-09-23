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

"""Relax frame finalization uses explicit locations captured before entry."""

from tvm import ir
from tvm.relax.script import builder as R
from tvm.script.ir_builder import IRBuilder, base


def location(line, filename="relax_frame.py"):
    return (filename, line, line, 1, 20)


def positions(span):
    spans = span.spans if isinstance(span, ir.SequentialSpan) else [span]
    return [(item.source_name.name, item.line) for item in spans]


def captured(line, factory):
    return base.with_at_group_(
        location(2, "caller.py"), lambda: base.with_at_group_(location(line), factory)
    )


def test_function_and_implicit_block_keep_stored_spans_after_context_returns():
    with IRBuilder() as builder:
        frame = captured(5, R.function)
        frame.__enter__()
        R.func_name("main")
        x = R.arg("x", R.Tensor((2,), "float32"))
        y = R.bind_(R.add(x, x), name="y", span=location(8), name_span=location(8))
        R.func_ret_value(y)
        with builder.with_source_span(base.source_span(location(90, "unrelated.py"))):
            frame.__exit__(None, None, None)
    function = builder.get()
    assert positions(function.span) == [("caller.py", 2), ("relax_frame.py", 5)]
    assert positions(function.body.span) == positions(function.span)
    assert positions(function.body.blocks[0].span) == positions(function.span)
    binding = function.body.blocks[0].bindings[0]
    assert binding.var.same_as(y)
    assert positions(binding.span) == [("relax_frame.py", 8)]
    assert function.body.body.same_as(y)


def test_dataflow_finalizer_preserves_block_and_individual_binding_spans():
    with IRBuilder() as builder, R.function():
        R.func_name("main")
        x = R.arg("x", R.Tensor((2,), "float32"))
        block = captured(12, R.dataflow)
        block.__enter__()
        y = R.bind_(R.add(x, x), name="y", span=location(13), name_span=location(13))
        R.output(y)
        R.func_ret_value(y)
        with builder.with_source_span(base.source_span(location(90, "unrelated.py"))):
            block.__exit__(None, None, None)
    function = builder.get()
    block = function.body.blocks[0]
    assert positions(block.span) == [("caller.py", 2), ("relax_frame.py", 12)]
    binding = block.bindings[0]
    assert positions(binding.span) == [("relax_frame.py", 13)]
    assert positions(binding.var.span) == [("relax_frame.py", 13)]
    assert function.body.body.same_as(binding.var)


def test_conditional_and_branch_results_ignore_exit_time_context():
    with IRBuilder() as builder, R.function():
        R.func_name("main")
        condition = R.arg("condition", R.Tensor((), "bool"))
        x = R.arg("x", R.Tensor((2,), "float32"))
        frame = captured(20, lambda: R.If(condition))
        frame.__enter__()
        first = captured(21, R.Then)
        first.__enter__()
        R.bind_(R.add(x, x), name="y", span=location(22))
        with builder.with_source_span(base.source_span(location(90, "unrelated.py"))):
            first.__exit__(None, None, None)
        second = captured(23, R.Else)
        second.__enter__()
        R.bind_(R.multiply(x, x), name="y", span=location(24))
        with builder.with_source_span(base.source_span(location(90, "unrelated.py"))):
            second.__exit__(None, None, None)
            frame.__exit__(None, None, None)
        R.func_ret_value(frame.var)
    binding = builder.get().body.blocks[0].bindings[0]
    conditional = binding.value
    assert positions(binding.span) == [("caller.py", 2), ("relax_frame.py", 20)]
    assert positions(conditional.span) == positions(binding.span)
    assert positions(conditional.true_branch.span) == [("caller.py", 2), ("relax_frame.py", 21)]
    assert positions(conditional.false_branch.span) == [("caller.py", 2), ("relax_frame.py", 23)]
    assert positions(conditional.true_branch.blocks[0].bindings[0].span) == [("relax_frame.py", 22)]
    assert positions(conditional.false_branch.blocks[0].bindings[0].span) == [
        ("relax_frame.py", 24)
    ]
    assert builder.get().body.body.same_as(frame.var)


def test_eager_relax_binding_composes_call_location_with_explicit_statement_span():
    with IRBuilder() as builder, R.function():
        R.func_name("main")
        x = R.arg("x", R.Tensor((2,), "float32"))
        y = base.with_at_group_(
            location(2, "caller.py"),
            lambda: R.bind_(R.add(x, x), name="y", span=location(31), name_span=location(32)),
        )
        R.func_ret_value(y)
    binding = builder.get().body.blocks[0].bindings[0]
    assert positions(binding.span) == [("caller.py", 2), ("relax_frame.py", 31)]
    assert binding.var.same_as(y)
    assert positions(binding.var.span)[-1] == ("relax_frame.py", 32)
