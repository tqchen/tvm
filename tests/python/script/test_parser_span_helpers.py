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


@pytest.mark.parametrize("value", [None, 3, "python", object()])
def test_at_preserves_python_values(value):
    with IRBuilder():
        assert I.at_(loc(5), value) is value


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
