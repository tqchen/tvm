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
"""Production call scopes preserve common node identity and definition locations."""

import pytest

from tvm import ir
from tvm.ir import prim
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.parser import entry


def loc(line, name="calls.py"):
    return (name, line, line, 1, 20)


def span_lines(value):
    spans = value.span.spans if isinstance(value.span, ir.SequentialSpan) else [value.span]
    return [span.line for span in spans]


@pytest.fixture
def spanned(spanned_language):
    language = spanned_language
    language.X.node = lambda value: prim.IntImm("int32", value)
    return language


def test_call_scope_keeps_result_identity_and_restores_after_exception():
    seen = []
    with IRBuilder():
        value = prim.IntImm("int32", 1)
        assert I.with_at_group_(loc(1), lambda: (seen.append(1), value)[1]) is value
        assert seen == [1]
        with pytest.raises(ValueError, match="failure"):
            I.with_at_group_(loc(2), lambda: (_ for _ in ()).throw(ValueError("failure")))
        other = I.at_(loc(3), prim.IntImm("int32", 2))
        assert span_lines(other) == [3]


def test_scoped_helper_multiple_emissions_and_normal_return(spanned):
    marker = object()
    with spanned.context(), spanned.X.function():

        def helper():
            first = I.with_at_group_(loc(8), lambda: prim.IntImm("int32", 1))
            second = I.with_at_group_(loc(9), lambda: prim.IntImm("int32", 2))
            spanned.X.emit_(first)
            spanned.X.emit_(second)
            return marker

        assert I.with_at_group_(loc(4), helper) is marker
    statements = [value for _, value in spanned.result.body]
    assert len(statements) == 2
    assert [int(value) for value in statements] == [1, 2]
    assert span_lines(statements[0]) == [4, 8]
    assert span_lines(statements[1]) == [4, 9]


def test_module_source_calls_have_context_before_annotations(spanned):
    seen = []

    def record():
        node = prim.IntImm("int32", 3)
        IRBuilder.current()._set_current_source_span(node)
        seen.append(node.span)
        return {}

    spanned.I.module_attrs = lambda attrs: None
    module = spanned.parse(
        """
@I.ir_module
class Module:
    I.module_attrs(record())
    @X.script
    def main():
        X.record(1)
""",
        I=spanned.I,
        record=record,
    )
    assert "main" in module
    assert len(seen) == 1
    assert seen[0] is not None
    assert seen[0].line == 4


def test_source_inline_keeps_caller_and_each_definition_location(spanned):
    function = entry.parse(
        """
@X.inline
def inner(x):
    X.record(X.node(x))
    X.record(X.node(x + 1))
@X.script
def main():
    inner(1)
""",
        extra_vars={"X": spanned.X},
        filename="inline_scope.py",
    )
    statements = [value for _, value in function.body if isinstance(value, prim.IntImm)]
    assert len(statements) == 2
    assert [int(value) for value in statements] == [1, 2]
    assert span_lines(statements[0]) == [8, 4]
    assert span_lines(statements[1]) == [8, 5]
    assert all(
        str(span.source_name.name) == "inline_scope.py"
        for value in statements
        for span in value.span.spans
    )
