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
"""A failed production visitor must restore its caller's annotation scope."""

import ast

import pytest

from tvm.script.parser import entry


@pytest.mark.parametrize(
    "source, expression",
    [
        pytest.param("lambda outer: (broken := 1)", True, id="lambda"),
        pytest.param("[(broken := 1) for outer in values]", True, id="comprehension"),
        pytest.param("value: (broken := 1) = 0", False, id="body-annotation"),
        pytest.param(
            "@X.script\ndef nested(x: (broken := 1)):\n    pass\n",
            False,
            id="function-signature",
        ),
        pytest.param(
            "@X.script\ndef nested(x: X.tensor((4,))):\n    value: outer = 0\n    (broken := 1)\n",
            False,
            id="function-body",
        ),
    ],
)
def test_failed_visitor_restores_later_name_and_body_annotation(language, source, expression):
    setup = "@X.script\ndef main():\n    pass\n"
    tree, filename, _ = entry.acquire_source(setup, filename="alias_restoration.py")
    transformer, namespace = entry._prepare_transpiler(
        tree, setup, {"X": language.X}, {}, filename, track_span=False
    )
    outer, body = object(), object()
    namespace.update(_outer_value=outer, _body_value=body)
    aliases = {"outer": "_outer_value"}
    body_aliases = {"outer": "_body_value"}
    transformer.annotation_aliases = aliases
    transformer.body_annotation_aliases = body_aliases
    previous_scope = transformer.current_scope
    previous_dialect = transformer.dialect_prefix
    failing = ast.parse(source, mode="eval").body if expression else ast.parse(source).body[0]

    with pytest.raises(SyntaxError, match="Unsupported expression: NamedExpr"):
        transformer.visit(failing)

    # Check the observable rewrite after failure, rather than the manager alone:
    # ordinary reads use the caller's alias, body annotations use its other map.
    reference = transformer.visit(ast.parse("outer", mode="eval").body)
    assert (
        eval(
            compile(ast.fix_missing_locations(ast.Expression(reference)), filename, "eval"),
            namespace,
        )
        is outer
    )
    assignment = transformer.visit(ast.parse("result: outer = 0").body[0])[0]
    annotation = next(keyword.value for keyword in assignment.value.keywords if keyword.arg == "ty")
    assert (
        eval(
            compile(ast.fix_missing_locations(ast.Expression(annotation)), filename, "eval"),
            namespace,
        )
        is body
    )
    assert transformer.annotation_aliases is aliases
    assert transformer.body_annotation_aliases is body_aliases
    assert transformer.current_scope is previous_scope
    assert transformer.dialect_prefix == previous_dialect
