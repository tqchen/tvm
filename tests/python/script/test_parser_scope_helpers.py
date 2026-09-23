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
"""Lexical-only body helpers execute inside their already-entered frames."""

import ast
import copy

import pytest

from tvm.error import DiagnosticError
from tvm.script.ir_builder import IRBuilder
from tvm.script.parser import entry


@pytest.mark.parametrize(
    "source, helper_count",
    [
        pytest.param(
            "@X.script\ndef main(x: X.tensor((4,))):\n    X.record(x)\n",
            1,
            id="ordinary",
        ),
        pytest.param(
            "@X.script\ndef main(x: X.tensor((4,))):\n    main(x)\n",
            1,
            id="recursive",
        ),
        pytest.param(
            """
@I.ir_module
class Module:
    @X.script
    def first(x: X.tensor((4,))):
        second(x)
    @X.script
    def second(y: X.tensor((4,))):
        X.record(y)
""",
            2,
            id="module",
        ),
        pytest.param(
            """
@X.script
def main(x: X.tensor((4,))):
    @X.script
    def inner(y: X.tensor((4,))):
        X.record(x)
        X.record(y)
    X.record(x)
""",
            2,
            id="nested",
        ),
        pytest.param(
            """
@X.script
def main(condition: X.tensor(())):
    if condition:
        X.record(1)
    else:
        X.record(2)
""",
            3,
            id="branches",
        ),
    ],
)
def test_lexical_helpers_are_defined_and_called_inside_frames(
    language, monkeypatch, source, helper_count
):
    # The requested structure is with frame: def body(): ...; body().
    # Observe the actual transpiled program and still execute its normal path.
    programs = []
    original = entry.recompose_builder

    def capture(translated, **kwargs):
        programs.append(copy.deepcopy(translated))
        return original(translated, **kwargs)

    monkeypatch.setattr(entry, "recompose_builder", capture)
    language.parse(source)
    assert len(programs) == 1
    tree = programs[0]
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    helpers = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert len(helpers) == helper_count
    for helper in helpers:
        scope = parents[helper]
        assert isinstance(scope, ast.With)
        assert len(scope.body) == 2 and scope.body[0] is helper
        invocation = scope.body[1]
        assert isinstance(invocation, ast.Expr) and isinstance(invocation.value, ast.Call)
        assert isinstance(invocation.value.func, ast.Name)
        assert invocation.value.func.id == helper.name
        assert invocation.value.args == [] and invocation.value.keywords == []
        assert helper.args.posonlyargs == [] and helper.args.args == []
        assert helper.args.kwonlyargs == []
        assert helper.args.vararg is None and helper.args.kwarg is None
        assert (scope.lineno, invocation.lineno) == (helper.lineno, helper.lineno)
        assert (scope.end_lineno, invocation.end_lineno) == (helper.end_lineno, helper.end_lineno)
    assert language.stack == [] and language.source_stack == []
    assert not IRBuilder.is_in_scope()


@pytest.mark.parametrize("fail", [False, True])
def test_nested_helper_frames_capture_parameters_and_unwind(language, fail):
    seen = []
    failure = ValueError("inner body failure")

    def observe(label, outer, current, scope):
        frame = language.frame()
        assert language.stack[-1] is frame
        seen.append((label, frame, outer, current, scope))
        if fail and label == "inner":
            raise failure

    source = """
@X.script
def main(x: X.tensor((4,))):
    scope = 1
    observe("outer_before", x, x, scope)
    @X.script
    def inner(y: X.tensor((4,))):
        scope = 2
        observe("inner", x, y, scope)
    observe("outer_after", x, x, scope)
"""
    if fail:
        with pytest.raises(DiagnosticError) as caught:
            language.parse(source, observe=observe)
        assert caught.value.__cause__ is failure
        source_line = next(
            index for index, line in enumerate(source.splitlines(), 1) if 'observe("inner"' in line
        )
        assert f"dummy.py:{source_line}:" in str(caught.value)
    else:
        result = language.parse(source, observe=observe)
        assert result is language.functions["main"]

    entries = [event for event in language.events if event[:2] == ("enter", "function")]
    exits = [event for event in language.events if event[:2] == ("exit", "function")]
    assert [event[2] for event in entries] == [False, True, False]
    outer, inner = entries[0][3], entries[1][3]
    assert entries[2][3] is inner
    assert [event[3] for event in exits] == [inner, inner, outer]
    assert len(outer.params) == len(inner.params) == 1
    assert seen[0] == ("outer_before", outer, outer.params[0], outer.params[0], 1)
    assert seen[1] == ("inner", inner, outer.params[0], inner.params[0], 2)
    if fail:
        assert len(seen) == 2
    else:
        assert seen[2] == ("outer_after", outer, outer.params[0], outer.params[0], 1)
    assert language.stack == [] and language.source_stack == []
    assert not IRBuilder.is_in_scope()


def test_zero_argument_helper_retains_real_closure_defaults(language, monkeypatch):
    token = object()
    X = language.X
    helpers = []
    original = entry.recompose_builder

    def capture(translated, **kwargs):
        result = original(translated, **kwargs)
        helpers.extend(node for node in ast.walk(translated) if hasattr(node, "_tvm_source_name"))
        return result

    monkeypatch.setattr(entry, "recompose_builder", capture)

    @X.script
    def main(x: X.tensor((4,))):
        X.record(token)
        X.record(x)

    assert main.body[0][1] is token
    assert main.body[1][1] is main.params[0]
    assert len(helpers) == 1
    helper = helpers[0]
    assert helper.args.args == [] and helper.args.posonlyargs == []
    captures = [argument.arg for argument in helper.args.kwonlyargs]
    assert "token" in captures
    assert len(helper.args.kw_defaults) == len(captures)
    assert all(default is not None for default in helper.args.kw_defaults)
