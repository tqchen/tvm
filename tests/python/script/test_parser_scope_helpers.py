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

import traceback

import pytest

from tvm.script.ir_builder import IRBuilder
from tvm.script.parser import entry, jit_support


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
        with pytest.raises(ValueError) as caught:
            language.parse(source, observe=observe)
        assert caught.value is failure
        source_line = next(
            index for index, line in enumerate(source.splitlines(), 1) if 'observe("inner"' in line
        )
        assert any(
            frame.filename == "dummy.py" and frame.lineno == source_line
            for frame in traceback.extract_tb(caught.value.__traceback__)
        )
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


def test_body_preserves_capture_and_parameter_identity(language):
    # Before: the body reads an enclosing token and its original runtime parameter.
    # Expected builder program: preserve both identities under their source names,
    # while preserving their original lexical lookup.
    token = object()
    X = language.X

    @X.script
    def main(x: X.tensor((4,))):
        X.record(token)
        X.record(x)

    assert main.body[0][1] is token
    assert main.body[1][1] is main.params[0]


@pytest.mark.parametrize("parameter_count,bindings", [(0, None), (2, None), (2, {})])
def test_parameter_setup_depends_on_explicit_specialization_request(
    language, monkeypatch, parameter_count, bindings
):
    # Before: a recursive signature parsed normally or with explicit empty JIT bindings.
    # Expected builder program: ordinary X.arg/frame.params bindings need no JIT
    # state/read/selectors/iterator; an explicit {} still enables specialization.
    signature = ", ".join(f"{name}: X.tensor((4,))" for name in ("x", "y")[:parameter_count])
    arguments = ", ".join(("x", "y")[:parameter_count])
    requested = bindings is not None
    reads = []
    read_bindings = jit_support.read_specialization_bindings

    def observe_read(name):
        reads.append(name)
        return read_bindings(name)

    monkeypatch.setattr(jit_support, "read_specialization_bindings", observe_read)
    result = entry.parse(
        f"@X.script\ndef main({signature}):\n    main({arguments})\n",
        extra_vars={"X": language.X},
        _specialization_bindings=bindings,
        root_builder=language.X,
    )
    assert reads == (["main"] if requested else [])
    assert len(result.params) == parameter_count
    call = result.body[0][1]
    assert call.args[0] is language.references["main"]
    assert all(actual is expected for actual, expected in zip(call.args[1:], result.params))
    assert len(call.args) == parameter_count + 1
