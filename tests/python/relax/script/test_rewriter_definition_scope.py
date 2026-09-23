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
"""Rewriter class parsing owns one temporary original definition scope."""

from __future__ import annotations

import gc
import weakref

import pytest

import tvm
from tvm.relax.dpl import PatternMatchingRewriter
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script.parser import entry, inspect_source


class Configuration:
    def __init__(self, width):
        self.width = width


@pytest.mark.parametrize("failure", [False, True])
def test_rewriter_annotation_scope_is_captured_once_and_released(monkeypatch, failure):
    captures = []
    annotation_calls = []
    capture = inspect_source.capture_definition_scope

    def observe(frame):
        scope = capture(frame)
        captures.append((frame.f_code.co_name, scope["config"].width))
        return scope

    monkeypatch.setattr(inspect_source, "capture_definition_scope", observe)

    def make_rewriter():
        config = Configuration(7)
        unused = Configuration(99)
        references = weakref.ref(config), weakref.ref(unused)

        def annotation_extent():
            annotation_calls.append(config.width)
            if failure:
                raise ValueError("annotation construction failed")
            return config.width

        try:

            @R.rewriter
            class Rewrite:
                @R.function
                def pattern(value: R.Tensor((annotation_extent(),), "float32")):
                    return R.add(value, value)

                @R.function
                def replacement(value: R.Tensor((annotation_extent(),), "float32")):
                    return R.multiply(value, R.const(2, "float32"))
        except ValueError as error:
            assert failure and str(error) == "annotation construction failed"
            return None, references
        assert not failure
        return Rewrite, references

    enabled = gc.isenabled()
    gc.disable()
    try:
        rewriter, references = make_rewriter()
        assert captures == [("make_rewriter", 7)]
        assert annotation_calls == ([7] if failure else [7, 7])
        assert all(reference() is None for reference in references)
    finally:
        if enabled:
            gc.enable()

    if not failure:

        @R.function
        def before(value: R.Tensor((7,), "float32")):
            R.func_attr({"global_symbol": "main"})
            return R.add(value, value)

        @R.function
        def expected(value: R.Tensor((7,), "float32")):
            R.func_attr({"global_symbol": "main"})
            return R.multiply(value, R.const(2, "float32"))

        tvm.ir.assert_structural_equal(rewriter(before), expected)


def test_rewriter_existing_module_does_not_capture_or_parse(monkeypatch):
    @I.ir_module
    class Rewrite:
        @R.function
        def pattern(value: R.Tensor((7,), "float32")):
            return R.add(value, value)

        @R.function
        def replacement(value: R.Tensor((7,), "float32")):
            return R.multiply(value, R.const(2, "float32"))

    def unexpected(*args, **kwargs):
        pytest.fail("an existing module does not need source inspection")

    monkeypatch.setattr(inspect_source, "capture_definition_scope", unexpected)
    monkeypatch.setattr(entry, "parse", unexpected)
    assert isinstance(R.rewriter(Rewrite), PatternMatchingRewriter)
