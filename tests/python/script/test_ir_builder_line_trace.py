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
"""Executed-line locations for imperative builder construction."""

import sys

import pytest

from tvm import ir
from tvm.script import ir_builder as I
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script import ir_builder as T


def _lines(loc):
    if isinstance(loc, ir.CallSiteLoc):
        return _lines(loc.caller) + _lines(loc.callee)
    if isinstance(loc, ir.UnknownLoc):
        return []
    assert isinstance(loc, ir.SourceLoc)
    return [(loc.source_name.name, loc.start_line)]


def _source(line):
    return (__file__, line)


def test_requires_builder():
    with pytest.raises(RuntimeError, match="active IRBuilder"):
        with I.line_trace():
            pass


def test_inline_body_and_normal_cleanup():
    previous_trace = sys.gettrace()
    with IRBuilder() as builder:
        with I.line_trace() as context:
            assert context is None
            expected = sys._getframe().f_lineno + 1
            receipt = T.evaluate(1)
        assert _lines(receipt.value.loc) == [_source(expected)]
        assert builder.get().same_as(receipt.value)
    assert sys.gettrace() is previous_trace
    # A subsequent builder operation has no stale location.
    with IRBuilder():
        assert isinstance(T.evaluate(2).value.loc, ir.UnknownLoc)


def test_helper_return_on_same_line():
    expected = []

    def helper():
        expected.append(sys._getframe().f_lineno + 1)
        return T.evaluate(1)

    with IRBuilder():
        with I.line_trace():
            caller_line = sys._getframe().f_lineno + 1
            first, second = helper(), T.evaluate(2)
    assert _lines(first.value.loc) == [_source(expected[0])]
    assert _lines(second.value.loc) == [_source(caller_line)]


def test_explicit_location_scopes_and_attachment():
    outer = ir.SourceLoc(ir.SourceName("outer.py"), 1, 1, 1, 5)
    explicit = ir.SourceLoc(ir.SourceName("explicit.py"), 7, 1, 7, 5)
    attached = ir.SourceLoc(ir.SourceName("attached.py"), 9, 1, 9, 5)
    with IRBuilder() as builder:
        with builder.with_loc(outer):
            with I.line_trace():
                with builder.with_loc(explicit):
                    first_line = sys._getframe().f_lineno + 1
                    first = T.evaluate(1)
                    second_line = sys._getframe().f_lineno + 1
                    second = I.at_(attached, T.evaluate(2))
                third_line = sys._getframe().f_lineno + 1
                third = T.evaluate(3)
            after = T.evaluate(4)
    assert _lines(first.value.loc) == [("outer.py", 1), _source(first_line), ("explicit.py", 7)]
    assert _lines(second.value.loc) == [
        ("outer.py", 1),
        _source(second_line),
        ("explicit.py", 7),
        ("attached.py", 9),
    ]
    assert _lines(third.value.loc) == [("outer.py", 1), _source(third_line)]
    assert _lines(after.value.loc) == [("outer.py", 1)]


def test_nested_scopes_share_callback_and_restore_locations():
    with IRBuilder():
        with I.line_trace():
            callback = sys.gettrace()
            nested_line = sys._getframe().f_lineno + 1
            with I.line_trace():
                assert sys.gettrace() is callback
                inner_line = sys._getframe().f_lineno + 1
                inner = T.evaluate(1)
            outer_line = sys._getframe().f_lineno + 1
            outer = T.evaluate(2)
    assert _lines(inner.value.loc) == [_source(nested_line), _source(inner_line)]
    assert _lines(outer.value.loc) == [_source(outer_line)]


def test_nested_builders_require_their_own_scope():
    with IRBuilder():
        with I.line_trace():
            with IRBuilder():
                untraced = T.evaluate(0)
                with I.line_trace():
                    inner_line = sys._getframe().f_lineno + 1
                    inner = T.evaluate(1)
            outer_line = sys._getframe().f_lineno + 1
            outer = T.evaluate(2)
    assert isinstance(untraced.value.loc, ir.UnknownLoc)
    assert _lines(inner.value.loc) == [_source(inner_line)]
    assert _lines(outer.value.loc) == [_source(outer_line)]


def test_error_restores_nested_trace_and_explicit_location():
    previous_trace = sys.gettrace()
    marker = ir.SourceLoc(ir.SourceName("saved.py"), 2, 1, 2, 5)
    with IRBuilder() as builder:
        with builder.with_loc(marker):
            with pytest.raises(ValueError, match="failure"):
                with I.line_trace():
                    callback = sys.gettrace()
                    try:
                        with I.line_trace():
                            raise ValueError("failure")
                    finally:
                        assert sys.gettrace() is callback
            assert sys.gettrace() is previous_trace
            after = T.evaluate(1)
    assert _lines(after.value.loc) == [("saved.py", 2)]


def test_existing_global_and_local_trace_state_is_restored():
    events = []

    def local_trace(frame, event, arg):
        events.append((frame.f_code.co_name, event))
        return local_trace

    def global_trace(frame, event, arg):
        return local_trace

    frame = sys._getframe()
    previous = sys.gettrace()
    saved = (frame.f_trace, frame.f_trace_lines, frame.f_trace_opcodes)
    try:
        sys.settrace(global_trace)
        frame.f_trace = local_trace
        frame.f_trace_lines = False
        frame.f_trace_opcodes = True
        with IRBuilder():
            with pytest.raises(ValueError, match="restore"):
                with I.line_trace():
                    events.clear()
                    expected = sys._getframe().f_lineno + 1
                    receipt = T.evaluate(1)
                    assert not events
                    raise ValueError("restore")
            assert sys.gettrace() is global_trace
            assert frame.f_trace is local_trace
            assert frame.f_trace_lines is False
            assert frame.f_trace_opcodes is True
        assert _lines(receipt.value.loc) == [_source(expected)]
    finally:
        sys.settrace(previous)
        frame.f_trace, frame.f_trace_lines, frame.f_trace_opcodes = saved


def test_synthetic_helper_keeps_caller_line():
    namespace = {"T": T}
    exec(compile("def helper():\n    return T.evaluate(1)\n", "<generated>", "exec"), namespace)
    with IRBuilder():
        with I.line_trace():
            caller_line = sys._getframe().f_lineno + 1
            receipt = namespace["helper"]()
    assert _lines(receipt.value.loc) == [_source(caller_line)]


def test_frame_keeps_construction_location():
    with IRBuilder() as builder:
        with I.line_trace():
            frame_line = sys._getframe().f_lineno + 1
            frame = T.serial(4)
            with frame:
                body_line = sys._getframe().f_lineno + 1
                receipt = T.evaluate(1)
    loop = builder.get()
    assert _lines(loop.loc) == [_source(frame_line)]
    assert loop.body.seq[0].same_as(receipt.value)
    assert _lines(loop.body.seq[0].loc) == [_source(body_line)]


def test_untraceable_caller_does_not_retain_helper_location():
    def helper():
        return T.evaluate(1)

    namespace = {"T": T, "I": I, "helper": helper}
    exec(
        compile(
            "def build():\n"
            "    with I.line_trace():\n"
            "        helper()\n"
            "        return T.evaluate(2)\n",
            "<generated>",
            "exec",
        ),
        namespace,
    )
    with IRBuilder():
        receipt = namespace["build"]()
    assert isinstance(receipt.value.loc, ir.UnknownLoc)


def test_generator_helper_restores_local_state():
    def values():
        yield T.evaluate(1)
        yield T.evaluate(2)

    generator = values()
    frame = generator.gi_frame
    saved = (frame.f_trace, frame.f_trace_lines, frame.f_trace_opcodes)
    with IRBuilder():
        with I.line_trace():
            first = next(generator)
            second = next(generator)
            assert first.value.loc.start_line + 1 == second.value.loc.start_line
        assert (frame.f_trace, frame.f_trace_lines, frame.f_trace_opcodes) == saved
    generator.close()


def test_existing_hook_is_suspended_for_synthetic_caller():
    events = []

    def trace(frame, event, arg):
        events.append(event)
        return trace

    namespace = {"T": T, "I": I, "events": events}
    exec(
        compile(
            "def build():\n"
            "    with I.line_trace():\n"
            "        events.clear()\n"
            "        return T.evaluate(1), bool(events)\n",
            "<generated>",
            "exec",
        ),
        namespace,
    )
    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        with IRBuilder():
            receipt, observed = namespace["build"]()
        assert sys.gettrace() is trace
    finally:
        sys.settrace(previous)
    assert not observed
    assert isinstance(receipt.value.loc, ir.UnknownLoc)
