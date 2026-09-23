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
"""Temporary parser ownership must release captures without cyclic collection."""

from __future__ import annotations

import ast
import gc
import inspect
import weakref
from contextlib import contextmanager

import pytest

from tvm.script import ir as I
from tvm.script.parser import entry


class Payload:
    pass


@contextmanager
def without_cyclic_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("module", [False, True])
def test_eager_entry_releases_unused_scope_but_keeps_annotation_value(language, module):
    # Before: unrelated payload and annotation-only width share an enclosing scope.
    # Expected builder program: use width, then drop the temporary definition scope.
    X = language.X

    def make():
        payload = Payload()
        reference = weakref.ref(payload)
        width = 7
        if module:

            @I.ir_module
            class Result:
                @X.script
                def main(value: X.tensor((width,))):
                    X.record(1)
        else:

            @X.script
            def Result(value: X.tensor((width,))):
                X.record(1)

        return Result, reference

    with without_cyclic_gc():
        result, reference = make()
        assert reference() is None
        function = result["main"] if module else result
        assert function.params[0].args[0].args[0] == (7,)


@pytest.mark.parametrize("failure", [False, True])
def test_owned_ast_context_and_builder_are_temporary(language, monkeypatch, failure):
    # Before: parse one original AST, either returning normally or failing in body execution.
    # Expected builder program: only copied nodes change; no context/builder cycle survives.
    source = "@X.script\ndef main():\n    " + ("missing()" if failure else "X.record(1)") + "\n"
    original, filename, flags = entry.acquire_source(source, filename="lifetime.py")
    original_fields = [(node, dict(vars(node))) for node in ast.walk(original)]
    original_dump = ast.dump(original, include_attributes=True)
    references = []
    prepare = entry._prepare_transpiler
    recompose = entry._recompose_builder

    def capture_prepare(tree, *args, **kwargs):
        transformer, namespace = prepare(tree, *args, **kwargs)
        references.extend(
            [
                weakref.ref(tree),
                weakref.ref(transformer),
                weakref.ref(transformer.module),
                weakref.ref(transformer.function),
            ]
        )
        return transformer, namespace

    def capture_recompose(*args, **kwargs):
        builder = recompose(*args, **kwargs)
        references.append(weakref.ref(builder))
        return builder

    monkeypatch.setattr(
        entry, "acquire_source", lambda *args, **kwargs: (original, filename, flags)
    )
    monkeypatch.setattr(entry, "_prepare_transpiler", capture_prepare)
    monkeypatch.setattr(entry, "_recompose_builder", capture_recompose)
    with without_cyclic_gc():
        if failure:
            try:
                language.parse(source)
            except NameError:
                pass
            else:
                pytest.fail("expected failed construction")
        else:
            result = language.parse(source)
            assert result.body == [("emit", 1)]
        assert references and all(reference() is None for reference in references)
        assert ast.dump(original, include_attributes=True) == original_dump
        assert all(vars(node) == fields for node, fields in original_fields)
        assert all(
            not any(name.startswith("_tvm_") for name in vars(node)) for node in ast.walk(original)
        )


class FixedFailure(ValueError):
    def __setattr__(self, name, value):
        if name == "__tvm_script_location__":
            raise AttributeError("exception metadata is immutable")
        super().__setattr__(name, value)


@pytest.mark.parametrize("error_type", [ValueError, FixedFailure])
def test_expression_exception_keeps_original_source_call_traceback(spanned_language, error_type):
    # Before: a normal Python helper fails within a generated function body.
    # Expected builder program: its real call frame retains the source expression range.
    import traceback

    # Exercise the production shared span boundary as well as the generated call.
    language = spanned_language
    original = error_type("original expression failure")

    def explode():
        raise original

    source = "@X.script\ndef main():\n    X.record(explode())\n"
    filename = "expression_traceback.py"
    with pytest.raises(error_type) as caught:
        entry.parse(source, extra_vars={"X": language.X, "explode": explode}, filename=filename)
    assert caught.value is original
    assert type(caught.value) is error_type
    frames = traceback.extract_tb(caught.value.__traceback__)
    calls = [frame for frame in frames if frame.filename == filename and frame.lineno == 3]
    assert calls
    helper_frames = [frame for frame in frames if frame.name == "explode"]
    assert len(helper_frames) == 1
    assert helper_frames[0].filename == inspect.getsourcefile(explode)
    assert helper_frames[0].lineno == inspect.getsourcelines(explode)[1] + 1
    if getattr(calls[-1], "colno", None) is not None:
        assert (calls[-1].colno, calls[-1].end_lineno, calls[-1].end_colno) == (13, 3, 22)


def test_live_macro_does_not_snapshot_unrelated_module_globals(language):
    # Before: an unrelated module global exists while a semantic closure is decorated.
    # Expected builder program: retain the closure value, release the removed global.
    key = "_unrelated_macro_payload"
    globals()[key] = Payload()
    reference = weakref.ref(globals()[key])
    decorator = entry.make_macro_decorator(language.X)
    offset = 3
    try:
        with without_cyclic_gc():

            @decorator
            def helper(value):
                return value + offset

            del globals()[key]
            assert reference() is None
            result = language.parse(
                "@X.script\ndef main():\n    X.record(helper(2))\n", helper=helper
            )
            assert result.body == [("emit", 5)]
    finally:
        globals().pop(key, None)


@pytest.mark.parametrize("lookup", ["globals", "eval"])
def test_macro_dynamic_global_lookup_uses_temporary_invocation_namespace(language, lookup):
    # Before: the macro dynamically names a global that has no LOAD_GLOBAL instruction.
    # Expected builder program: it is available during invocation without a stored snapshot.
    key = "_macro_dynamic_value"
    globals()[key] = 7
    decorator = entry.make_macro_decorator(language.X)
    try:
        if lookup == "globals":

            @decorator
            def helper():
                return globals()["_macro_dynamic_value"]
        else:

            @decorator
            def helper():
                return eval("_macro_dynamic_value")

        result = language.parse("@X.script\ndef main():\n    X.record(helper())\n", helper=helper)
        assert result.body == [("emit", 7)]
    finally:
        del globals()[key]
