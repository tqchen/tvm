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
import copy
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


def test_live_jit_retains_only_needed_scope_across_uncached_builds(language):
    # A live deferred owner needs width, but never the unrelated enclosing payload.
    from tvm.tirx.script.jit import make_jit

    X = language.X
    X.jit = make_jit(X)

    def make():
        payload = Payload()
        reference = weakref.ref(payload)
        width = 7

        @X.jit
        def kernel(x: X.tensor((width,)), *, value: I.constexpr):
            X.record(value)

        return kernel, reference

    with without_cyclic_gc():
        kernel, reference = make()
        assert reference() is None
        first = kernel.specialize(value=1)
        second = kernel.specialize(value=2)
        assert reference() is None
        assert first is kernel.specialize(value=1) and first is not second
        for function, value in ((first, 1), (second, 2)):
            assert function.params[0].args[0].args[0] == (7,)
            assert function.body == [("emit", value)]


def test_reentrant_specialization_restores_root_bindings_after_failure(language):
    # An ordinary nested parse of the same name must not inherit selected values;
    # a failing nested specialization must restore the outer selection unchanged.
    from tvm.script.parser import jit_support

    source = "@X.script\ndef kernel(n: X.tensor((1,))):\n    X.record(n)\n"
    seen = []
    failure = ValueError("nested failure")

    def fail():
        raise failure

    def nested():
        ordinary = entry.parse(source, extra_vars={"X": language.X})
        assert len(ordinary.params) == 1
        assert ordinary.body[0][1] is ordinary.params[0]
        with pytest.raises(ValueError) as caught:
            entry.parse(
                "@X.script\ndef kernel(n: I.constexpr):\n    fail()\n",
                extra_vars={"X": language.X, "I": I, "fail": fail},
                _specialization_bindings={"n": 8},
            )
        assert caught.value is failure
        seen.append(jit_support.read_specialization_bindings("kernel"))

    result = entry.parse(
        "@X.script\ndef kernel(n: I.constexpr):\n    nested()\n    X.record(n)\n",
        extra_vars={"X": language.X, "I": I, "nested": nested},
        _specialization_bindings={"n": 4},
    )
    assert result.params == [] and result.body[-1] == ("emit", 4)
    assert seen == [{"n": 4}]
    assert jit_support.read_specialization_bindings("kernel") is None


@pytest.mark.parametrize("module", [False, True])
def test_eager_entry_releases_unused_scope_but_keeps_annotation_value(
    language, module, monkeypatch
):
    # Before: unrelated payload and annotation-only width share an enclosing scope.
    # Expected builder program: use width, then drop the temporary definition scope.
    X = language.X
    from tvm.script.parser import protocol_registry as registry

    source_references = []
    copy_info = registry.copy_function_info

    def observe_copy(source, target):
        source_references.append(weakref.ref(target))
        return copy_info(source, target)

    monkeypatch.setattr(registry, "copy_function_info", observe_copy)

    def temporary_metadata():
        captured, option = Payload(), Payload()

        def source():
            return captured

        registry.copy_function_info(X.script, source)
        registry.register_function_options(source, {"temporary": option})
        assert registry.function_info(source) is registry.function_info(X.script)
        assert registry.get_function_options(source)["temporary"] is option
        assert vars(source) == {}
        return weakref.ref(source), weakref.ref(captured), weakref.ref(option)

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
        metadata_references = temporary_metadata()
        assert all(reference() is None for reference in metadata_references)
        result, reference = make()
        assert reference() is None
        # An ordinary Python class itself has cyclic type/MRO ownership. Its
        # member lifetime is not a registry leak; the raw copied-function guard
        # above isolates that requirement without requiring class collection.
        if not module:
            assert source_references and all(reference() is None for reference in source_references)
        function = result["main"] if module else result
        assert function.params[0].args[0].args[0] == (7,)


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("macro", [False, True])
def test_acquired_ast_context_and_builder_are_temporary(language, monkeypatch, failure, macro):
    # Before: parse a function or invoke a statement macro, succeeding or raising.
    # Expected builder program: consume each freshly acquired tree directly; release
    # syntax, contexts and private builders without cyclic GC, on either exit path.
    X = language.X

    def finish():
        if failure:
            raise NameError("body failure")
        return 1

    @entry.make_macro_decorator(X)
    def helper():
        X.record(finish())

    body = "helper()" if macro else "X.record(finish())"
    source = "@X.script\ndef main():\n    " + body + "\n"
    references, acquired, parse_calls = [], [], []
    acquire, prepare, recompose = (
        entry.acquire_source,
        entry._prepare_transpiler,
        entry._recompose_builder,
    )
    parse_ast, shallow_copy, deep_copy = ast.parse, copy.copy, copy.deepcopy

    def capture_acquire(*args, **kwargs):
        tree, filename, flags = acquire(*args, **kwargs)
        acquired.append(weakref.ref(tree))
        references.append(weakref.ref(tree))
        return tree, filename, flags

    def capture_prepare(tree, *args, **kwargs):
        assert tree is acquired[-1]()
        transformer, namespace = prepare(tree, *args, **kwargs)
        references.extend(
            [
                weakref.ref(transformer),
                weakref.ref(transformer.module),
                weakref.ref(transformer.function),
            ]
        )
        return transformer, namespace

    def capture_recompose(translated, **kwargs):
        assert all(
            not any(name.startswith("_tvm_") for name in vars(node))
            for node in ast.walk(translated)
        )
        builder = recompose(translated, **kwargs)
        references.append(weakref.ref(builder))
        return builder

    def parse_once(*args, **kwargs):
        parse_calls.append(None)
        return parse_ast(*args, **kwargs)

    def forbid_ast_copy(value, *args, **kwargs):
        assert not isinstance(value, ast.AST), "the acquired tree is already owned"
        return shallow_copy(value, *args, **kwargs)

    def forbid_ast_deepcopy(value, *args, **kwargs):
        assert not isinstance(value, ast.AST), "the acquired tree is already owned"
        return deep_copy(value, *args, **kwargs)

    monkeypatch.setattr(entry, "acquire_source", capture_acquire)
    monkeypatch.setattr(entry, "_prepare_transpiler", capture_prepare)
    monkeypatch.setattr(entry, "_recompose_builder", capture_recompose)
    monkeypatch.setattr(ast, "parse", parse_once)
    monkeypatch.setattr(copy, "copy", forbid_ast_copy)
    monkeypatch.setattr(copy, "deepcopy", forbid_ast_deepcopy)
    with without_cyclic_gc():
        for _ in range(2):
            if failure:
                try:
                    language.parse(source, finish=finish, helper=helper)
                except NameError:
                    pass
                else:
                    pytest.fail("expected failed construction")
            else:
                result = language.parse(source, finish=finish, helper=helper)
                assert result.body[0] == ("emit", 1)
            assert references and all(reference() is None for reference in references)
        assert len(acquired) == (4 if macro else 2)
        # This source contains no expression-string policy or extra source objects.
        # Each acquisition parses once; no second parse substitutes for an AST copy.
        assert len(parse_calls) == len(acquired)


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
