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

"""Definition and body scope rules through production parser and recording frames."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tvm.script import ir as I
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script.jit import make_jit

EXTENT = 11
VALUE = 1


@pytest.fixture(autouse=True)
def jit_namespace(language):
    language.X.jit = make_jit(language.X)


def _delayed(n, decorator, X):
    @decorator
    def function(output: X.tensor((n,), "int32")):
        local_n = 2
        X.record(local_n + VALUE)

    return function


@pytest.mark.parametrize("factory", [False, True])
def test_reused_eager_decorator_captures_application_scope(factory, language):
    # Before: a reused plain/factory decorator is applied where n=4, before body local_n=2.
    # Expected builder: X.arg receives extent 4 from application scope.
    X = language.X
    decorator = X.script() if factory else X.script
    function = _delayed(4, decorator, X)
    assert [int(value) for value in function.params[0].args[0].args[0]] == [4]


def test_ir_module_class_annotation_scope_does_not_replace_body_global(language):
    # Before: class EXTENT=3 appears in an annotation and a body global read.
    # Expected builder: X.arg receives 3; X.record receives global 11.
    X = language.X

    @I.ir_module
    class Module:
        EXTENT = 3

        @X.script
        def function(output: X.tensor((EXTENT,), "int32")):
            X.record(EXTENT)

    function = Module["function"]
    assert [int(value) for value in function.params[0].args[0].args[0]] == [3]
    assert function.body[0][1] == 11


def test_ordinary_class_annotation_scope_does_not_replace_body_global(language):
    # Before: an ordinary class shadows global EXTENT only for annotations.
    # Expected builder: eager X.arg receives 3; X.record receives global 11.
    X = language.X

    class Holder:
        EXTENT = 3

        @X.script
        def function(output: X.tensor((EXTENT,), "int32")):
            X.record(EXTENT)

    assert [int(value) for value in Holder.function.params[0].args[0].args[0]] == [3]
    assert Holder.function.body[0][1] == 11


def test_jit_annotation_snapshot_precedes_later_closure_mutation(language):
    # Before: a delayed annotation/body capture n=4 before the closure changes to 9.
    # Expected builder: specialization uses the retained 4 in both places.
    X = language.X

    def build():
        n = 4

        @X.jit
        def function(output: X.tensor((n,), "int32")):
            X.record(n)

        n = 9
        return function

    function = build().specialize()
    assert [int(value) for value in function.params[0].args[0].args[0]] == [4]
    assert function.body[0][1] == 4


def test_annotation_constructor_executes_in_required_builder_context(language):
    # Before: a postponed annotation calls a context-sensitive constructor.
    # Expected builder: constructor runs once at specialization inside the real IRBuilder.
    X = language.X
    calls = []

    def annotation():
        calls.append(IRBuilder.is_in_scope())
        return X.tensor((5,), "int32")

    @X.jit
    def function(output: annotation()):
        X.record(7)

    assert calls == []  # postponed annotations are not evaluated at decoration
    result = function.specialize()
    assert calls == [True]
    assert [int(value) for value in result.params[0].args[0].args[0]] == [5]


def test_enclosing_namespace_used_only_by_annotation_is_retained(language):
    # Before: an annotation-only namespace creates symbolic shape n after scope exit.
    # Expected builder: resolve_type_var_("n") supplies the named annotation symbol.
    X = language.X

    def build():
        dtype_namespace = X

        @X.jit
        def function(output: dtype_namespace.tensor(("n",), "int32")):
            X.record(7)

        return function

    result = build().specialize()
    assert result.params[0].args[0].args[0][0].op == "symbol"
    assert result.params[0].args[0].args[0][0].name == "n"


def test_attribute_name_does_not_turn_a_closure_binding_into_a_global(language):
    # Before: closure dtype="int32" and descriptor.dtype="int64" share a spelling.
    # Expected builder: X.record receives ("int32", 3), then ("int64", 4).
    X = language.X
    dtype = "int32"
    descriptor = SimpleNamespace(dtype="int64")

    @X.script
    def function():
        X.record((dtype, 3))
        X.record((descriptor.dtype, 4))

    first, second = function.body
    assert first[1][0] == "int32"
    assert first[1][1] == 3
    assert second[1][0] == "int64"
    assert second[1][1] == 4


def test_active_lexical_ancestor_retains_annotation_only_names(language):
    # Before: nested annotation-only extent/width come from two active ancestors.
    # Expected builder: X.arg receives (extent, 3) for both extent=4 and extent=8.
    X = language.X

    def outer(extent):
        def middle(width):
            @X.script
            def function(output: X.tensor((extent, width), "int32")):
                X.record(VALUE)

            return function

        return middle(3)

    for extent in (4, 8):
        function = outer(extent)
        assert [int(value) for value in function.params[0].args[0].args[0]] == [extent, 3]
        assert function.body[0][1] == 1


def test_active_lexical_ancestor_respects_nearest_annotation_binding(language):
    # Before: inner extent=3 shadows outer extent=8 in a signature.
    # Expected builder: X.arg receives nearest extent 3.
    X = language.X

    def outer(extent):
        def middle():
            extent = 3

            @X.script
            def function(output: X.tensor((extent,), "int32")):
                X.record(VALUE)

            return function

        return middle()

    function = outer(8)
    assert [int(value) for value in function.params[0].args[0].args[0]] == [3]


def test_unrelated_same_file_caller_does_not_supply_annotation_locals(language):
    # Before: unrelated caller locals shadow both annotation and body global names.
    # Expected builder: X.arg receives global 11; X.record receives global 1.
    X = language.X

    def build():
        @X.script
        def function(output: X.tensor((EXTENT,), "int32")):
            X.record(VALUE)

        return function

    def caller():
        EXTENT = 99  # noqa: F841
        VALUE = 99  # noqa: F841
        return build()

    function = caller()
    assert [int(value) for value in function.params[0].args[0].args[0]] == [11]
    assert function.body[0][1] == 1


def test_inactive_lexical_ancestor_is_not_replaced_by_unrelated_caller(language):
    # Before: annotation-only extent has left scope while an unrelated caller has extent.
    # Expected builder: raise NameError naming extent; do not capture the caller.
    X = language.X

    def outer(extent):
        def middle():
            @X.script
            def function(output: X.tensor((extent,), "int32")):
                X.record(VALUE)

            return function

        return middle

    middle = outer(8)
    extent = 99  # noqa: F841
    with pytest.raises(NameError, match="extent"):
        middle()


def test_enclosing_class_locals_are_not_nested_annotation_scope(language):
    # Before: outer class EXTENT=99 surrounds a nested class definition.
    # Expected builder: nested annotation and body read global 11.
    X = language.X

    class Outer:
        EXTENT = 99

        class Inner:
            @X.script
            def function(output: X.tensor((EXTENT,), "int32")):
                X.record(EXTENT)

    function = Outer.Inner.function
    assert [int(value) for value in function.params[0].args[0].args[0]] == [11]
    assert function.body[0][1] == 11


@pytest.mark.parametrize("factory", [False, True])
def test_jit_retains_active_lexical_ancestor_snapshot(factory, language):
    # Before: plain/factory JIT captures extent=8,width=3 before extent changes to 99.
    # Expected builder: delayed X.arg receives (8, 3), X.record receives global 1.
    X = language.X
    decorator = X.jit() if factory else X.jit

    def outer(extent):
        def middle(width):
            @decorator
            def function(output: X.tensor((extent, width), "int32")):
                X.record(VALUE)

            return function

        pending = middle(3)
        extent = 99
        return pending

    function = outer(8).specialize()
    assert [int(value) for value in function.params[0].args[0].args[0]] == [8, 3]
    assert function.body[0][1] == 1


def test_unrelated_intervening_call_ends_annotation_capture(language):
    # Before: an unrelated invoke(callback) frame interrupts lexical ancestor capture.
    # Expected builder: raise NameError naming unavailable extent.
    X = language.X

    def invoke(callback):
        return callback()

    def outer(extent):
        def middle():
            @X.script
            def function(output: X.tensor((extent,), "int32")):
                X.record(VALUE)

            return function

        return invoke(middle)

    with pytest.raises(NameError, match="extent"):
        outer(8)


def _annotation_bindings(language):
    bindings = []
    original = language.X.bind_

    def bind(value, *, name=None, ty=None, **kwargs):
        bindings.append((name, value, ty))
        return original(value, name=name, **kwargs)

    language.X.bind_ = bind
    return bindings


def test_local_annotation_keeps_rhs_before_constructor_evaluation(language):
    # Before: y: annotation() = value(x)
    # Expected builder: X.bind_(value(x), name="y", ty=annotation()).
    X = language.X
    events = []
    bindings = _annotation_bindings(language)

    def annotation():
        events.append("annotation")
        assert IRBuilder.is_in_scope()
        return X.tensor((2,), "float32")

    def value(x):
        events.append("rhs")
        return X.value(x, x)

    @X.script
    def function(x: X.tensor((2,), "float32")):
        y: annotation() = value(x)
        return y

    assert events == ["rhs", "annotation"]
    name, rhs, ty = bindings[0]
    assert name == "y" and ty.args[:2] == ((2,), "float32")
    assert rhs.args == (function.params[0], function.params[0])
    assert function.body[0] == ("return", rhs)


def test_local_annotation_scope_does_not_overwrite_ordinary_body_global(language):
    # Before: class EXTENT=3; body reads global EXTENT; y: tensor((EXTENT,), dtype).
    # Expected builder: X.record(11); X.bind_(rhs, ty=X.tensor((3,), "float32")).
    X = language.X
    dtype = "float32"
    bindings = _annotation_bindings(language)

    @I.ir_module
    class Module:
        EXTENT = 3

        @X.script
        def main(x: X.tensor((3,), "float32")):
            X.record(EXTENT)
            y: X.tensor((EXTENT,), dtype) = X.value(x, x)
            return y

    function = Module["main"]
    name, rhs, ty = next(binding for binding in bindings if binding[0] == "y")
    assert function.body[0] == ("emit", 11)
    assert name == "y" and ty.args[:2] == ((3,), "float32")
    assert rhs.args == (function.params[0], function.params[0])
    assert function.body[1] == ("return", rhs)


def test_local_annotation_preserves_lambda_and_comprehension_bindings(language):
    # Before: annotation's lambda dtype and comprehension size shadow captured names.
    # Expected builder: X.bind_(rhs, ty=X.tensor((2,), "float32")).
    X = language.X
    dtype = "int64"  # noqa: F841
    size = 9  # noqa: F841
    sizes = (2,)
    bindings = _annotation_bindings(language)

    @X.script
    def function(x: X.tensor((2,), "float32")):
        y: (lambda dtype: X.tensor(tuple(size for size in sizes), dtype))("float32") = X.value(x, x)
        return y

    name, rhs, ty = bindings[0]
    assert name == "y" and ty.args[:2] == ((2,), "float32")
    assert rhs.args == (function.params[0], function.params[0])
    assert function.body[0] == ("return", rhs)
