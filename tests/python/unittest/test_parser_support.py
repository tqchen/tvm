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
"""Native resolver ownership and eager annotation contracts."""

from typing import TypeVar

import pytest

from tvm import ir
from tvm.relax.script import builder as R
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.parser import protocol
from tvm.tirx.script import builder as T


def test_global_info_requires_module_and_keeps_objects():
    # Before: X.Tensor((4,), "float32", vdevice="cuda:1")
    # Expected builder program: X.Tensor((4,), "float32", I.resolve_global_info("cuda:1"))
    # Without an enclosing module, a string lookup reports a missing context.
    marker = object()
    assert I.resolve_global_info(marker) is marker
    for active_builder in (False, True):
        if active_builder:
            with IRBuilder(), T.function():
                with pytest.raises(ValueError, match="module"):
                    I.resolve_global_info("mesh[0]")
        else:
            with pytest.raises(ValueError, match="module"):
                I.resolve_global_info("cuda:1")


def test_global_info_selectors_use_module_map():
    # Before: X.Tensor((4,), "float32", vdevice="cuda:1")
    # Expected builder program: I.resolve_global_info("cuda:1") is mod.global_infos["vdevice"][2]
    with IRBuilder(), I.ir_module() as module:
        devices = [I.vdevice("llvm"), I.vdevice("cuda"), I.vdevice("cuda", 1)]
        I.module_global_infos({"vdevice": devices, "other": [I.dummy_global_info()]})
        for spelling, index in (("cuda:1", 2), ("vdevice[1]", 1), ("cuda", 1)):
            assert I.resolve_global_info(spelling).__chandle__() == devices[index].__chandle__()
            assert (
                module.resolve_global_info(spelling).__chandle__() == devices[index].__chandle__()
            )
        assert I.resolve_global_info("other[0]") is not None


@pytest.mark.parametrize("dialect", [T, R])
def test_symbol_lookup_requires_function_and_reuses_native_identity(dialect):
    # Before: n = X.int64(); X.Tensor(("n",), "float32")
    # Expected builder program: n = X.resolve_type_var_("n", "int64")
    # Repeated lookup delegates to the active fn.type_var_map; absent fn is an error.
    with IRBuilder():
        with pytest.raises(ValueError, match="function"):
            dialect.resolve_type_var_("n")
        with dialect.function() as function:
            n = function.resolve_type_var("n", "int32")
            assert dialect.resolve_type_var_("n").same_as(n)
            assert function.type_var_map["n"].same_as(n)
            explicit = ir.Var("m", "int64")
            assert dialect.resolve_type_var_("m", value=explicit).same_as(explicit)
            assert function.resolve_type_var("m").same_as(explicit)
            if dialect is T:
                T.evaluate(0)
            else:
                R.func_ret_value(R.const(0))


def test_nested_functions_have_distinct_symbol_maps():
    # Before: @X.function outer(..."n"...): @X.function inner(..."n"...): ...
    # Expected builder program: X.resolve_type_var_("n") selects the nearest entered fn.
    with IRBuilder(), R.function() as outer:
        outer_n = R.resolve_type_var_("n")
        with R.function(decl=True, local=True) as inner:
            R.func_name("inner")
            inner_n = R.resolve_type_var_("n")
            assert not inner_n.same_as(outer_n)
            assert inner.type_var_map["n"].same_as(inner_n)
        with inner:
            assert R.resolve_type_var_("n").same_as(inner_n)
            R.func_ret_value(R.const(0))
        assert R.resolve_type_var_("n").same_as(outer_n)
        assert outer.type_var_map["n"].same_as(outer_n)
        R.func_ret_value(R.const(0))


def test_eager_annotations_preserve_missing_type_and_concrete_construction():
    # Before: X.Tensor(("n",)); captured_shape = (n,); X.Tensor(captured_shape)
    # Expected builder program: X.Tensor((X.resolve_type_var_("n"),)); X.Tensor(captured_shape)
    @protocol.args_policy({"shape": "expr_str"}, scalar_strings=False)
    def tensor(shape):
        return shape

    assert tensor(("n",)).is_missing()
    assert tensor((TypeVar("n"),)).is_missing()
    assert tensor((16,)) == (16,)
    with IRBuilder(), T.function() as function:
        n = function.resolve_type_var("n")
        assert tensor((TypeVar("n"),))[0].same_as(n)
        assert tensor((n,))[0].same_as(n)
        with pytest.raises(TypeError, match="concrete symbols"):
            tensor(("n",))
        T.evaluate(0)


def test_annotation_class_supports_union_and_missing_type():
    # Before: X.Annotation("n") | None
    # Expected builder program: X.Annotation((X.resolve_type_var_("n"))) in a function frame.
    @protocol.args_policy({"shape": "expr_str"}, as_type=True)
    def annotation(shape):
        return shape

    assert isinstance(annotation, type)
    assert annotation | None
    assert annotation(16) == 16
    assert annotation("n").is_missing()


def test_generated_module_enters_global_info_scope_before_annotations():
    # Before: I.module_global_infos(info); def f(x: X.Tensor(..., vdevice="cuda:1")): ...
    # Expected builder program: with I.ir_module(): I.module_global_infos(info); X.arg(...)
    module = parser.parse("""
@I.ir_module
class Module:
    I.module_global_infos({
        "vdevice": [I.vdevice("llvm"), I.vdevice("cuda"), I.vdevice("cuda", 1)],
        "mesh": [R.device_mesh((2,), [0, 1])],
    })
    @R.function
    def tensor(x: R.Tensor((4,), "float32", vdevice="cuda:1")):
        return x
    @R.function
    def distributed(x: R.DTensor((4,), "float32", device_mesh="mesh[0]", placement="S[0]")):
        return x
""")
    assert (
        module["tensor"].params[0].ty.vdevice.__chandle__()
        == module.global_infos["vdevice"][2].__chandle__()
    )
    assert (
        module["distributed"].params[0].ty.device_mesh.__chandle__()
        == module.global_infos["mesh"][0].__chandle__()
    )
