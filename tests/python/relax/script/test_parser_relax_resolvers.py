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
"""Relax declaration and distributed annotation ownership."""

import pytest

from tvm import ir
from tvm.relax.script import builder as R
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script import builder as T


@pytest.mark.parametrize("dialect", [R])
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
