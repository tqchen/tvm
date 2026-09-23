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
"""Native function and module resolver ownership."""

from typing import TypeVar

import pytest

from tvm import ir
from tvm.relax.script import builder as R
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder.ir import parser_protocol as protocol
from tvm.tirx.script import builder as T


def test_global_info_requires_module_and_keeps_objects():
    # Before: X.Tensor((4,), "float32", vdevice="cuda:1")
    # Expected builder program: X.Tensor((4,), "float32", I.resolve_global_info_("cuda:1"))
    # Without an enclosing module, a string lookup reports a missing context.
    marker = object()
    assert I.resolve_global_info_(marker) is marker
    for active_builder in (False, True):
        if active_builder:
            with IRBuilder(), T.function():
                with pytest.raises(ValueError, match="module"):
                    I.resolve_global_info_("mesh[0]")
        else:
            with pytest.raises(ValueError, match="module"):
                I.resolve_global_info_("cuda:1")


@pytest.mark.parametrize("dialect", [T])
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
