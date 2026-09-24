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
"""Common IR constructors and argument policy metadata."""

import pytest

from tvm import ir
from tvm.script import parser
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I


def test_global_info_selectors_use_module_map():
    # Before: X.Tensor((4,), "float32", vdevice="cuda:1")
    # Expected builder program: I.resolve_global_info_("cuda:1") is mod.global_infos["vdevice"][2]
    with IRBuilder(), I.ir_module() as module:
        devices = [I.vdevice("llvm"), I.vdevice("cuda"), I.vdevice("cuda", 1)]
        I.module_global_infos({"vdevice": devices, "other": [I.dummy_global_info()]})
        for spelling, index in (("cuda:1", 2), ("vdevice[1]", 1), ("cuda", 1)):
            resolved = I.resolve_global_info_(spelling)
            assert resolved.__chandle__() == devices[index].__chandle__()
            assert resolved.__chandle__() == module.global_infos["vdevice"][index].__chandle__()
        assert I.resolve_global_info_("other[0]") is not None


def test_module_string_constants_keep_common_constructor_values():
    actual = parser.parse("""
@I.ir_module
class Module:
    I.module_attrs({"tag": I.StringImm("label"), "type": I.StringType()})
""")
    with IRBuilder() as builder, I.ir_module():
        I.module_attrs({"tag": ir.StringImm("label"), "type": ir.StringType()})
    ir.assert_structural_equal(builder.get(), actual)
    assert actual.attrs["tag"].value == "label"
    assert isinstance(actual.attrs["type"], ir.StringType)


def test_constexpr_uses_registered_namespace_in_source():
    class Ordinary:
        constexpr = T.int32

    source = "@T.prim_func\ndef main(value: annotation):\n    T.evaluate(value)\n"
    ordinary = parser.parse(
        source.replace("annotation", "Ordinary.constexpr"), extra_vars={"Ordinary": Ordinary}
    )
    assert len(ordinary.params) == 1 and ordinary.params[0].ty.dtype == "int32"
    with pytest.raises(TypeError, match="requires a specialization binding"):
        parser.parse(source.replace("annotation", "I.constexpr"))


@pytest.mark.parametrize("dialect_name", ["tirx", "relax"])
def test_native_function_reentry_retains_its_symbol_map_and_reference(dialect_name):
    # Source: independently declared functions use the same symbolic spelling.
    # Builder: each native frame owns its map across declaration/body entries.
    from tvm.relax.script import builder as R
    from tvm.tirx.script import builder as B

    dialect = B if dialect_name == "tirx" else R
    with IRBuilder() as builder, I.ir_module() as module:
        with pytest.raises(ValueError, match="function"):
            dialect.resolve_type_var_("n")
        with dialect.function_(decl=True) as frame:
            dialect.func_name("identity")
            n = dialect.resolve_type_var_("n")
            annotation = B.Buffer((n,), "float32") if dialect is B else R.Tensor((n,), "float32")
            parameter = dialect.arg("x", annotation)
            assert frame.type_var_map["n"].same_as(n)
            assert builder.frames[-1].same_as(frame)
        reference = frame.global_var
        assert module.identity.same_as(reference)
        with frame:
            assert dialect.resolve_type_var_("n").same_as(n)
            assert frame.params[0].same_as(parameter)
            # A separate nested build must resolve its own same-spelling symbol.
            with IRBuilder(), dialect.function_() as nested:
                inner_n = dialect.resolve_type_var_("n")
                assert not inner_n.same_as(n)
                assert nested.type_var_map["n"].same_as(inner_n)
                if dialect is B:
                    B.evaluate(0)
                else:
                    R.func_ret_value(R.const(0))
            assert dialect.resolve_type_var_("n").same_as(n)
            if dialect is B:
                B.evaluate(parameter[0])
            else:
                R.func_ret_value(parameter)
        assert frame.global_var.same_as(reference)
        assert frame.function.params[0].same_as(parameter)
    assert builder.get().get_global_var("identity").same_as(reference)
    assert builder.get()["identity"].params[0].same_as(parameter)


@pytest.mark.parametrize("annotation", ["T.Buffer(shape, 'float32')", "R.Tensor(shape, 'float32')"])
def test_captured_shape_requires_concrete_symbols(annotation):
    # Literal source strings are decoded; a captured tuple reaches its concrete
    # constructor unchanged and must already contain native symbols.
    decorator, body = (
        ("T.prim_func", "T.evaluate(0)")
        if annotation.startswith("T.")
        else ("R.function", "return x")
    )
    source = f"@{decorator}\ndef main(x: {annotation}):\n    {body}\n"
    n = ir.Var("n", "int64")
    function = parser.parse(source, extra_vars={"shape": (n, 16)})
    assert function.params[0].ty.shape[0].same_as(n)
    with pytest.raises(
        TypeError, match="^Builder expression arguments require concrete symbols, not strings$"
    ) as caught:
        parser.parse(source, extra_vars={"shape": ("n", 16)})
    assert type(caught.value) is TypeError
