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

import tvm
from tvm import ir
from tvm.script import parser
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder.ir import parser_protocol as protocol


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


@pytest.mark.parametrize("name", ["Range", "StringType", "StringImm", "GenericConst"])
def test_shared_ir_constructor_identity(name):
    # Initialize the source namespace before inspecting its shared constructors.
    parse = parser.parse
    assert callable(parse)
    assert getattr(I, name) is getattr(tvm.ir, name)
    assert getattr(parser.I, name) is getattr(tvm.ir, name)


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


def test_dialect_marker_is_same_protocol_identity():
    assert T.constexpr is I.constexpr
