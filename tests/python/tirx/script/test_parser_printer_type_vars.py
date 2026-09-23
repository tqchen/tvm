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

"""Printed symbolic buffer metadata survives parsing and Python decoration."""

import sys
import types
from typing import TypeVar

import pytest

import tvm
from tvm.script import ir as I
from tvm.script import tirx as T

_PRINT_MODES = [False, True] if sys.version_info >= (3, 12) else [False]


def _symbolic_buffer_function(field):
    annotations = {
        "layout": 'T.Buffer(("n", "n + 1", N), "float32")',
        "layout_allocated_addr": 'T.Buffer(("n", "n + 1", N), "float32", allocated_addr=0)',
        "layout_allocated_addr_tuple": (
            'T.Buffer(("n", "n + 1", N), "float32", allocated_addr=(0, 16))'
        ),
        "strides": 'T.Buffer((N, 4), "float32", strides=("n + 1", 1), layout=None)',
        "elem_offset": 'T.Buffer((N, 4), "float32", elem_offset="n + 1", layout=None)',
        "allocated_addr": 'T.Buffer((N, 4), "float32", allocated_addr="n + 1", layout=None)',
    }
    return tvm.script.from_source(
        'N = TypeVar("N")\n@T.prim_func\n'
        f"def main(A: {annotations[field]}, n: T.int32):\n    T.evaluate(n)\n"
    )


def _execute_printed(source, tmp_path, monkeypatch):
    # The printer leaves TVM imports as comments. Supply those documented names,
    # while compiling the unchanged source with its own future-import semantics.
    path = tmp_path / "printed_buffer.py"
    path.write_text(source)
    module = types.ModuleType("_printed_buffer")
    module.__file__ = str(path)
    module.__dict__.update(T=T, I=I, TypeVar=TypeVar)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


@pytest.mark.parametrize(
    "field",
    [
        "layout",
        "layout_allocated_addr",
        "layout_allocated_addr_tuple",
        "strides",
        "elem_offset",
        "allocated_addr",
    ],
)
@pytest.mark.parametrize(
    "pep695", _PRINT_MODES, ids=lambda value: "pep695" if value else "portable"
)
@pytest.mark.parametrize("in_module", [False, True], ids=["function", "module"])
@pytest.mark.parametrize("execute", [False, True], ids=["from_source", "eager_exec"])
def test_symbolic_buffer_fields_roundtrip_with_later_scalar_parameter(
    tmp_path, monkeypatch, field, pep695, in_module, execute
):
    # Before: def main[N](A: X.Buffer(("n", "n + 1", N), ...), n: X.int32): ...
    # Expected builder program: declare n in the native symbol map, then build
    # X.arg("A", ...) and X.arg("n", ...) without an early Python n binding.
    # Printed non-string layout expressions use A_handle: X.handle followed by
    # A = X.match_buffer(A_handle, ...) in the body. Other symbolic fields stay
    # quoted in X.Buffer annotations. Both forms preserve the exact native IR.
    function = _symbolic_buffer_function(field)
    has_layout = field.startswith("layout")
    original = tvm.IRModule({"main": function}) if in_module else function
    printed = original.script(extra_config={"script.use_pep695": pep695})
    assert ("from __future__ import annotations" in printed) is pep695
    assert ("def main[N](" in printed) is pep695
    assert ('N = TypeVar("N")' in printed) is not pep695
    if has_layout:
        assert "A_handle: T.handle" in printed
        assert "A = T.match_buffer(A_handle," in printed
        assert "layout=T.TileLayout(" in printed
    else:
        assert "A: T.Buffer(" in printed
        assert "T.match_buffer(" not in printed
        assert f"{field}=" in printed
        assert '"n + 1"' in printed
    if execute:
        namespace = _execute_printed(printed, tmp_path, monkeypatch)
        actual = namespace.Module if in_module else namespace.main
    else:
        actual = tvm.script.from_source(printed)
    tvm.ir.assert_structural_equal(original, actual)
    actual_function = actual["main"] if in_module else actual
    buffer, n = actual_function.params
    assert str(n.ty.dtype) == "int32"
    assert actual_function.body.value.same_as(n)
    generic = buffer.ty.shape[2] if has_layout else buffer.ty.shape[0]
    assert str(generic.ty.dtype) == "int64"
    assert not generic.same_as(n)
    if has_layout:
        assert buffer.ty.shape[0].same_as(n)
        expression = buffer.ty.shape[1]
        assert buffer.ty.layout is not None
        expected_addresses = {
            "layout": [],
            "layout_allocated_addr": [0],
            "layout_allocated_addr_tuple": [0, 16],
        }[field]
        assert [int(address) for address in buffer.ty.allocated_addr] == expected_addresses
    elif field == "strides":
        expression = buffer.ty.strides[0]
    elif field == "elem_offset":
        expression = buffer.ty.elem_offset
    else:
        expression = buffer.ty.allocated_addr[0]
    assert expression.a.same_as(n)
    assert int(expression.b) == 1
    assert str(expression.ty.dtype) == "int32"
