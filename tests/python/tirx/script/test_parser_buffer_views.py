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
"""Shared buffer views construct native IR through the new builder."""

from functools import wraps
from types import MethodType

import pytest

from tvm import ir, tirx
from tvm.error import DiagnosticError
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder.ir import parser_protocol
from tvm.tirx import script as T
from tvm.tirx.layout import TCol, TLane
from tvm.tirx.script import builder as B


def parse(source, **captures):
    return parser.parse(source, extra_vars={"T": T, "TLane": TLane, "TCol": TCol, **captures})


def statements(function):
    return list(function.body.seq) if isinstance(function.body, tirx.SeqStmt) else [function.body]


def assert_alias(function, shape, dtype="float32", offset=0, physical_index=5):
    nodes = statements(function)
    declaration = [node for node in nodes if isinstance(node, tirx.DeclBuffer)][-1]
    store = nodes[-1]
    assert isinstance(store, tirx.BufferStore)
    assert store.buffer.same_as(declaration.buffer)
    buffer = store.buffer
    assert tuple(int(dim) for dim in buffer.shape) == shape
    assert buffer.dtype == dtype
    assert int(buffer.elem_offset) == offset
    assert store.value.value == 0
    assert all(int(index) == 0 for index in store.indices)
    assert len(store.indices) == len(shape)
    mapped = buffer.layout.apply(*([1] * len(shape)), shape=buffer.shape)
    assert set(mapped) == {"m"}
    assert int(mapped["m"]) == physical_index
    return declaration


@pytest.mark.parametrize(
    "view,index,shape,dtype,offset,physical_index",
    [
        ("A.view(16)", "0", (16,), "float32", 0, 1),
        ('A.view("uint16")', "0, 0", (4, 8), "uint16", 0, 9),
        ("A.permute(1, 0)", "0, 0", (4, 4), "float32", 0, 5),
        ('A.rearrange("a b -> b a")', "0, 0", (4, 4), "float32", 0, 5),
        ('A.rearrange(pattern="a b -> b a")', "0, 0", (4, 4), "float32", 0, 5),
        ("A.local()", "0", (16,), "float32", 0, 1),
        ("A.local(16)", "0", (16,), "float32", 0, 1),
        ("A.sub[1]", "0", (4,), "float32", 4, 1),
        ("A.sub[1:3, :]", "0, 0", (2, 4), "float32", 4, 5),
        ("A.sub[:, 1::2]", "0, 0", (4, 2), "float32", 1, 6),
        ("A.tile(1, (2, 2))[1, :]", "0, 0", (4, 2), "float32", 2, 5),
    ],
)
def test_buffer_view_shape_layout_and_storage(view, index, shape, dtype, offset, physical_index):
    result = parse(f"""
@T.prim_func
def main(A: T.Buffer((4, 4), "float32")):
    B = {view}
    B[{index}] = 0
""")
    declaration = assert_alias(result, shape, dtype, offset, physical_index)
    parent = result.params[0]
    for alias in statements(result):
        if isinstance(alias, tirx.DeclBuffer):
            ir.assert_structural_equal(alias.data, parent.data)
            assert alias.data.args[0].same_as(parent)
            parent = alias.buffer
    if "permute" in view or "rearrange" in view:
        assert int(declaration.buffer.layout.apply(1, 2, shape=shape)["m"]) == 9


@pytest.mark.parametrize("declaration", ["parameter", "match", "decl", "alloc"])
def test_buffer_declaration_boundaries_share_native_methods(declaration):
    if declaration == "parameter":
        signature = 'A: T.Buffer((4, 4), "float32")'
        setup = ""
    elif declaration == "match":
        signature = "a: T.handle"
        setup = '    A = T.match_buffer(a, (4, 4), "float32")\n'
    else:
        signature = ""
        setup = f'    A = T.{declaration}_buffer((4, 4), "float32", scope="local")\n'
    result = parse(f"""
@T.prim_func
def main({signature}):
{setup}    B = A.view(16)
    B[0] = 0
""")
    view = assert_alias(result, (16,), physical_index=1)
    if signature:
        parent = result.params[0]
    else:
        parent = statements(result)[0].buffer
    ir.assert_structural_equal(view.data, parent.data)
    assert view.buffer.view.__func__ is parent.view.__func__


@pytest.mark.parametrize(
    "dtype,base,start,stop,expected", [("float32", 64, 32, 64, 96), ("bfloat16", 256, 64, 96, 288)]
)
def test_tmem_subview_keeps_physical_column_offsets(dtype, base, start, stop, expected):
    result = parse(f"""
@T.prim_func
def main():
    A = T.decl_buffer(
        (64, 128), "{dtype}", scope="tmem", allocated_addr={base},
        layout=T.TileLayout(T.S[(64, 128) : (1 @ TLane, 1 @ TCol)]),
    )
    B = A.sub[:, {start}:{stop}]
    T.evaluate(B[0, 0])
""")
    nodes = statements(result)
    view = nodes[-2].buffer
    assert tuple(int(dim) for dim in view.shape) == (64, stop - start)
    assert view.scope() == "tmem"
    assert int(view.allocated_addr[0]) == expected
    assert int(view.layout.offset.get(TCol, 0)) == 0
    assert nodes[-1].value.source.same_as(view)


def test_native_roundtrip_preserves_buffer_identity_and_shared_methods():
    buffer = tirx.decl_buffer((4, 4), "float32")
    raw = ir.Array([buffer])[0]
    assert raw.same_as(buffer)
    assert type(raw) is type(buffer)
    assert raw.view.__func__ is buffer.view.__func__
    with IRBuilder() as builder:
        with B.function():
            B.func_name("roundtrip")
            view = raw.view(16)
    declaration = builder.get().body
    assert declaration.buffer.same_as(view)
    ir.assert_structural_equal(declaration.data, buffer.data)
    assert tuple(int(dim) for dim in view.shape) == (16,)


@pytest.mark.parametrize(
    "view,error",
    [
        ("A.local(3)", "physical storage span"),
        ("A.sub[-1]", "out of range"),
        ("A.sub[1:8]", "exceeds dim"),
        ("A.tile(1, (2, 2))[:, :]", "picks no factor"),
    ],
)
def test_view_validation(view, error):
    with pytest.raises(DiagnosticError, match=error):
        parse(f"""
@T.prim_func
def main():
    A = T.alloc_buffer((4, 4), "float32", scope="local")
    B = {view}
    T.evaluate(1)
""")


@pytest.mark.parametrize("operation", ["captured.view(16)", "view(16)"])
def test_captured_buffer_and_bound_view_keep_the_native_parameter(operation):
    captured = B.Buffer((4, 4), "float32")
    result = parse(
        f"""
@T.prim_func
def main(A: captured):
    B = {operation}
    B[0] = 0
""",
        captured=captured,
        view=captured.view,
    )
    assert result.params[0].same_as(captured)
    declaration = assert_alias(result, (16,), physical_index=1)
    ir.assert_structural_equal(declaration.data, captured.data)


@pytest.mark.parametrize("expression", ["roundtrip(A).view(16)", "make_view(A)"])
def test_opaque_helper_expression_uses_shared_buffer_methods(expression):
    def roundtrip(buffer):
        return ir.Array([buffer])[0]

    def make_view(buffer):
        return roundtrip(buffer).view(16)

    result = parse(
        f"""
@T.prim_func
def main(A: T.Buffer((4, 4), "float32")):
    B = {expression}
    B[0] = 0
""",
        roundtrip=roundtrip,
        make_view=make_view,
    )
    declaration = assert_alias(result, (16,), physical_index=1)
    ir.assert_structural_equal(declaration.data, result.params[0].data)


@pytest.mark.parametrize("callee", ["captured.operation", "operation"])
@pytest.mark.parametrize(
    "method,arguments,index,shape,physical_index",
    [
        ("view", 'mark("shape", 16)', "0", (16,), 1),
        ("permute", 'mark("first", 1), mark("second", 0)', "0, 0", (4, 4), 5),
        ("rearrange", 'mark("pattern", "a b -> b a")', "0, 0", (4, 4), 5),
        ("local", 'mark("shape", 16)', "0", (16,), 1),
    ],
)
def test_direct_view_keeps_producer_identity_name_span_and_evaluation(
    monkeypatch, callee, method, arguments, index, shape, physical_index
):
    # Source: renamed = captured.operation(mark(...)); renamed[...] = 0
    # Builder: renamed = captured.operation(mark(...)); X.buffer_store(...)
    # The native producer owns the view name/span; assignment does not bind it.
    captured = B.Buffer((4, 4), "float32")
    original = getattr(captured, method).__func__
    seen = []
    produced = []
    observed = []

    @wraps(original)
    def operation(buffer, *args, **kwargs):
        seen.append("producer")
        result = original(buffer, *args, **kwargs)
        result = I.at_(("producer.py", 7, 7, 2, 19), result)
        produced.append((result, result.name, result.span))
        return result

    def mark(name, value):
        seen.append(name)
        return value

    @I.direct_call
    def observe(view):
        observed.append((view, view.span))

    # A real bound method exercises registration lookup through __func__.
    monkeypatch.setattr(type(captured), "operation", operation, raising=False)
    bound = MethodType(operation, captured)
    assert parser_protocol.is_direct_call(bound)
    result = parse(
        f"""
@T.prim_func
def main(A: captured):
    renamed = {callee}({arguments})
    observe(renamed)
    renamed[{index}] = 0
""",
        captured=captured,
        operation=bound,
        mark=mark,
        observe=observe,
    )
    expected_arguments = {
        "view": ["shape"],
        "permute": ["first", "second"],
        "rearrange": ["pattern"],
        "local": ["shape"],
    }
    assert seen == [*expected_arguments[method], "producer"]
    assert len(produced) == 1
    view, producer_name, producer_span = produced[0]
    assert producer_name != "renamed"
    assert view.name == producer_name
    assert len(observed) == 1
    observed_view, observed_span = observed[0]
    assert observed_view.same_as(view)
    assert observed_span.same_as(producer_span)
    assert observed_span.source_name.name == "producer.py"
    assert (observed_span.line, observed_span.column, observed_span.end_column) == (7, 2, 19)
    # A later ordinary use still receives that use's source location.
    assert not view.span.same_as(producer_span)
    assert isinstance(view.span, ir.SequentialSpan)
    assert any(
        span.line == 6 and span.source_name.name != "producer.py" for span in view.span.spans
    )
    assert any(span.same_as(producer_span) for span in view.span.spans)
    declaration = assert_alias(result, shape, physical_index=physical_index)
    assert declaration.buffer.same_as(view)
    assert statements(result)[-1].buffer.same_as(view)
    assert not any(isinstance(node, tirx.Bind) for node in statements(result))
