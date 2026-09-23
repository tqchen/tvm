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
"""Native declaration frames retain identity across generated body execution."""

import pytest

from tvm import ir, tirx
from tvm.relax.script import builder as R
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.tirx.script import builder as T
from tvm.tirx.script.builder.frame import ForFrame


@pytest.mark.parametrize("dialect", [T, R])
def test_function_declaration_reentry_reuses_parameters_symbols_and_reference(dialect):
    # Before: @X.function def identity(x: ...): return x
    # Expected builder program:
    # with X.function(decl=True) as fn: x = X.arg("x", annotation)
    # with fn: X.return_(fn.params[0])
    with IRBuilder() as builder, I.ir_module():
        with dialect.function(decl=True) as frame:
            dialect.func_name("identity")
            annotation = T.int32() if dialect is T else R.Tensor((4,), "float32")
            parameter = dialect.arg("x", annotation)
            symbol = frame.resolve_type_var("n")
        reference = frame.reference
        assert frame.params[0].same_as(parameter)
        with frame:
            assert frame.resolve_type_var("n").same_as(symbol)
            assert frame.params[0].same_as(parameter)
            if dialect is T:
                T.evaluate(parameter)
            else:
                R.func_ret_value(parameter)
        assert frame.reference.same_as(reference)
        assert frame.function.params[0].same_as(parameter)
    module = builder.get()
    assert module.get_global_var("identity").same_as(reference)
    assert module["identity"].params[0].same_as(parameter)


@pytest.mark.parametrize(
    "names, expected",
    [
        ("ids", ["ids_0", "ids_1", "ids_2"]),
        (("i", "*tail"), ["i", "tail_0", "tail_1"]),
        (("*head", "last"), ["head_0", "head_1", "last"]),
    ],
)
def test_native_grid_names_and_unpacking_preserve_entered_variables(names, expected):
    # Before: for i, *tail in X.grid(2, 3, 4): X.evaluate(i)
    # Expected builder program:
    # with X.for_(X.grid(2, 3, 4), names=("i", "*tail")) as (i, *tail): ...
    with IRBuilder() as builder:
        frame = T.grid(2, 3, 4)
        assert isinstance(frame, ForFrame)
        assert T.for_(frame, names=names) is frame
        with frame as variables:
            assert [variable.name for variable in variables] == expected
            for variable in variables:
                T.evaluate(variable)
    loop = builder.get()
    actual = []
    while isinstance(loop, tirx.For):
        actual.append(loop.loop_var)
        loop = loop.body
    assert len(actual) == 3
    assert all(a.same_as(b) for a, b in zip(actual, variables))
    assert all(
        statement.value.same_as(variable) for statement, variable in zip(loop.seq, variables)
    )


def test_singleton_tuple_loop_target_keeps_sequence_shape():
    # Before: for (i,) in X.grid(4): X.evaluate(i)
    # Expected builder program: with X.for_(X.grid(4), names=("i",)) as (i,): ...
    with IRBuilder() as builder:
        frame = T.for_(T.grid(4), names=("i",))
        with frame as variables:
            assert len(variables) == 1
            T.evaluate(variables[0])
    assert builder.get().loop_var.same_as(variables[0])


def test_conditional_output_uses_same_name_and_native_frame_result():
    # Before: if condition: y = X.add(x, x); else: y = X.multiply(x, x); return y
    # Expected builder program:
    # with X.If(condition) as frame:
    #     with X.Then():
    #         def then(): y = X.bind_(X.add(x, x), name="y")
    #         then()
    #     with X.Else():
    #         def otherwise(): y = X.bind_(X.multiply(x, x), name="y")
    #         otherwise()
    # y = frame.var; X.return_(y)
    actual = parser.parse("""
@R.function
def main(condition: R.Tensor((), "bool"), x: R.Tensor((4,), "float32")):
    if condition:
        y = R.add(x, x)
    else:
        y = R.multiply(x, x)
    return y
""")
    with IRBuilder() as builder, R.function():
        R.func_name("main")
        condition = R.arg("condition", R.Tensor((), "bool"))
        x = R.arg("x", R.Tensor((4,), "float32"))
        with R.If(condition) as frame:
            with R.Then():
                R.bind_(R.add(x, x), name="y")
            with R.Else():
                R.bind_(R.multiply(x, x), name="y")
        R.func_ret_value(frame.var)
    ir.assert_structural_equal(actual, builder.get())


def test_ir_branch_incoming_read_and_rebinding_obeys_python_scope():
    # Before: y = x; if condition: y = X.add(y, x); else: y = x
    # Expected builder program: a local branch helper reads y before assignment;
    # report Python's unbound-name error, without synthetic y=y capture parameters.
    with pytest.raises(Exception) as error:
        parser.parse("""
@R.function
def main(condition: R.Tensor((), "bool"), x: R.Tensor((4,), "float32")):
    y = x
    if condition:
        y = R.add(y, x)
    else:
        y = x
    return y
""")
    assert isinstance(error.value.__cause__, NameError)
    assert "y" in str(error.value.__cause__)


def test_body_attribute_overrides_default_after_declaration():
    # Before: @X.prim_func def main(): X.func_attr({"global_symbol": "custom"})
    # Expected builder program:
    # with X.function(decl=True) as fn: X.func_name("main")
    # with fn: X.func_attr({"global_symbol": "custom"}); X.emit_(X.evaluate(0))
    with IRBuilder() as builder, I.ir_module():
        with T.function(decl=True) as frame:
            T.func_name("main")
        with frame:
            T.func_attr({"global_symbol": "custom"})
            T.evaluate(0)
    assert builder.get()["main"].attrs["global_symbol"] == "custom"
