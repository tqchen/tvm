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
"""Relax comparisons select tensor operators or the primitive fallback explicitly."""

import pytest

from tvm import ir, tirx
from tvm.script import parser

OPERATORS = [("<", "LT"), ("<=", "LE"), (">", "GT"), (">=", "GE"), ("==", "EQ"), ("!=", "NE")]


@pytest.mark.parametrize("operator,kind", OPERATORS)
def test_unmarked_host_comparison_never_calls_python_overload(operator, kind):
    calls = []

    class Host:
        pass

    def comparison(self, other):
        calls.append(other)
        return True

    setattr(Host, "__" + kind.lower() + "__", comparison)
    decorator = "R.function"
    statement = "return left " + operator + " right"
    with pytest.raises(TypeError, match="(PrimExpr|primitive|convert|type)"):
        parser.parse(
            f"@{decorator}\ndef main():\n    {statement}\n",
            extra_vars={"left": Host(), "right": Host()},
        )
    assert calls == []


@pytest.mark.parametrize("operator,kind", OPERATORS)
def test_relax_tensor_comparison_constructs_written_operator(operator, kind):
    from tvm import relax

    operation = {
        "LT": "less",
        "LE": "less_equal",
        "GT": "greater",
        "GE": "greater_equal",
        "EQ": "equal",
        "NE": "not_equal",
    }[kind]
    actual = parser.parse(
        '@R.function\ndef main(x: R.Tensor((2,), "float32"), y: R.Tensor((2,), "float32")):\n'
        f"    return x {operator} y\n",
        track_span=True,
    )
    call = actual.body.blocks[0].bindings[0].value
    expected = relax.BlockBuilder().normalize(getattr(relax.op, operation)(*actual.params))
    ir.assert_structural_equal(call, expected)
    assert call.span is not None


@pytest.mark.parametrize("operator,kind", OPERATORS)
def test_relax_primitive_comparison_is_concrete_ir(operator, kind):
    actual = parser.parse(
        f'@R.function\ndef main(x: R.Prim("int32")):\n    return 0 {operator} x\n'
    )
    ir.assert_structural_equal(actual.body.body, getattr(tirx, kind)(0, actual.params[0]))
