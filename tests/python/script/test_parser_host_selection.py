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
"""Compile-time choices preserve source order, scope and missing-value errors."""

import pytest

from tvm.script.parser import entry


def parse(language, source, *, extra_vars=None, **options):
    return entry.parse(source, extra_vars={"X": language.X, **(extra_vars or {})}, **options)


def test_marked_expressions_are_lazy_operand_valued_and_ordered(language):
    seen = []

    def operand(value):
        seen.append(value)
        return value

    result = parse(
        language,
        """
@X.script
def main():
    X.record(operand(1) if I.constexpr(operand(True)) else invalid())
    X.record(I.constexpr(operand(0)) and invalid())
    X.record(I.constexpr(operand(4)) or invalid())
    X.record(I.constexpr(operand(2)) and operand(7))
    X.record(I.constexpr(operand(0)) or operand(8))
""",
        extra_vars={"operand": operand},
    )
    assert seen == [True, 1, 0, 4, 2, 7, 0, 8]
    assert [value for _, value in result.body] == [1, 0, 4, 7, 8]


def test_unmarked_expression_eagerly_constructs_both_ir_arms(language):
    seen = []

    def operand(value):
        seen.append(value)
        return value

    result = parse(
        language,
        """
@X.script
def main(condition: X.value):
    X.record(operand(1) if condition else operand(2))
    X.record(condition and operand(True))
    X.record(condition or operand(False))
""",
        extra_vars={"operand": operand},
    )
    assert seen == [1, 2, True, False]
    assert result.body[0][1].op == "select"
    assert result.body[1][1].op == "and"
    assert result.body[2][1].op == "or"


def test_nested_unmarked_statement_retains_ir_frame(language):
    result = parse(
        language,
        """
@X.script
def main(condition: X.value):
    if I.constexpr(True):
        if condition:
            X.record(1)
        else:
            X.record(2)
""",
    )
    assert result.body == [("emit", 1), ("emit", 2)]
    assert [event[1] for event in language.events if event[0] == "enter"] == [
        "function",
        "if",
        "then",
        "else",
    ]


@pytest.mark.parametrize("condition", [True, False])
def test_conditional_binding_can_be_assigned_after_skipped_branch(language, condition):
    result = parse(
        language,
        f"""
@X.script
def main():
    if I.constexpr({condition}):
        x = 1
    x = 2
    X.record(x)
""",
    )
    assert result.body[-1] == ("emit", 2)


@pytest.mark.parametrize("track_span", [True, False])
def test_constexpr_keeps_named_expression_unsupported(language, track_span):
    with pytest.raises(SyntaxError, match="Unsupported expression: NamedExpr"):
        parse(
            language,
            """
@X.script
def main():
    if I.constexpr(bool(x := 3)):
        X.record(x)
""",
            track_span=track_span,
        )


def test_ir_optional_binding_survives_skipped_host_assignment(language):
    result = parse(
        language,
        """
@X.script
def main(condition: X.value):
    if condition:
        x = X.value(1)
    if I.constexpr(False):
        x = X.value(2)
    x = X.value(3)
    X.record(x)
""",
    )
    assert result.body[-1][1].args == (3,)


@pytest.mark.parametrize(
    "expression", ["1 if I.constexpr(x) else 0", "I.constexpr(x) and 1", "I.constexpr(x) or 1"]
)
def test_missing_host_binding_cannot_be_truth_tested(language, expression):
    with pytest.raises(NameError) as caught:
        parse(
            language,
            f"""
@X.script
def main():
    if I.constexpr(False):
        x = 1
    X.record({expression})
""",
        )
    assert isinstance(caught.value, NameError)


def test_missing_host_if_binding_raises_before_branch_assignments(language):
    with pytest.raises(NameError) as caught:
        parse(
            language,
            """
@X.script
def main():
    if I.constexpr(False):
        x = 1
    if I.constexpr(x):
        x = 2
    else:
        x = 3
""",
        )
    assert isinstance(caught.value, NameError)


def test_host_lambda_local_does_not_capture_optional_binding(language):
    result = parse(
        language,
        """
@X.script
def main():
    if I.constexpr(False):
        x = 1
    if I.constexpr((lambda x: x)(True)):
        X.record(222)
""",
    )
    assert result.body[-1] == ("emit", 222)


def test_parser_skips_invalid_ramp_and_preserves_support_name(language):
    # Before: X.ramp(0, 1, _PS) if I.constexpr(_PS > 1) else 0
    # Expected builder program: X.ramp(0, 1, _PS) if _PS > 1 else 0
    # The captured _PS name remains an ordinary user binding.
    source = """
@X.script
def main():
    X.record(X.ramp(0, 1, _PS) if I.constexpr(_PS > 1) else 0)
"""
    function = parse(language, source, extra_vars={"_PS": 1})
    assert function.body == [("emit", 0)]


def test_source_logical_operands_skip_at_construction(language):
    # Before: value if I.constexpr(False and fail()) else fallback
    # Expected builder program: value if False and fail() else fallback
    # Calls inside the constexpr operand retain Python short-circuit behavior.
    def fail():
        raise AssertionError("skipped source operand was evaluated")

    function = parse(
        language,
        """
@X.script
def main():
    X.record(1 if I.constexpr(False and fail()) else 2)
    X.record(3 if I.constexpr(True or fail()) else 4)
""",
        extra_vars={"fail": fail},
    )
    assert [value for _, value in function.body] == [2, 3]


@pytest.mark.parametrize("operator", ["and", "or"])
def test_unmarked_logical_chain_preserves_left_association(language, operator):
    result = language.parse(f"""
@X.script
def main(a: X.value(), b: X.value(), c: X.value()):
    X.record(a {operator} b {operator} c)
""")
    expression = result.body[0][1]
    assert expression.op == operator
    assert expression.args[1] is result.params[2]
    assert expression.args[0].op == operator
    assert expression.args[0].args == tuple(result.params[:2])


def test_ir_branch_incoming_read_and_rebinding_obeys_python_scope(language):
    with pytest.raises(NameError) as error:
        language.parse("""
@X.script
def main(condition: X.value(), x: X.value()):
    y = x
    if condition:
        y = y + x
    else:
        y = x
    return y
""")
    assert isinstance(error.value, NameError)
    assert "y" in str(error.value)
