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
"""Quoted symbols share identity without adding Python bindings."""

import pytest

from tvm.script.parser import entry


def parse(language, source, *, extra_vars=None, **options):
    language.X.tuple = lambda *fields: fields
    return entry.parse(source, extra_vars={"X": language.X, **(extra_vars or {})}, **options)


@pytest.mark.parametrize("track_span", [True, False])
def test_signature_symbols_cross_nested_calls_parameters_return_and_body(language, track_span):
    # Before: quoted "n" in nested annotations, then n = X.symbol() in the body.
    # Expected builder program: annotations use X.resolve_type_var_("n");
    # n = X.resolve_type_var_("n", "int64") introduces the same object in the body.
    function = parse(
        language,
        """
@X.script
def main(
    x: X.tuple(X.tensor(("n",), "float32"), X.tensor(("n",), "float32")),
    y: X.tensor(("n",), "float32"),
) -> X.tensor(("n",), "float32"):
    n = X.symbol()
    X.record(n)
    return y
""",
        track_span=track_span,
    )
    n = function.params[0].args[0][0].args[0][0]
    assert function.params[0].args[0][1].args[0][0] is n
    assert function.params[1].args[0].args[0][0] is n
    assert function.ret_type.args[0][0] is n
    assert function.body[0][1] is n


@pytest.mark.parametrize(
    "annotation",
    [
        'X.tensor((n, "n"), "float32")',
        'X.tuple(X.tensor((n,), "float32"), X.tensor(("n",), "float32"))',
    ],
)
def test_signature_read_before_introduction_remains_unbound(language, annotation):
    source = f"""
@X.script
def main(x: {annotation}):
    return x
"""
    with pytest.raises(NameError) as error:
        parse(language, source)
    assert isinstance(error.value, NameError)
    assert "n" in str(error.value)


def test_signature_strings_do_not_replace_captured_python_names(language):
    function = parse(
        language,
        """
@X.script
def main(x: X.tensor((n, "n"), "float32"), y: X.tensor((n,), "float32")):
    X.record(n)
    return y
""",
        extra_vars={"n": 7},
    )
    first, n = function.params[0].args[0].args[0]
    assert first == 7
    assert function.params[1].args[0].args[0][0] == 7
    assert function.body[0][1] == 7
    assert n.name == "n"
