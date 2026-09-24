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
"""Enclosing lexical declarations retain captured values during construction."""

import pytest

from tvm.script.parser import entry


@pytest.mark.parametrize(
    "source,captured,expected_source",
    [
        (
            """
@X.script
def main(x: X.tensor((2,), dtype)):
    nonlocal dtype
    return x
""",
            {"dtype": "float32"},
            '@X.script\ndef main(x: X.tensor((2,), "float32")):\n    return x\n',
        ),
        (
            """
@X.script
def main():
    nonlocal value
    X.record(value + 1)
""",
            {"value": 3},
            "@X.script\ndef main():\n    X.record(4)\n",
        ),
    ],
)
def test_nonlocal_declaration_preserves_captured_values(
    language, source, captured, expected_source
):
    # Before: nonlocal appears in signature-only and body-arithmetic captures.
    # Expected builder: preserve annotation/value, return identity, and caller captures.
    captures = {"X": language.X, **captured}
    original_captures = dict(captures)
    expected = entry.parse(expected_source, extra_vars={"X": language.X}, root_builder=language.X)
    actual = entry.parse(source, extra_vars=captures, root_builder=language.X)
    assert actual.name == expected.name
    assert len(actual.params) == len(expected.params)
    if actual.params:
        assert actual.params[0].args[0].args == expected.params[0].args[0].args
        assert actual.body[0] == ("return", actual.params[0])
        assert expected.body[0] == ("return", expected.params[0])
    else:
        assert actual.body == expected.body == [("emit", 4)]
    assert captures == original_captures
