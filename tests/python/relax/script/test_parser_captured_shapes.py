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
"""Captured Relax tensor shapes need concrete symbols."""

import pytest

from tvm import ir
from tvm.script import parser


def test_captured_shapes_require_explicit_symbols():
    n = ir.Var("n", "int64")
    source = """
@R.function
def main(x: R.Tensor(shape, "float32")):
    return x
"""
    result = parser.parse(source, extra_vars={"shape": (n, 16)})
    assert result.params[0].ty.shape[0].same_as(n)
    with pytest.raises(
        TypeError, match="^Builder expression arguments require concrete symbols, not strings$"
    ) as error:
        parser.parse(source, extra_vars={"shape": ("n", 16)})
    assert type(error.value) is TypeError
