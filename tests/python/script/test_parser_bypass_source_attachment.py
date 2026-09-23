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
"""Source attachment preserves common IR identity through direct scope values."""

import pytest

import tvm
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I


@pytest.mark.parametrize("count", [1, 2])
def test_scope_value_source_attachment_preserves_identity(count):
    span = tvm.ir.Span(tvm.ir.SourceName("scope.py"), 3, 3, 4, 25)
    with IRBuilder():
        values = tuple(tvm.ir.Var("", "int32") for _ in range(count))
        result = values[0] if count == 1 else values
        assert I.at_(span, result) is result
        for value in values:
            assert value.span.same_as(span)
