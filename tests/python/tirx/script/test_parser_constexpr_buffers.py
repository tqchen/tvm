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
"""Compile-time selection preserves native parameter buffer identity."""

import pytest

from tvm import tirx
from tvm.script import parser


@pytest.mark.parametrize("marker", ["I.constexpr", "T.constexpr"])
@pytest.mark.parametrize("track_span", [True, False])
def test_marked_if_shares_parent_scope(marker, track_span):
    seen = []

    def choose():
        seen.append("condition")
        return True

    result = parser.parse(
        f"""
@T.prim_func
def main(a: T.handle):
    if {marker}(choose()):
        A = T.match_buffer(a, (4,), "int32")
    else:
        invalid()
    A[0] = 3
""",
        extra_vars={"choose": choose},
        track_span=track_span,
    )
    assert seen == ["condition"]
    assert len(result.params) == 1
    assert isinstance(result.body, tirx.BufferStore)
    assert result.body.buffer.same_as(result.params[0])
