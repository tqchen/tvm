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
"""Scheduled block axes reject duplicate source bindings in their native frame."""

import pytest

from tvm.script import parser


@pytest.mark.parametrize(
    "axes",
    [
        "vi = T.axis.spatial(16, i)\n            vi = T.axis.spatial(16, j)",
        'vi, vi = T.axis.remap("SS", [i, j])',
        'vi, vj = T.axis.remap("SS", [i, j])\n            vi = T.axis.spatial(16, j)',
    ],
)
def test_duplicate_block_axis_source_name_is_rejected(axes):
    source = f"""
@T.prim_func(s_tir=True)
def main():
    for i, j in T.grid(16, 16):
        with T.sblock("block"):
            {axes}
            T.evaluate(vi)
"""
    with pytest.raises(ValueError):
        parser.parse(source)
