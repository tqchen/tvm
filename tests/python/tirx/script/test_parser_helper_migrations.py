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
"""External helpers construct their IR directly in the new builder."""

import pytest

import tvm
from tvm.backend.cuda.iket import IketProfiler
from tvm.backend.cuda.lang.warp_role import WarpgroupRole, WarpRole
from tvm.relax.frontend.nn.llm import _kernel_common as kernels
from tvm.script import parser


def test_iket_annotations():
    source = """
@T.prim_func
def main(payload: T.int32):
    iket = IketProfiler()
    iket.range_push("outer")
    iket.range_push("inner", payload)
    iket.mark("point")
    iket.mark("point_payload", payload)
    token = iket.range_start("range")
    token_with_payload = iket.range_start("range_payload", payload)
    sentinel = iket.sentinel_token("sentinel")
    iket.range_end(token)
    iket.range_end(token_with_payload, payload)
    iket.range_end(sentinel)
    iket.range_pop()
    iket.range_pop()
"""
    expected_source = """
@T.prim_func
def main(payload: T.int32):
    T.evaluate(T.cuda.iket.range_push("outer"))
    T.evaluate(T.cuda.iket.range_push("inner", payload))
    T.evaluate(T.cuda.iket.mark("point"))
    T.evaluate(T.cuda.iket.mark("point_payload", payload))
    token = T.cuda.iket.range_start("range")
    token_with_payload = T.cuda.iket.range_start("range_payload", payload)
    sentinel = T.cuda.iket.sentinel_token("sentinel")
    T.evaluate(T.cuda.iket.range_end(token))
    T.evaluate(T.cuda.iket.range_end(token_with_payload, payload))
    T.evaluate(T.cuda.iket.range_end(sentinel))
    T.evaluate(T.cuda.iket.range_pop())
    T.evaluate(T.cuda.iket.range_pop())
"""
    actual = parser.parse(source, {"IketProfiler": IketProfiler})
    tvm.ir.assert_structural_equal(actual, parser.parse(expected_source))


@pytest.mark.parametrize("role", [WarpRole, WarpgroupRole])
@pytest.mark.parametrize("regs,increase", [(None, False), (48, False), (232, True)])
def test_warp_role_guards_and_register_budgets(role, regs, increase):
    source = f"""
@T.prim_func
def main(role_id: T.int32):
    with role(role_id, 1, regs={regs}, increase={increase}):
        T.evaluate(T.cuda.warp_sync())
"""
    budget = ""
    if regs is not None:
        direction = "inc" if increase else "dec"
        budget = f'        T.evaluate(T.ptx["setmaxnreg.{direction}.sync.aligned.u32"]({regs}))\n'
    expected_source = f"""
@T.prim_func
def main(role_id: T.int32):
    if role_id == 1:
{budget}        T.evaluate(T.cuda.warp_sync())
"""
    actual = parser.parse(source, {"role": role})
    tvm.ir.assert_structural_equal(actual, parser.parse(expected_source))


def test_warpgroup_range_guard():
    source = """
@T.prim_func
def main(role_id: T.int32):
    with WarpgroupRole(role_id, (1, 3)):
        T.evaluate(T.cuda.warp_sync())
"""
    expected_source = """
@T.prim_func
def main(role_id: T.int32):
    if 1 <= role_id and role_id < 3:
        T.evaluate(T.cuda.warp_sync())
"""
    actual = parser.parse(source, {"WarpgroupRole": WarpgroupRole})
    tvm.ir.assert_structural_equal(actual, parser.parse(expected_source))


@pytest.mark.parametrize("sliding", [False, True])
def test_llm_length_info_buffer(sliding):
    source = f"""
@T.prim_func
def main(lengths: T.handle):
    length_info = declare(lengths, 4, {sliding}, 8)
    T.evaluate(length_info[{"0, 0" if sliding else "0"}])
"""
    expected_source = source.replace(
        f"declare(lengths, 4, {sliding}, 8)",
        f'T.match_buffer(lengths, {"(3, 4)" if sliding else "(4,)"}, "int32", elem_offset=8)',
    )
    actual = parser.parse(source, {"declare": kernels._declare_length_info})
    tvm.ir.assert_structural_equal(actual, parser.parse(expected_source))
