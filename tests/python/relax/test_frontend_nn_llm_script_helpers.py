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
"""LLM allocation and prefill helpers construct their native scheduled IR."""

import pytest

import tvm
from tvm.relax.frontend.nn.llm import _kernel_common as kernels
from tvm.script import parser


@pytest.mark.parametrize(
    "helper,args,specs",
    [
        (kernels._var, '("float32")', [((1,), "float32", "local")]),
        (kernels._var_cpu, '("float32")', [((1,), "float32", "global")]),
        (
            kernels._alloc_softmax_state_buffers,
            "(8, 16, 2, 2)",
            [((8, 16), "float32", "shared"), ((8, 16), "float32", "local")]
            + [((8,), "float32", "shared")] * 3
            + [((2,), "float32", "local")] * 3,
        ),
        (
            kernels._alloc_mha_qkvo_buffers,
            '(8, 16, 32, 64, "float16")',
            [
                ((8, 32), "float16", "shared"),
                ((16, 32), "float16", "shared"),
                ((16, 64), "float16", "shared"),
                ((8, 64), "float32", "local"),
            ],
        ),
        (
            kernels._alloc_mla_qkvo_buffers,
            '(8, 16, 32, 64, "float16")',
            [
                ((8, 32), "float16", "shared"),
                ((16, 32), "float16", "shared"),
                ((8, 64), "float32", "local"),
            ],
        ),
        (kernels._alloc_tile_walk_state, "()", [((1,), "int32", "local")] * 6),
    ],
)
def test_llm_buffer_allocations(helper, args, specs):
    names = [f"buffer_{index}" for index in range(len(specs))]
    uses = "".join(
        f"    T.evaluate({name}[{', '.join('0' for _ in shape)}])\n"
        for name, (shape, _, _) in zip(names, specs)
    )
    prefix = "@T.prim_func(s_tir=True)\ndef main():\n"
    source = prefix + f"    {', '.join(names)} = helper{args}\n" + uses
    expected_source = (
        prefix
        + "".join(
            f'    {name} = T.sblock_alloc_buffer({shape}, "{dtype}", scope="{scope}")\n'
            for name, (shape, dtype, scope) in zip(names, specs)
        )
        + uses
    )
    actual = parser.parse(source, {"helper": helper})
    tvm.ir.assert_structural_equal(actual, parser.parse(expected_source))


def test_llm_prefill_macro_uses_callers_builder():
    init_states = kernels._make_prefill_macros(4, 8, 16, 8, 2, 2, 1)[0]
    prefix = """
@T.prim_func(s_tir=True)
def main(m: T.Buffer((4,), "float32"), d: T.Buffer((4,), "float32"),
         out: T.Buffer((4, 8), "float32"), ty: T.int32, tx: T.int32):
"""
    source = prefix + "    init_states(m, d, out, ty, tx)\n"
    expected_source = (
        prefix
        + """
    for i in T.serial(T.ceildiv(4, 4)):
        row: T.let[T.int32] = i * 2 * 2 + ty * 2 + tx
        if row < 4:
            m[row] = -5e4
            d[row] = 1.0
    for li, lj in T.grid(4, 8):
        with T.sblock("O_init"):
            i, j = T.axis.remap("SS", [li, lj])
            out[i, j] = 0.0
    T.tvm_storage_sync("shared")
"""
    )
    actual = parser.parse(source, {"init_states": init_states})
    tvm.ir.assert_structural_equal(actual, parser.parse(expected_source))
