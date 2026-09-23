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
"""Distributed annotations retain common Range construction."""

import tvm
from tvm.relax.script import builder as R
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I


def test_distributed_module_range():
    source = """
@I.ir_module
class Module:
    I.module_global_infos({"mesh": [R.device_mesh((2,), I.Range(0, 2))]})

    @R.function
    def main(
        x: R.DTensor((16,), "float32", "mesh[0]", "R"),
    ) -> R.DTensor((16,), "float32", "mesh[0]", "R"):
        return x
"""
    with IRBuilder() as builder:
        with I.ir_module():
            mesh = R.device_mesh((2,), tvm.ir.Range(0, 2))
            I.module_global_infos({"mesh": [mesh]})
            with R.function():
                R.func_name("main")
                annotation = R.DTensor((16,), "float32", mesh, "R")
                x = R.arg("x", annotation)
                R.func_ret_type(annotation)
                R.func_ret_value(x)
    expected = builder.get()
    actual = parser.parse(source)
    tvm.ir.assert_structural_equal(expected, actual)
