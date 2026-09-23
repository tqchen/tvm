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
"""Construction regressions exercised by ordinary TVMScript entry points."""

import tvm
from tvm.script import ir as I
from tvm.script import parser
from tvm.script import relax as R


def test_eager_module_annotations_resolve_global_info_in_module_scope():
    @I.ir_module
    class Module:
        I.module_global_infos(
            {
                "mesh": [R.device_mesh((2,), I.Range(0, 2))],
                "vdevice": [I.vdevice("llvm")],
            }
        )

        @R.function
        def distributed(x: R.DTensor((4,), "float32", "mesh[0]", "R")):
            return x

        @R.function
        def tensor(x: R.Tensor((4,), "float32", vdevice="llvm:0:global")):
            return x

    assert (
        Module["distributed"].params[0].ty.device_mesh.__chandle__()
        == Module.global_infos["mesh"][0].__chandle__()
    )
    assert (
        Module["tensor"].params[0].ty.vdevice.__chandle__()
        == Module.global_infos["vdevice"][0].__chandle__()
    )


def test_scoped_vdevice_selects_target_ordinal_and_roundtrips():
    module = parser.parse("""
@I.ir_module
class Module:
    I.module_global_infos({"vdevice": [
        I.vdevice("llvm"), I.vdevice("cuda"), I.vdevice("metal"), I.vdevice("cuda", 1),
    ]})
    @R.function
    def main(x: R.Tensor((4,), "float32", vdevice="cuda:1:global")):
        return x
""")
    assert (
        module["main"].params[0].ty.vdevice.__chandle__()
        == module.global_infos["vdevice"][3].__chandle__()
    )
    tvm.ir.assert_structural_equal(module, tvm.script.from_source(module.script()))
