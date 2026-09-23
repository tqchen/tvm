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

"""Public Relax construction retains validation controls and Python attachments."""

import pytest

import tvm
from tvm import relax
from tvm.relax import BasePyModule
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T


def _invalid_inplace_module(**options):
    @I.ir_module(**options)
    class Module:
        @T.prim_func
        def kernel(A: T.Buffer((2,), "int32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2,), "int32")):
            R.func_attr({"relax.force_pure": True})
            result = R.call_tir_inplace(
                Module.kernel,
                (x,),
                [0, 0],
                [R.Tensor((2,), "int32"), R.Tensor((2,), "int32")],
            )
            return result

    return Module


def test_default_relax_validation_rejects_repeated_inplace_input():
    with pytest.raises(ValueError):
        _invalid_inplace_module()


def test_relax_validation_can_be_explicitly_disabled():
    module = _invalid_inplace_module(check_well_formed=False)
    assert isinstance(module["main"], relax.Function)
    call = module["main"].body.blocks[0].bindings[0].value
    assert list(call.attrs.inplace_indices) == [0, 0]
    assert not relax.analysis.check_well_formed(module)


def test_python_module_factory_and_attached_function_execute():
    @R.py_module
    class Module(BasePyModule):
        @I.pyfunc
        def twice(value):
            return value * 2

    assert callable(Module)
    assert Module.__pyfuncs__["twice"](3) == 6
    instance = Module(tvm.cpu())
    assert isinstance(instance, BasePyModule)
    assert instance.twice(4) == 8
    assert instance.pyfuncs["twice"](5) == 10
