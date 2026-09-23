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

"""Reused JIT decorators retain annotation shape and compiled lexical values."""

from __future__ import annotations

import numpy as np
import pytest

import tvm
from tvm.script import tirx as T

VALUE = 1


def _delayed(n, decorator):
    @decorator
    def function(output: T.Buffer((n,), "int32")):
        n = 2
        output[0] = n + VALUE

    return function


@pytest.mark.parametrize("factory", [False, True])
def test_reused_jit_decorator_captures_application_scope(factory):
    decorator = T.jit() if factory else T.jit
    pending = _delayed(8, decorator)
    n = 99
    VALUE = 20
    function = pending.specialize()
    assert [int(value) for value in function.params[0].ty.shape] == [8]
    compiled = tvm.compile(function, target="llvm", tir_pipeline="tirx")
    output = tvm.runtime.tensor(np.zeros(8, dtype="int32"))
    compiled(output)
    np.testing.assert_array_equal(output.numpy(), [3, 0, 0, 0, 0, 0, 0, 0])
