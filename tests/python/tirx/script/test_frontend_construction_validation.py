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

"""Public TIRx construction validates native scopes and executable specialization."""

import numpy as np
import pytest

import tvm
from tvm import tirx
from tvm.error import DiagnosticError
from tvm.script import ir as I
from tvm.script import tirx as T

VALUE = 1


def _build_lexical_global():
    @T.prim_func
    def write(A: T.Buffer((1,), "int32")):
        A[0] = VALUE

    return write


def test_default_tirx_validation_rejects_non_divisible_scope_extents():
    with pytest.raises(DiagnosticError):

        @T.prim_func
        def invalid():
            T.device_entry()
            block = T.cta_id([1])
            warp = T.warp_id([3])
            thread = T.thread_id([100])
            T.evaluate(block + warp + thread)


def test_tirx_validation_can_be_explicitly_disabled():
    @T.prim_func(check_well_formed=False)
    def invalid():
        T.device_entry()
        block = T.cta_id([1])
        warp = T.warp_id([3])
        thread = T.thread_id([100])
        T.evaluate(block + warp + thread)

    assert isinstance(invalid, tirx.PrimFunc)
    definitions = invalid.body.body.seq[:3]
    assert all(isinstance(statement, tirx.ScopeIdDefStmt) for statement in definitions)
    with pytest.raises(tvm.error.InternalError, match="100 is not divisible by 3"):
        tirx.analysis.verify_tirx_well_formed(invalid)


def test_optional_annotation_is_restricted_to_jit():
    with pytest.raises(DiagnosticError, match="only supported by @T.jit"):

        @T.prim_func(private=True)
        def invalid(value: T.Optional(T.handle)):
            T.evaluate(0)

    @T.jit(private=True)
    def valid(value: T.Optional(T.handle)):
        if I.constexpr(value is None):
            T.evaluate(1)
        else:
            T.evaluate(2)

    present, absent = valid.specialize(), valid.specialize(value=None)
    assert len(present.params) == 1
    assert len(absent.params) == 0
    assert present.body.value.value == 2
    assert absent.body.value.value == 1


def test_lexical_global_is_not_replaced_by_dynamic_caller_local():
    VALUE = 2
    function = _build_lexical_global()
    assert function.body.value.value == 1
    compiled = tvm.compile(function, target="llvm", tir_pipeline="tirx")
    output = tvm.runtime.tensor(np.zeros(1, dtype="int32"))
    compiled(output)
    np.testing.assert_array_equal(output.numpy(), np.array([1], dtype="int32"))
