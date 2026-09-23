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
"""Native Relax obligations of the builder parser contract."""

import pytest

from tvm import ir, relax
from tvm.relax.script.builder import parser_protocol as RP
from tvm.script import ir as I
from tvm.script import relax as R


def test_native_relax_validation_rejects_free_function_result():
    value = relax.Var("undefined", relax.TensorType((1,), "float32"))
    function = relax.Function([], value, value.ty)
    with pytest.raises(ValueError, match="Program is not well-formed"):
        RP.check_well_formed_(function)


def test_relax_return_does_not_relocate_a_direct_metadata_result():
    span = ir.Span(ir.SourceName("producer.py"), 8, 8, 3, 9)
    constant = relax.const([7], "int32")
    existing = ir.GenericConst(constant.value, constant.ty, span)

    @R.function
    def function():
        return I.meta_var(existing)

    assert existing.span.same_as(span)
    assert isinstance(function, relax.Function)
