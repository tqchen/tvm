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
"""Common IR and syntax metadata obligations of the builder parser contract."""

import inspect
import re
import typing

import pytest

from tvm import ir
from tvm.relax.script.builder import parser_protocol as RP
from tvm.script.ir_builder import ir as IB
from tvm.script.ir_builder.ir import parser_protocol as P
from tvm.tirx.script.builder import parser_protocol as TP


@pytest.mark.parametrize("protocol", [P, TP, RP])
def test_protocol_signatures_and_local_examples_are_complete(protocol):
    for name, function in vars(protocol).items():
        if name.startswith("_") or not inspect.isfunction(function):
            continue
        if function.__module__ != protocol.__name__:
            continue
        signature = inspect.signature(function)
        assert signature.return_annotation is not inspect.Signature.empty, name
        assert all(
            p.annotation is not inspect.Parameter.empty for p in signature.parameters.values()
        )
        typing.get_type_hints(function)
        doc = inspect.getdoc(function)
        assert ".. code:: python" in doc, name
        for parameter in signature.parameters:
            assert re.search(rf"^{parameter} :", doc, re.MULTILINE), (name, parameter)
        assert "Returns\n-------" in doc and "Notes\n-----" in doc, name


def test_registration_keeps_callable_identity_and_bound_method_metadata():
    calls = []

    class Factory:
        @P.direct_call
        def make(self, value):
            calls.append(value)
            return value

    factory = Factory()
    function = Factory.make
    assert P.direct_call(function) is function
    assert P.is_direct_call(factory.make)
    assert calls == []
    token = object()
    assert factory.make(token) is token
    assert calls == [token]
    assert not P.is_direct_call(object())

    def scope_value():
        return token

    assert P.register_scope_var_query_or_decl(scope_value) is scope_value
    assert P.is_scope_var_query_or_decl(scope_value)
    assert not P.is_mutable_var_decl(scope_value, syntax="call")
    assert scope_value() is token


def test_meta_var_retains_exact_payload_and_existing_span():
    span = ir.Span(ir.SourceName("producer.py"), 3, 3, 2, 8)
    value = ir.Var("producer_name", "int32", span)
    assert IB.meta_var(value) is value
    assert value.span.same_as(span)
    pair = (value, object())
    assert IB.meta_var(pair) is pair
    left, right = IB.meta_var(pair)
    assert left is value and right is pair[1]
    assert P.is_direct_call(IB.meta_var)
