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
import subprocess
import sys
import textwrap
import typing

import pytest

from tvm import ir
from tvm.relax.script.builder import parser_protocol as RP
from tvm.script.ir_builder import ir as IB
from tvm.script.ir_builder.ir import parser_protocol as P
from tvm.script.parser import protocol_registry as registry
from tvm.tirx.script.builder import parser_protocol as TP


@pytest.mark.parametrize("protocol", [P, TP, RP, registry])
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
        @registry.direct_call
        def make(self, value):
            calls.append(value)
            return value

    factory = Factory()
    function = Factory.make
    assert registry.direct_call(function) is function
    assert registry.is_direct_call(factory.make)
    assert calls == []
    token = object()
    assert factory.make(token) is token
    assert calls == [token]
    assert not registry.is_direct_call(object())

    def scope_value():
        return token

    assert registry.register_scope_var_query_or_decl(scope_value) is scope_value
    assert registry.is_scope_var_query_or_decl(scope_value)
    assert not registry.is_mutable_var_decl(scope_value, syntax="call")
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
    assert registry.is_direct_call(IB.meta_var)


def test_registration_has_one_parser_owner_and_no_builder_reexports():
    for name in (
        "ExprStrPolicy",
        "ArgsPolicy",
        "DeclarationArguments",
        "FunctionDecoratorInfo",
        "args_policy",
        "get_args_policy",
        "direct_call",
        "is_direct_call",
        "register_type_var_decl",
        "get_type_var_decl",
        "register_binding_decl",
        "is_binding_decl",
        "register_mutable_var_decl",
        "is_mutable_var_decl",
        "register_scope_var_query_or_decl",
        "is_scope_var_query_or_decl",
        "register_function",
        "function_info",
        "copy_function_info",
        "register_result_members",
        "get_result_members",
    ):
        assert getattr(registry, name).__module__ == registry.__name__, name
        assert name not in vars(P), name
        assert name not in vars(IB), name
    # constexpr is a real source marker, shared by builder namespaces by identity.
    assert IB.constexpr is registry.constexpr

    def source():
        raise AssertionError("metadata lookup must not invoke its callable")

    def target():
        raise AssertionError("metadata transfer must not invoke its callable")

    assert registry.get_type_var_decl(source) is None
    assert not registry.is_binding_decl(source)
    assert registry.function_info(target) is None
    with pytest.raises(AttributeError, match="__tvm_function_info__"):
        registry.copy_function_info(source, target)
    assert registry.function_info(target) is None

    dtype = object()
    assert registry.register_type_var_decl(source, dtype=dtype) is source
    declaration = registry.get_type_var_decl(source)
    assert declaration.dtype is dtype
    assert registry.get_type_var_decl(source) is declaration
    assert registry.register_binding_decl(source) is source
    assert registry.is_binding_decl(source)
    builder = object()
    assert registry.register_function(source, builder) is source
    metadata = registry.function_info(source)
    assert metadata.builder is builder
    assert registry.copy_function_info(source, target) is None
    assert registry.function_info(target) is metadata
    assert registry.function_info(source) is metadata


def test_registry_import_and_metadata_registration_need_no_tvm_initialization():
    # Load the actual module in a fresh interpreter with every TVM import blocked.
    script = textwrap.dedent(
        """
        import importlib.abc
        import importlib.util
        import sys

        class BlockTVM(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "tvm" or fullname.startswith("tvm."):
                    raise AssertionError("Registry initialized TVM: " + fullname)
                return None

        sys.meta_path.insert(0, BlockTVM())
        spec = importlib.util.spec_from_file_location("isolated_registry", sys.argv[1])
        registry = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = registry
        spec.loader.exec_module(registry)

        def constructor(device):
            return device

        assert registry.args_policy({"device": "global_info"})(constructor) is constructor
        policy = registry.get_args_policy(constructor)
        assert dict(policy.fields) == {"device": "global_info"}
        assert registry.direct_call(constructor) is constructor
        assert registry.is_direct_call(constructor)
        assert registry.register_function(constructor, object()) is constructor
        assert registry.function_info(constructor) is not None
        assert not any(name == "tvm" or name.startswith("tvm.") for name in sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, registry.__file__],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_registration_persists_across_successful_and_failed_parse_contexts(language):
    calls = []
    first, invalid, second = object(), object(), object()
    failure = RuntimeError("constructor failed")

    @registry.args_policy({"device": "global_info"})
    @registry.direct_call
    def constructor(device):
        calls.append(device)
        if device is invalid:
            raise failure
        return device

    policy = registry.get_args_policy(constructor)
    function_info = registry.function_info(language.X.script)
    source = '@X.script\ndef main():\n    X.record(constructor(device="mesh"))\n'
    language.global_infos["mesh"] = first
    assert language.parse(source, constructor=constructor).body == [("emit", first)]
    language.global_infos["mesh"] = invalid
    with pytest.raises(RuntimeError, match="constructor failed") as caught:
        language.parse(source, constructor=constructor)
    assert caught.value is failure
    assert registry.get_args_policy(constructor) is policy
    assert registry.function_info(language.X.script) is function_info
    assert registry.is_direct_call(constructor)
    language.global_infos["mesh"] = second
    assert language.parse(source, constructor=constructor).body == [("emit", second)]
    assert calls == [first, invalid, second]
    assert registry.get_args_policy(constructor) is policy
    assert registry.function_info(language.X.script) is function_info
    assert registry.is_direct_call(constructor)
