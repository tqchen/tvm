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
"""Relax adaptation after one shared module parse."""

import gc
import weakref

import pytest

import tvm
from tvm import relax
from tvm.relax import BasePyModule
from tvm.script import ir as I
from tvm.script import relax as R


def test_py_module_preserves_original_callable_and_closure():
    calls = []
    factor = 3

    class Original:
        @I.pyfunc
        def multiply(value):
            calls.append(value)
            return value * factor

    original = Original.multiply
    module = R.py_module(Original)
    assert calls == []
    assert module.__pyfuncs__["multiply"] is original
    assert isinstance(module["multiply"], relax.ExternFunc)
    assert module["multiply"].attrs["python_function_name"] == "multiply"
    assert module.__pyfuncs__["multiply"](4) == 12
    assert calls == [4]
    assert module["multiply"].span.source_name.name == __file__


def test_shared_module_stays_generic_and_relax_entry_adds_metadata():
    class Original:
        @I.pyfunc
        def value():
            return 9

    shared = I.ir_module(Original)
    assert len(shared.functions) == 0
    assert shared.__pyfuncs__["value"] is Original.value
    adapted = R.py_module(Original, track_span=False)
    assert adapted.__pyfuncs__["value"] is Original.value
    assert isinstance(adapted["value"], relax.ExternFunc)
    assert adapted["value"].span is None


def test_py_module_delegates_once_with_original_annotation_context(monkeypatch):
    from tvm.script.parser import entry

    original_parse = entry.parse
    seen = []

    def counted(source, **options):
        seen.append(options["definition_scope"])
        return original_parse(source, **options)

    monkeypatch.setattr(entry, "parse", counted)
    shape = (3,)

    @R.py_module
    class Module:
        @R.function
        def identity(x: R.Tensor(shape, "float32")):
            return x

    assert len(seen) == 1
    assert seen[0]["shape"] is shape
    assert list(Module["identity"].params[0].ty.shape) == [3]
    assert Module.__pyfuncs__ == {}


def test_factory_creates_independent_runtime_registries():
    @R.py_module
    class Module(BasePyModule):
        @I.pyfunc
        def twice(value):
            return value * 2

    assert Module.__pyfuncs__["twice"](3) == 6
    first, second = Module(tvm.cpu()), Module(tvm.cpu())
    assert isinstance(first, BasePyModule)
    assert first.twice(4) == 8 and second.twice(5) == 10
    assert first.pyfuncs is not second.pyfuncs
    first.add_python_function("local", lambda: 11)
    assert "local" not in second.pyfuncs
    assert "local" not in Module.__pyfuncs__


def test_decorator_does_not_retain_unrelated_definition_scope():
    class Sentinel:
        pass

    def create():
        unrelated = Sentinel()
        ref = weakref.ref(unrelated)

        @R.py_module
        class Module:
            @I.pyfunc
            def value():
                return 3

        return Module, ref

    enabled = gc.isenabled()
    gc.disable()
    try:
        module, reference = create()
        assert reference() is None
        assert module.__pyfuncs__["value"]() == 3
    finally:
        if enabled:
            gc.enable()


def test_py_module_rejects_non_class():
    with pytest.raises(TypeError, match="Expect a class"):
        R.py_module(lambda: None)
