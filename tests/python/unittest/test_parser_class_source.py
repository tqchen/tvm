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
"""Class source recovery uses the exact decorator location in gallery runners."""

import importlib.util
import linecache
import sys

import pytest

from tvm.script import ir as I
from tvm.script import tirx as T


def _execute(source, filename, monkeypatch):
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("__main__", None))
    module.__dict__.update(I=I, T=T)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "__main__", module)
        exec(compile(source, filename, "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("cached", [False, True])
def test_gallery_classes_retain_distinct_source_locations(tmp_path, monkeypatch, cached):
    source = """
@I.ir_module
class Repeated:
    @T.prim_func
    def main(A: T.Buffer((1,), "int32")):
        A[0] = 11
first = Repeated

@I.ir_module()
class Repeated:
    @T.prim_func
    def main(A: T.Buffer((1,), "int32")):
        A[0] = 22
second = Repeated
"""
    filename = "<gallery-cached-classes>" if cached else str(tmp_path / "gallery.py")
    if cached:
        monkeypatch.setitem(
            linecache.cache, filename, (len(source), None, source.splitlines(True), filename)
        )
    else:
        (tmp_path / "gallery.py").write_text(source)
    module = _execute(source, filename, monkeypatch)
    for result, value, line in ((module.first, 11, 6), (module.second, 22, 13)):
        store = result["main"].body
        assert store.value.value == value
        assert store.span.line == line
        assert store.span.source_name.name == filename


def test_gallery_nested_class_preserves_definition_scope(tmp_path, monkeypatch):
    source = """
VALUE = 7

def build(n):
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((n,), "int32")):
            A[0] = VALUE
    return Module

def caller():
    VALUE = 99
    return build(4)

result = caller()
"""
    path = tmp_path / "nested_gallery.py"
    path.write_text(source)
    module = _execute(source, str(path), monkeypatch)
    function = module.result["main"]
    assert function.params[0].ty.shape[0].value == 4
    assert function.body.value.value == 7


def test_gallery_empty_class_needs_no_member_source(tmp_path, monkeypatch):
    source = """
@I.ir_module
class Empty:
    pass
"""
    path = tmp_path / "empty_gallery.py"
    path.write_text(source)
    module = _execute(source, str(path), monkeypatch)
    assert len(module.Empty.functions) == 0


def test_unavailable_class_source_still_raises(monkeypatch):
    with pytest.raises(OSError, match="source code not available"):
        _execute("@I.ir_module\nclass Empty:\n    pass\n", "<missing-gallery-source>", monkeypatch)
