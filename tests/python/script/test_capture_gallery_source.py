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

from tvm import ir
from tvm.script import ir as I
from tvm.script.ir_builder import base


def _execute(source, filename, monkeypatch, X):
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("__main__", None))
    module.__dict__.update(I=I, X=X)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "__main__", module)
        exec(compile(source, filename, "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("cached", [False, True])
def test_gallery_classes_retain_distinct_source_locations(tmp_path, monkeypatch, cached, language):
    # Before: repeated class names use plain/factory decorators and file/cache source.
    # Expected builder: each class retains its own body value, filename, and line.
    _native_locations(language)
    source = """
@I.ir_module
class Repeated:
    @X.script
    def main(A: X.tensor((1,), "int32")):
        X.store(11)
first = Repeated

@I.ir_module()
class Repeated:
    @X.script
    def main(A: X.tensor((1,), "int32")):
        X.store(22)
second = Repeated
"""
    filename = "<gallery-cached-classes>" if cached else str(tmp_path / "gallery.py")
    if cached:
        monkeypatch.setitem(
            linecache.cache, filename, (len(source), None, source.splitlines(True), filename)
        )
    else:
        (tmp_path / "gallery.py").write_text(source)
    module = _execute(source, filename, monkeypatch, language.X)
    for result, value, line in ((module.first, 11, 6), (module.second, 22, 13)):
        store = result["main"].body[0][1]
        assert store.args[0].value == value
        assert store.span.line == line
        assert store.span.source_name.name == filename


def test_gallery_nested_class_preserves_definition_scope(tmp_path, monkeypatch, language):
    # Before: nested class captures n; an unrelated caller shadows global VALUE.
    # Expected builder: X.arg receives shape=(4,), body records lexical VALUE=7.
    _native_locations(language)
    source = """
VALUE = 7

def build(n):
    @I.ir_module
    class Module:
        @X.script
        def main(A: X.tensor((n,), "int32")):
            X.store(VALUE)
    return Module

def caller():
    VALUE = 99
    return build(4)

result = caller()
"""
    path = tmp_path / "nested_gallery.py"
    path.write_text(source)
    module = _execute(source, str(path), monkeypatch, language.X)
    function = module.result["main"]
    assert function.params[0].args[0].args[0][0] == 4
    assert function.body[0][1].args[0].value == 7


def test_gallery_empty_class_needs_no_member_source(tmp_path, monkeypatch, language):
    # Before: module class contains only pass.
    # Expected builder: an empty module frame still recovers the class source.
    source = """
@I.ir_module
class Empty:
    pass
"""
    path = tmp_path / "empty_gallery.py"
    path.write_text(source)
    module = _execute(source, str(path), monkeypatch, language.X)
    assert len(module.Empty) == 0


def test_unavailable_class_source_still_raises(monkeypatch, language):
    # Before: an uncached synthetic filename has no recoverable class source.
    # Expected builder: source acquisition raises OSError before construction.
    with pytest.raises(OSError, match="source code not available"):
        _execute(
            "@I.ir_module\nclass Empty:\n    pass\n",
            "<missing-gallery-source>",
            monkeypatch,
            language.X,
        )


def _native_locations(language):
    language.I.at_ = base.at_
    language.I.with_at_group_ = base.with_at_group_
    language.X.store = lambda value: ir.Call(
        ir.GlobalVar("store"), [ir.prim.IntImm("int32", value)], ret_ty="int32"
    )
