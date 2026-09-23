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
"""Recording builder fixtures for the production source-to-builder parser."""

import pytest
from dummy_builder import Language

from tvm.script.ir_builder import base
from tvm.script.parser import entry


@pytest.fixture
def language(monkeypatch):
    language = Language()
    monkeypatch.setattr(entry, "builder_ir", language.I)
    return language


@pytest.fixture
def spanned_language(language):
    language.I.at_ = base.at_
    language.I.with_at_group_ = base.with_at_group_
    language.X.inline = entry.make_macro_decorator(language.X)
    return language
