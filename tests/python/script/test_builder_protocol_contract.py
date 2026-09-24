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
"""Parsed argument-policy behavior across successful and failed construction."""

import pytest

from tvm.script.parser import protocol_registry as registry


def test_registration_persists_across_successful_and_failed_parse_contexts(language):
    calls = []
    first, invalid, second = object(), object(), object()
    failure = RuntimeError("constructor failed")

    @registry.args_policy("X.constructor", {"device": "global_info"})
    def constructor(device):
        calls.append(device)
        if device is invalid:
            raise failure
        return device

    language.X.constructor = constructor
    source = (
        '@X.script\ndef main():\n    X.record(X.constructor(device="mesh"))\n'
        '    X.record(alias(device="mesh"))\n'
    )
    language.global_infos["mesh"] = first
    assert language.parse(source, alias=constructor).body == [("emit", first), ("emit", "mesh")]
    language.global_infos["mesh"] = invalid
    with pytest.raises(RuntimeError, match="constructor failed") as caught:
        language.parse(source, alias=constructor)
    assert caught.value is failure
    language.global_infos["mesh"] = second
    assert language.parse(source, alias=constructor).body == [("emit", second), ("emit", "mesh")]
    assert calls == [first, "mesh", invalid, second, "mesh"]
