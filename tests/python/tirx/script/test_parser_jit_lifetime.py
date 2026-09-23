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
"""Deferred JIT captures retain semantic names without retaining outer payloads."""

from __future__ import annotations

import gc
import weakref

from tvm.script import tirx as T


class Payload:
    pass


def test_live_jit_releases_unused_scope_before_and_after_uncached_specializations():
    # Before: annotation-only width and unrelated payload surround a deferred JIT kernel.
    # Expected builder program: each specialization creates/discards one temporary builder.
    def make():
        payload = Payload()
        reference = weakref.ref(payload)
        width = 7

        @T.jit(private=True)
        def kernel(A: T.Buffer((width,), "int32"), *, value: T.constexpr):
            A[0] = value

        return kernel, reference

    enabled = gc.isenabled()
    gc.disable()
    try:
        kernel, reference = make()
        assert reference() is None
        first = kernel.specialize(value=1)
        assert reference() is None
        second = kernel.specialize(value=2)
        assert reference() is None
        assert first is kernel.specialize(value=1)
        assert first is not second
        assert int(first.params[0].ty.shape[0]) == 7
        assert int(second.params[0].ty.shape[0]) == 7
        assert int(first.body.value) == 1
        assert int(second.body.value) == 2
    finally:
        if enabled:
            gc.enable()


def test_annotation_capture_is_released_when_jit_owner_is_dropped():
    # Before: config is needed only by the deferred annotation, so JIT owns it.
    # Expected: deleting the only JIT owner releases config without cyclic collection.
    def make():
        config = Payload()
        config.width = 7
        reference = weakref.ref(config)

        @T.jit(private=True)
        def kernel(A: T.Buffer((config.width,), "int32")):
            A[0] = 1

        return kernel, reference

    enabled = gc.isenabled()
    gc.disable()
    try:
        kernel, reference = make()
        assert reference() is not None
        del kernel
        assert reference() is None
    finally:
        if enabled:
            gc.enable()
