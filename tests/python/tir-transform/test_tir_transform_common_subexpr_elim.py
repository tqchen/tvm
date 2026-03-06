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
# ruff: noqa: F401
import hashlib

import pytest

import tvm
from tvm.ir.base import save_json
from tvm.ir.module import IRModule
from tvm.script import tir as T


def _apply_cse(func, identify_equiv_terms=False):
    """Apply CSE pass and return the transformed function."""
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.CommonSubexprElimTIR(identify_equiv_terms=identify_equiv_terms)(mod)
    return mod["main"]


def _check(original, expected, identify_equiv_terms=False):
    """Apply CSE and check structural equality."""
    result = _apply_cse(original, identify_equiv_terms)
    tvm.ir.assert_structural_equal(result, expected.with_attr("global_symbol", "main"))


# =====================================================================
# T1: Basic multi-level CSE
# =====================================================================
def test_basic():
    @T.prim_func
    def before(B: T.Buffer((50,), "int32"), i1: T.int32, i2: T.int32, z3: T.int32):
        z1 = T.Bind(1)
        z2 = T.Bind(2)
        B[i1] = z1 + z2
        x = T.Bind(1)
        y = T.Bind(1)
        a = T.Bind((x + y) + (z1 + z2))
        b = T.Bind((x + y) + z3)
        B[i2] = a + b

    @T.prim_func
    def expected(B: T.Buffer((50,), "int32"), i1: T.int32, i2: T.int32, z3: T.int32):
        z1 = T.Bind(1)
        z2 = T.Bind(2)
        cse_v1 = T.Bind(z1 + z2)
        B[i1] = cse_v1
        x = T.Bind(1)
        y = T.Bind(1)
        cse_v2 = T.Bind(x + y)
        a = T.Bind(cse_v2 + cse_v1)
        b = T.Bind(cse_v2 + z3)
        B[i2] = a + b

    _check(before, expected)


# =====================================================================
# T2: If -- single-branch CSE
# =====================================================================
def test_if_single_branch():
    @T.prim_func
    def before(
        B: T.Buffer((50,), "int32"),
        i1: T.int32,
        i2: T.int32,
        i3: T.int32,
        y: T.int32,
        z: T.int32,
    ):
        b = T.Bind(1)
        if b:
            B[i1] = y + z
            B[i2] = y + z
        else:
            B[i3] = y

    @T.prim_func
    def expected(
        B: T.Buffer((50,), "int32"),
        i1: T.int32,
        i2: T.int32,
        i3: T.int32,
        y: T.int32,
        z: T.int32,
    ):
        b = T.Bind(1)
        if b:
            cse_v1 = T.Bind(y + z)
            B[i1] = cse_v1
            B[i2] = cse_v1
        else:
            B[i3] = y

    _check(before, expected)


# =====================================================================
# T3: If -- both-branch CSE
# =====================================================================
def test_if_both_branches():
    @T.prim_func
    def before(
        B: T.Buffer((50,), "int32"),
        i1: T.int32,
        i2: T.int32,
        i3: T.int32,
        y: T.int32,
        z: T.int32,
    ):
        b = T.Bind(1)
        if b:
            B[i1] = y + z
            B[i2] = y
        else:
            B[i3] = y + z

    @T.prim_func
    def expected(
        B: T.Buffer((50,), "int32"),
        i1: T.int32,
        i2: T.int32,
        i3: T.int32,
        y: T.int32,
        z: T.int32,
    ):
        b = T.Bind(1)
        cse_v1 = T.Bind(y + z)
        if b:
            B[i1] = cse_v1
            B[i2] = y
        else:
            B[i3] = cse_v1

    _check(before, expected)


# =====================================================================
# T4: Cascade CSE
# =====================================================================
def test_cascade():
    @T.prim_func
    def before(
        B: T.Buffer((50,), "int32"),
        i1: T.int32,
        i2: T.int32,
        i3: T.int32,
        x: T.int32,
        y: T.int32,
        z: T.int32,
    ):
        B[i1] = (x + y) + z
        B[i2] = (x + y) + z
        B[i3] = x + y

    @T.prim_func
    def expected(
        B: T.Buffer((50,), "int32"),
        i1: T.int32,
        i2: T.int32,
        i3: T.int32,
        x: T.int32,
        y: T.int32,
        z: T.int32,
    ):
        cse_v2 = T.Bind(x + y)
        cse_v1 = T.Bind(cse_v2 + z)
        B[i1] = cse_v1
        B[i2] = cse_v1
        B[i3] = cse_v2

    _check(before, expected)


# =====================================================================
# T5: No change when no duplication
# =====================================================================
def test_no_duplication():
    @T.prim_func
    def before(x: T.int32, y: T.int32, z: T.int32):
        a = T.Bind(x + (y + z))
        T.evaluate(a)

    @T.prim_func
    def expected(x: T.int32, y: T.int32, z: T.int32):
        a = T.Bind(x + (y + z))
        T.evaluate(a)

    _check(before, expected)


# =====================================================================
# T8: Deterministic output
# =====================================================================
def test_deterministic():
    NUM_TERMS = 10
    REPEATS = 10

    x = tvm.tir.Var("x", "int32")
    result = tvm.tir.Var("result", "int32")

    offsets = sorted([i + 1 for i in range(NUM_TERMS)])
    inc1 = [(x + offsets[i]) for i in range(NUM_TERMS)]
    inc2 = [(x + offsets[i]) for i in range(NUM_TERMS)]

    expression = x
    for add in inc1 + inc2:
        expression = expression + add
    body = tvm.tir.SeqStmt([tvm.tir.Bind(result, expression), tvm.tir.Evaluate(result)])
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([x], body))

    initial_hash = None
    for _ in range(REPEATS):
        out = tvm.tir.transform.CommonSubexprElimTIR()(mod)
        func = out["main"]
        json_val = save_json(func)
        json_hash = hashlib.sha256(json_val.encode()).hexdigest()
        if initial_hash is None:
            initial_hash = json_hash
        assert json_hash == initial_hash


# =====================================================================
# T9: CSE inside for-loop
# =====================================================================
def test_for_loop():
    @T.prim_func
    def before(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
        for i in range(10):
            B[i] = y + z
            B[i + 10] = y + z

    @T.prim_func
    def expected(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
        for i in range(10):
            cse_v1 = T.Bind(y + z)
            B[i] = cse_v1
            B[i + 10] = cse_v1

    _check(before, expected)


# =====================================================================
# T10: CSE across for-loop and outer scope
# =====================================================================
def test_for_hoist():
    @T.prim_func
    def before(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
        B[0] = y + z
        for i in range(10):
            B[i + 1] = y + z

    @T.prim_func
    def expected(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
        cse_v1 = T.Bind(y + z)
        B[0] = cse_v1
        for i in range(10):
            B[i + 1] = cse_v1

    _check(before, expected)


# =====================================================================
# T11: Cannot-lift -- expressions containing BufferLoad
# =====================================================================
def test_cannot_lift_bufferload():
    @T.prim_func
    def before(A: T.Buffer((50,), "int32"), B: T.Buffer((50,), "int32")):
        B[0] = A[0] + A[0]
        B[1] = A[0] + A[0]

    @T.prim_func
    def expected(A: T.Buffer((50,), "int32"), B: T.Buffer((50,), "int32")):
        B[0] = A[0] + A[0]
        B[1] = A[0] + A[0]

    _check(before, expected)


# =====================================================================
# T12: Nested if -- multi-level scope LCA
# =====================================================================
def test_nested_if():
    @T.prim_func
    def before(
        B: T.Buffer((50,), "int32"),
        c1: T.int32,
        c2: T.int32,
        y: T.int32,
        z: T.int32,
    ):
        if c1:
            if c2:
                B[0] = y + z
            else:
                B[1] = y + z
        else:
            B[2] = y

    @T.prim_func
    def expected(
        B: T.Buffer((50,), "int32"),
        c1: T.int32,
        c2: T.int32,
        y: T.int32,
        z: T.int32,
    ):
        if c1:
            cse_v1 = T.Bind(y + z)
            if c2:
                B[0] = cse_v1
            else:
                B[1] = cse_v1
        else:
            B[2] = y

    _check(before, expected)


# =====================================================================
# T13: Multiple independent CSE candidates
# =====================================================================
def test_multi_independent():
    @T.prim_func
    def before(
        B: T.Buffer((50,), "int32"),
        a: T.int32,
        b: T.int32,
        c: T.int32,
        d: T.int32,
    ):
        B[0] = a + b
        B[1] = c + d
        B[2] = a + b
        B[3] = c + d

    @T.prim_func
    def expected(
        B: T.Buffer((50,), "int32"),
        a: T.int32,
        b: T.int32,
        c: T.int32,
        d: T.int32,
    ):
        cse_v1 = T.Bind(a + b)
        B[0] = cse_v1
        cse_v2 = T.Bind(c + d)
        B[1] = cse_v2
        B[2] = cse_v1
        B[3] = cse_v2

    _check(before, expected)


# =====================================================================
# T14: Expression in if-condition and branch
# =====================================================================
def test_if_condition():
    @T.prim_func
    def before(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
        if y + z > 0:
            B[0] = y + z

    @T.prim_func
    def expected(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
        cse_v1 = T.Bind(y + z)
        if cse_v1 > 0:
            B[0] = cse_v1

    _check(before, expected)


# =====================================================================
# T15: Cannot-lift -- expression containing Call
# =====================================================================
def test_cannot_lift_call():
    @T.prim_func
    def before(B: T.Buffer((50,), "int32"), x: T.int32):
        B[0] = T.call_extern("my_func", x, dtype="int32") + 1
        B[1] = T.call_extern("my_func", x, dtype="int32") + 1

    @T.prim_func
    def expected(B: T.Buffer((50,), "int32"), x: T.int32):
        B[0] = T.call_extern("my_func", x, dtype="int32") + 1
        B[1] = T.call_extern("my_func", x, dtype="int32") + 1

    _check(before, expected)


# =====================================================================
# No normalization without commoning (identify_equiv_terms=True)
# =====================================================================
def test_no_normalization_without_commoning():
    @T.prim_func
    def before(x: T.int32, y: T.int32, z: T.int32):
        a = T.Bind(x + (y + z))
        T.evaluate(a)

    @T.prim_func
    def expected(x: T.int32, y: T.int32, z: T.int32):
        a = T.Bind(x + (y + z))
        T.evaluate(a)

    _check(before, expected, identify_equiv_terms=True)


# =====================================================================
# Semantic equivalence -- distributivity (identify_equiv_terms=True)
# =====================================================================
@pytest.mark.xfail(reason="identify_equiv_terms not yet implemented in two-phase CSE")
def test_semantic_equiv_distributivity():
    @T.prim_func
    def before(
        B: T.Buffer((50,), "int32"),
        i1: T.int32,
        i2: T.int32,
        x: T.int32,
        y: T.int32,
        z: T.int32,
    ):
        B[i1] = (y + z) * x
        B[i2] = x * y + x * z

    @T.prim_func
    def expected(
        B: T.Buffer((50,), "int32"),
        i1: T.int32,
        i2: T.int32,
        x: T.int32,
        y: T.int32,
        z: T.int32,
    ):
        cse_v1 = T.Bind((y + z) * x)
        B[i1] = cse_v1
        B[i2] = cse_v1

    _check(before, expected, identify_equiv_terms=True)


# =====================================================================
# Semantic equivalence -- associativity (identify_equiv_terms=True)
# =====================================================================
@pytest.mark.xfail(reason="identify_equiv_terms not yet implemented in two-phase CSE")
def test_semantic_equiv_associativity():
    @T.prim_func
    def before(
        B: T.Buffer((50,), "int32"),
        i1: T.int32,
        i2: T.int32,
        x: T.int32,
        y: T.int32,
        z: T.int32,
    ):
        B[i1] = (x + y) + z
        B[i2] = x + (y + z)

    @T.prim_func
    def expected(
        B: T.Buffer((50,), "int32"),
        i1: T.int32,
        i2: T.int32,
        x: T.int32,
        y: T.int32,
        z: T.int32,
    ):
        cse_v1 = T.Bind(x + y + z)
        B[i1] = cse_v1
        B[i2] = cse_v1

    _check(before, expected, identify_equiv_terms=True)


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    test_basic()
    test_if_single_branch()
    test_if_both_branches()
    test_cascade()
    test_no_duplication()
    test_deterministic()
    test_for_loop()
    test_for_hoist()
    test_cannot_lift_bufferload()
    test_nested_if()
    test_multi_independent()
    test_if_condition()
    test_cannot_lift_call()
    test_no_normalization_without_commoning()
    test_semantic_equiv_distributivity()
    test_semantic_equiv_associativity()
    print("All tests passed!")
