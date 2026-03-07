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

import tvm
from tvm.ir.base import save_json
from tvm.script import tir as T


def test_basic():
    """Basic multi-level CSE: two common subexprs at different scoping levels."""

    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), i1: T.int32, i2: T.int32, z3: T.int32):
            z1 = T.Bind(1)
            z2 = T.Bind(2)
            B[i1] = z1 + z2
            x = T.Bind(1)
            y = T.Bind(1)
            a = T.Bind((x + y) + (z1 + z2))
            b = T.Bind((x + y) + z3)
            B[i2] = a + b

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), i1: T.int32, i2: T.int32, z3: T.int32):
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

    result = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(result, After)


def test_if_single_branch():
    """Duplicated expression only in then-branch stays inside then-branch."""

    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
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

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(
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

    result = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(result, After)


def test_if_both_branches():
    """Duplicated expression in both branches is hoisted before the if."""

    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
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

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(
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

    result = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(result, After)


def test_cascade():
    """Cascading CSE: introducing (x+y)+z creates opportunity for x+y."""

    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
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

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(
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

    result = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(result, After)


def test_no_duplication():
    """No change when no expression is duplicated."""

    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(x: T.int32, y: T.int32, z: T.int32):
            a = T.Bind(x + (y + z))
            T.evaluate(a)

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(x: T.int32, y: T.int32, z: T.int32):
            a = T.Bind(x + (y + z))
            T.evaluate(a)

    result = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(result, After)


def test_deterministic():
    """Multiple runs on same input produce identical output."""
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
        out = tvm.tir.transform.CommonSubexprElim()(mod)
        func = out["main"]
        json_val = save_json(func)
        json_hash = hashlib.sha256(json_val.encode()).hexdigest()
        if initial_hash is None:
            initial_hash = json_hash
        assert json_hash == initial_hash


def test_for_loop():
    """Common sub-expression inside a for-loop body."""

    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
            for i in range(10):
                B[i] = y + z
                B[i + 10] = y + z

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
            for i in range(10):
                cse_v1 = T.Bind(y + z)
                B[i] = cse_v1
                B[i + 10] = cse_v1

    result = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(result, After)


def test_for_hoist():
    """Expression appears both outside and inside a for-loop; hoisted to outer scope."""

    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
            B[0] = y + z
            for i in range(10):
                B[i + 1] = y + z

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
            cse_v1 = T.Bind(y + z)
            B[0] = cse_v1
            for i in range(10):
                B[i + 1] = cse_v1

    result = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(result, After)


def test_cannot_lift_bufferload():
    """Expressions containing BufferLoad are ineligible."""

    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((50,), "int32"), B: T.Buffer((50,), "int32")):
            B[0] = A[0] + A[0]
            B[1] = A[0] + A[0]

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((50,), "int32"), B: T.Buffer((50,), "int32")):
            B[0] = A[0] + A[0]
            B[1] = A[0] + A[0]

    result = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(result, After)


def test_nested_if():
    """Expression in both branches of inner if; binding at inner if's parent scope."""

    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
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

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(
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

    result = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(result, After)


def test_multi_independent():
    """Several independent expressions, each duplicated."""

    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
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

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(
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

    result = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(result, After)


def test_if_condition():
    """Expression in if-condition and branch; hoisted before the if."""

    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
            if y + z > 0:
                B[0] = y + z

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), y: T.int32, z: T.int32):
            cse_v1 = T.Bind(y + z)
            if cse_v1 > 0:
                B[0] = cse_v1

    result = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(result, After)


def test_cannot_lift_call():
    """Expressions containing Call are ineligible."""

    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), x: T.int32):
            B[0] = T.call_extern("my_func", x, dtype="int32") + 1
            B[1] = T.call_extern("my_func", x, dtype="int32") + 1

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(B: T.Buffer((50,), "int32"), x: T.int32):
            B[0] = T.call_extern("my_func", x, dtype="int32") + 1
            B[1] = T.call_extern("my_func", x, dtype="int32") + 1

    result = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(result, After)


def test_no_single_use_binding():
    """Sub-expression fully consumed by parent should not get its own binding."""

    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
            B: T.Buffer((50,), "int32"),
            x: T.int32,
            y: T.int32,
            z: T.int32,
        ):
            B[0] = (x + y) + z
            B[1] = (x + y) + z

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(
            B: T.Buffer((50,), "int32"),
            x: T.int32,
            y: T.int32,
            z: T.int32,
        ):
            cse_v1 = T.Bind((x + y) + z)
            B[0] = cse_v1
            B[1] = cse_v1

    result = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(result, After)


def test_no_normalization_without_commoning():
    """Single-occurrence expression is not normalized."""

    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(x: T.int32, y: T.int32, z: T.int32):
            a = T.Bind(x + (y + z))
            T.evaluate(a)

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(x: T.int32, y: T.int32, z: T.int32):
            a = T.Bind(x + (y + z))
            T.evaluate(a)

    result = tvm.tir.transform.CommonSubexprElim()(Before)
    tvm.ir.assert_structural_equal(result, After)


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
    test_no_single_use_binding()
    test_no_normalization_without_commoning()
