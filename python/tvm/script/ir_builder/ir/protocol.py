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
"""The source-to-builder protocol shared by TVMScript dialects.

The parser preserves source syntax and evaluation order while rewriting it into
ordinary Python. Its state describes syntax, never IR values. Generated code
uses ``I`` for shared module and source-location support and ``X`` for the
current dialect's builders and protocol hooks. Native function frames own
parameters, symbols and completed functions; module and region frames own their
references and results. A declaration entry reserves each function signature
before any body entry, allowing sibling calls without another ownership record.
Syntax policy registration belongs to ``tvm.script.parser.protocol``.

For example, source functions can share a symbolic shape spelling while each
function retains its own symbol identity:

.. code-block:: python

    @I.ir_module
    class Module:
        @T.prim_func
        def first(a: T.Buffer(("n",), "float32")):
            n = T.int64()
            Module.second(a)

        @T.prim_func
        def second(a: T.Buffer(("n",), "float32")):
            T.evaluate(a[0])

The expected builder program uses ``X`` for the TIRx builder. Source-location
wrappers are omitted here for clarity. Quoted ``"n"`` is a lookup without a
Python binding; the explicit dtype declaration deliberately binds Python ``n``.

.. code-block:: python

    def first_body(fn):
        with fn:
            a = fn.params[0]
            n = X.resolve_type_var_("n", dtype="int64")
            X.emit_(X.call_global_var_(second.reference, [a]))

    def second_body(fn):
        with fn:
            a = fn.params[0]
            X.emit_(X.evaluate(a[0]))

    with IRBuilder() as builder:
        with I.ir_module():
            with X.function(decl=True) as first:
                X.func_name("first")
                X.arg("a", X.Buffer((X.resolve_type_var_("n"),), "float32"))
            with X.function(decl=True) as second:
                X.func_name("second")
                X.arg("a", X.Buffer((X.resolve_type_var_("n"),), "float32"))
            first_body(first)
            second_body(second)
        result = builder.get()

The function stubs below document dialect hooks, implemented in each dialect's
``builder.protocol`` and exported through ``X``. They introduce no base class or
runtime dispatcher. ``resolve_global_info`` is the real shared ``I`` operation.
"""

from ..base import MISSING, IRBuilder
from .frame import IRModuleFrame


def resolve_global_info(content):
    """Resolve a module-owned reference, preserving concrete objects unchanged.

    ``"cuda:1"`` selects the second CUDA device; ``"mesh[0]"`` selects a
    named map entry. The nearest native module supplies its existing map.
    Missing module context raises ValueError; no global information is created.
    """
    if not isinstance(content, str):
        return content
    if IRBuilder.is_in_scope():
        for frame in reversed(IRBuilder.current().frames):
            if isinstance(frame, IRModuleFrame):
                return frame.resolve_global_info(content)
    raise ValueError("Global-info lookup requires an enclosing module frame")


def resolve_type_var_(name, dtype=None, *, value=None, span=None):
    """Resolve a name in the nearest native function's symbol map.

    Quoted symbols produce a lookup only; explicit dtype declarations assign
    the returned Var to the source Python name. Existing symbols retain their
    identity. A missing name uses the declared dtype or int64; ``value`` can
    supply an existing primitive Var. ``span`` locates a newly created symbol.
    """
    raise NotImplementedError


def bind_(value=MISSING, *, ty=None, name=None, span=None, name_span=None, frame_value=False):
    """Bind an ordinary RHS and return its dialect value.

    ``x = expr`` becomes ``x = X.bind_(expr, name="x")``. ``ty`` carries a
    source annotation, ``span`` the value location and ``name_span`` an optional
    separate name location. ``frame_value`` names an existing with-target.
    Already-bound values bypass construction. Mutable allocation and stores
    have their own hooks below.
    """
    raise NotImplementedError


def emit_(value, *, span=None):
    """Consume a source expression statement under its optional source span.

    ``expr`` becomes ``X.emit_(expr)``. Dialects validate statement values and
    emit their effects; self-emitting receipts prevent duplicate emission.
    The operation has no source-visible result.
    """
    raise NotImplementedError


def decl_mutable_var_(value=MISSING, *, ty=None, name=None, span=None, name_span=None):
    """Introduce storage for an explicitly registered mutable declaration.

    A call declaration passes its once-evaluated storage as ``value``; an
    annotation passes ``ty`` and an optional initializer. Return the storage
    handle bound to ``name``. ``span`` locates allocation; ``name_span`` can
    supply the separate binding location.
    """
    raise NotImplementedError


def set_mutable_var_(target, value, *, span=None):
    """Emit a store into a previously declared mutable handle.

    After a mutable declaration, ``x = expr`` becomes
    ``X.set_mutable_var_(x, expr)``. Operands are evaluated once; ``x`` is not
    rebound. ``span`` locates the store and the operation returns no value.
    """
    raise NotImplementedError


def call_global_var_(function, args):
    """Build a call to a plain GlobalVar using the caller dialect's policy.

    ``function`` is the native reference and ``args`` its once-evaluated
    positional arguments. Return the caller dialect's call expression,
    preserving declared callee type information when it is available.
    """
    raise NotImplementedError


def module_member_(name, value):
    """Register a concrete function member or retain an ordinary class value."""
    from tvm.ir import BaseFunc

    from .ir import decl_function, def_function

    if isinstance(value, BaseFunc):
        reference = decl_function(name, value)
        def_function(name, value)
        return reference
    return value
