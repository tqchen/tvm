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
"""Parser-generated binding and statement operations for relax."""

from __future__ import annotations

import builtins as _python
import numbers as _numbers
from collections.abc import Sequence
from typing import Any, NoReturn

import tvm_ffi as _ffi

from tvm import ir as _ir
from tvm import relax as _relax
from tvm import tirx as _tir
from tvm.script.ir_builder import IRBuilder as _IRBuilder
from tvm.script.ir_builder import base as _base

from .. import builder as _builder
from . import _ffi_api
from . import frame as _frame
from . import ir as _native

_Span = _base.SpanEntry | _ir.Span | tuple[_ir.SourceName, int, int, int, int] | None


def bind_(
    value: Any = _base.MISSING,
    *,
    ty: Any = None,
    name: str | None = None,
    span: _Span = None,
    name_span: _Span = None,
    frame_value: bool = False,
) -> Any:
    """Apply the dialect's ordinary assignment policy.

    Parameters
    ----------
    value : Any, optional
        Once-evaluated RHS. MISSING (the default) denotes an omitted initializer and is
        rejected unless the dialect supports the annotation-only form.
    ty : Type or annotation callable, optional
        Already-rewritten source annotation. None (the default) lets the dialect infer
        the binding type.
    name : str, optional
        Source target name. None (the default) requests no source-derived name.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.
    name_span : SpanEntry, Span or source-location tuple, optional
        Location of the target identifier. None (the default) uses span; it can differ
        from the emitted statement location.
    frame_value : bool, optional
        False by default. True names an already-entered with-target without constructing
        another binding or entering its frame.

    Returns
    -------
    Any
        The value bound to the Python target; usually a native variable, or an unchanged
        host/frame-owned value.

    Notes
    -----
    Requires the dialect construction context when producing IR. TIRx emits immutable Bind
    statements; Relax emits normalized bindings and match-casts. Unsupported
    values/annotations raise TypeError or ValueError. Mutable storage and scope declarations
    use distinct hooks. A DSL may opt into concise scope entry: register the returned child
    frame's exit callback on the active parent before child entry, then return its entered
    value. Later statements enter that child, and parent exit closes it. This is dialect
    policy; it adds no parser-owned scope state. Direct-call results and source module
    aliases bypass this operation.

    .. code:: python

        # Source
        x = value
        # Generated builder
        x = X.bind_(value, name="x")

        # TIRx concise scope entry
        tid = T.launch_thread("threadIdx.x", 128)
        # Generated builder
        tid = X.bind_(X.launch_thread("threadIdx.x", 128), name="tid")
    """
    name_span = _base.source_span(span if name_span is None else name_span)
    if frame_value:
        if isinstance(value, _python.list | _python.tuple | _ir.Array):
            for index, item in enumerate(value):
                bind_(
                    item,
                    name=None if name is None else f"{name}_{index}",
                    span=_base.source_span(span),
                    name_span=name_span,
                    frame_value=True,
                )
        elif isinstance(value, _ir.Var):
            if name is not None:
                _IRBuilder.name(name, value)
            _base.at_(name_span if name_span is not None else span, value)
        return value
    if value is _base.MISSING:
        raise ValueError("Relax bindings require an initializer")
    ty = None if ty is None else _builder._type(ty)
    value = _builder._value(value, ty)
    if isinstance(value, _relax.MatchCast):
        if ty is not None and not _ffi.structural_equal(ty, value.ty):
            raise TypeError("The binding annotation differs from the match-cast type")
        result = _ffi_api.EmitMatchCastWithSpan(
            value.value, value.ty, name_span, _base.source_span(span)
        )
    elif isinstance(value, _relax.Expr):
        result = _ffi_api.EmitWithSpan(value, ty, name_span, _base.source_span(span))
    else:
        return value
    if name is not None:
        _IRBuilder.name(name, result)
    return _base.at_(name_span if name_span is not None else span, result)


def emit_(value: Any, *, span: _Span = None) -> None:
    """Consume a source expression statement.

    Parameters
    ----------
    value : Any
        Once-evaluated expression result. AlreadyEmitted receipts and None produce no
        additional emission.
    span : SpanEntry, Span or source-location tuple, optional
        Location of the emitted statement and expression, or the existing node in an
        AlreadyEmitted receipt. None leaves explicit attribution unspecified. Active
        caller provenance is composed without adding a construction context; native
        frames retain the location for finalization.

    Returns
    -------
    None
        No source-visible value.

    Notes
    -----
    Requires an active dialect function/region when emitting IR. TIRx adds statements,
    evaluates expressions, and can enter concise frames; variables/text are inert and
    sequences are consumed elementwise. Relax accepts only void expressions (or
    None/AlreadyEmitted); unsupported values raise TypeError and non-void expressions raise
    ValueError. An explicit span annotates the exact previously emitted node in a receipt
    without emitting it again. Known builder results need no separate result wrapper;
    opaque source calls keep their scoped provenance before this hook. Registered
    direct_call statements bypass this hook.

    .. code:: python

        # Source
        T.evaluate(1)
        # Generated builder
        X.emit_(X.evaluate(1), span=_S[0])
    """
    if isinstance(value, _base.AlreadyEmitted):
        _base.at_(span, value)
        return None
    if value is None:
        return
    if not isinstance(value, _relax.Expr):
        raise TypeError(f"Unsupported expression statement value: {type(value).__name__}")
    result = bind_(_base.at_(span, value), name="_", span=span)
    if not isinstance(result.ty, _ir.TupleType) or len(result.ty.fields) != 0:
        raise ValueError(
            "Non-void expressions must be bound to a variable; "
            f"expression of type {result.ty} was used as a statement"
        )


def resolve_type_var_(
    name: str,
    dtype: str | _ir.Type | _ir.Var | None = None,
    *,
    value: _ir.Var | None = None,
    span: _Span = None,
) -> _ir.Var:
    """Resolve or declare a symbolic variable in the nearest native function.

    Parameters
    ----------
    name : str
        Function-local lookup key; quoted symbols do not create a Python binding.
    dtype : str, Type or Var, optional
        Explicit primitive type or supplied variable. None defaults new symbols to
        int64; existing symbols must agree with an explicit dtype.
    value : Var, optional
        Existing primitive variable to register without replacement. None creates a
        variable only if the name is not already registered.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Var
        The exact existing or newly registered variable.

    Notes
    -----
    Requires an active native function frame. The signature and resumed body use the same
    symbol map; nested functions use separate maps. Conflicting dtype/variable declarations
    or missing context raise a builder error. No statement is emitted.

    .. code:: python

        # Source
        n = T.int64()
        # Generated builder
        n = X.resolve_type_var_("n", dtype="int64")
    """
    return _base._current_function_frame().resolve_type_var(name, dtype, value=value, span=span)


def call_global_var_(function: _ir.GlobalVar, args: Sequence[Any]) -> _ir.Expr:
    """Construct a call to a declared module function.

    Parameters
    ----------
    function : GlobalVar
        Native callee reference reserved by the module declaration phase.
    args : sequence of Any
        Positional operands evaluated once, in source order. Generated global calls do
        not accept keyword arguments.

    Returns
    -------
    _ir.Expr
        The caller dialect's call expression; constructing it does not emit a statement.

    Notes
    -----
    The callee declaration must be available in the active module context. TIRx preserves
    its exact declared return type; Relax converts operands using its expression conversion.
    Invalid types or an unavailable declaration produce native builder errors. Source-call
    handling attaches locations to the returned expression.

    .. code:: python

        # Source
        Module.callee(x)
        # Generated builder
        X.call_global_var_(callee_reference, [x])
    """
    return _relax.Call(function, [_relax.utils.convert_to_expr(value) for value in args])


def decl_mutable_var_(
    value: Any = _base.MISSING,
    *,
    ty: Any = None,
    name: str | None = None,
    span: _Span = None,
    name_span: _Span = None,
) -> NoReturn:
    """Introduce explicitly declared mutable storage.

    Parameters
    ----------
    value : Any, optional
        Once-evaluated storage handle for a call declaration, or initializer for an
        annotation declaration. MISSING (the default) means no initializer.
    ty : Type or annotation callable, optional
        Scalar or vector storage annotation. None (the default) identifies an already-
        created storage handle.
    name : str, optional
        Source target name. None (the default) requests no source-derived name.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.
    name_span : SpanEntry, Span or source-location tuple, optional
        Location of the target identifier. None (the default) uses span; it can differ
        from the emitted statement location.

    Returns
    -------
    NoReturn
        Always raises TypeError; Relax does not support this imperative operation.
        No frame is entered and no binding, store or statement is created.

    Notes
    -----
    TIRx requires an active primitive function; scalar primitive annotations allocate local
    storage and may initialize it. Vector annotations allocate storage but reject an
    initializer. Invalid handle/type combinations raise TypeError or ValueError. Relax
    always rejects mutable storage with TypeError. Declaration syntax takes precedence over
    any outer mutable target name.

    .. code:: python

        # Source
        x: T.int32 = 1
        # Generated builder
        x = X.decl_mutable_var_(1, ty=X.int32, name="x")
    """
    raise TypeError("Relax does not support mutable local storage")


def set_mutable_var_(target: Any, value: Any, *, span: _Span = None) -> NoReturn:
    """Emit an update through an existing mutable handle without rebinding it.

    Parameters
    ----------
    target : TensorLoad, scalar wrapper or one-element buffer Var
        Storage handle returned by an explicit mutable declaration. Its identity is
        retained; this argument is not a source name or a new declaration.
    value : Expr or scalar convertible to Expr
        Once-evaluated value to store. Native store checking validates its type and
        indices against the target.
    span : SpanEntry, Span or source-location tuple, optional
        Location of the emitted store. None (the default) leaves explicit location
        unspecified; existing source-call provenance is retained.

    Returns
    -------
    NoReturn
        Always raises TypeError; Relax does not support this imperative operation.
        No frame is entered and no binding, store or statement is created.

    Notes
    -----
    TIRx requires an active primitive function/statement region and appends a native store
    to that region. Scalar wrappers are unwrapped, TensorLoad indices are preserved, and
    one-element buffer targets use index zero. Unsupported targets raise TypeError; native
    type/index checks propagate. Relax always raises TypeError because its bindings are
    immutable. No new allocation or immutable binding is created.

    .. code:: python

        # Source
        x: T.int32 = 0
        x = value
        # Generated builder
        x = X.decl_mutable_var_(0, ty=X.int32, name="x")
        X.set_mutable_var_(x, value)
    """
    raise TypeError("Relax does not support mutable local storage")


def function(
    is_pure: bool = True,
    is_private: bool = False,
    *,
    decl: bool = False,
    local: bool = False,
    reference: _ir.Var | None = None,
    span: _Span = None,
) -> _frame.FunctionFrame:
    """Create the native function frame used for signature and body construction.

    Parameters
    ----------
    is_pure : bool, optional
        True by default; controls Relax purity.
    is_private : bool, optional
        False by default; True makes the function private.
    decl : bool, optional
        False (default) constructs a complete function on one entry. True collects a
        signature on the first entry and retains this frame for body re-entry.
    local : bool, optional
        False by default. True constructs a local function.
    reference : Var, optional
        Declared local function reference; None is valid except when entering a local
        body.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    IRBuilderFrame
        The native context manager. It owns params, result type, symbol map, reference
        and completed result.

    Notes
    -----
    Requires an active IRBuilder; module definitions also require its module frame. Declare
    every sibling signature before any body for forward references. Re-enter the same frame,
    define and invoke a zero-argument lexical helper inside it, then exit before validation.
    Invalid options/context raise builder errors. The frame stores its source location
    independently of exit-time ambient context.

    .. code:: python

        # Source
        @T.prim_func
        def f(a: T.int32):
            T.evaluate(a)
        # Generated builder
        with X.function() as fn:
            X.func_name("f")
            a = X.arg("a", X.int32)
            X.emit_(X.evaluate(a))
    """
    if decl:
        return _base.at_(span, _ffi_api.DeclFunction(is_pure, is_private, local))
    if local:
        if reference is None:
            raise ValueError("A local function requires its declared reference")
        return _base.at_(span, _ffi_api.LocalFunction(is_pure, reference))
    return _base.at_(span, _native.function(is_pure, is_private))


def arg(name: str, ty: Any, *, span: _Span = None) -> _ir.Var:
    """Add a parameter to the active native function signature.

    Parameters
    ----------
    name : str
        Source parameter name.
    ty : Type, Var, Buffer or callable
        Concrete rewritten annotation or existing native parameter. A callable
        annotation is evaluated; an existing variable retains identity.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Var
        The created or retained parameter variable (buffers are native variables).

    Notes
    -----
    Requires an active function signature frame. Primitive symbols use that frame's
    resolver, preserving annotation/body identity. TIRx handles buffer layout according to
    its function policy; Relax converts its type annotation. Invalid annotations or context
    raise TypeError/native builder errors.

    .. code:: python

        # Source
        def f(a: T.int32):
            pass
        # Generated builder, inside the signature frame
        a = X.arg("a", X.int32)
    """
    if not isinstance(ty, _ir.Var):
        ty = _builder._type(ty)
    if isinstance(ty, _ir.PrimType) or _ir.is_prim_var(ty):
        ty = resolve_type_var_(name, ty, span=span)
    if isinstance(ty, _ir.Var):
        return _ffi_api.ArgVar(name, ty)
    return _base.at_(span, _native.arg(name, _builder._type(ty)))


def func_ret_type(annotation: Any, *, span: _Span = None) -> None:
    """Set the active native function's return annotation.

    Parameters
    ----------
    annotation : Type, Expr or callable
        Rewritten return annotation; expression annotations supply their type. None
        denotes a void/empty tuple return as supported by the dialect.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    None
        No value.

    Notes
    -----
    Requires a function signature frame. Resolves callable annotations and records the type;
    no body is emitted. Unsupported or conflicting types raise native builder errors. The
    completed function retains its frame span.

    .. code:: python

        # Source
        def f() -> T.int32:
            return 1
        # Generated builder, inside the signature frame
        X.func_ret_type(X.int32)
    """
    return _native.func_ret_type(_builder._type(_base._return_annotation(annotation)))


def if_(condition: Any, *, span: _Span = None) -> _frame.IfFrame:
    """Create the native conditional frame for a source if statement.

    Parameters
    ----------
    condition : Expr or bool
        Already-evaluated predicate; both branch bodies construct IR without testing it
        in Python.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    IRBuilderFrame
        A native conditional context manager. Relax exposes its same-named merged output
        as frame.var after exit.

    Notes
    -----
    Requires an active function region. Enter Then/Else frames inside it; each branch's
    lexical helper is defined and called inside that branch frame. TIRx permits a missing
    else; value-producing Relax branches must agree on their final binding name/type.
    Invalid context/condition/outputs produce builder diagnostics. Explicit constexpr
    conditions remain ordinary Python if statements.

    .. code:: python

        # Source
        if condition:
            T.evaluate(1)
        # Generated builder
        with X.if_(condition):
            with X.Then():
                def branch():
                    X.emit_(X.evaluate(1))
                branch()
    """
    return _base.at_(span, _native.If(condition))


def Then(*, span: _Span = None) -> _frame.ThenFrame:
    """Create the true branch of the active conditional.

    Parameters
    ----------
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    IRBuilderFrame
        The native branch context manager; its exit finalizes its region.

    Notes
    -----
    Requires an active ``if_`` frame and records the true branch. Invalid ordering or repeated
    branches raise native builder errors. The frame stores its source location; statements
    keep their individual locations. Lexical helpers take zero explicit arguments and run
    inside the entered frame.

    .. code:: python

        # Source
        if condition:
            pass
        else:
            pass
        # Generated builder, inside X.if_(condition)
        with X.Then():
            def branch():
                pass
            branch()
    """
    return _base.at_(span, _native.Then())


def Else(*, span: _Span = None) -> _frame.ElseFrame:
    """Create the false branch of the active conditional.

    Parameters
    ----------
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    IRBuilderFrame
        The native branch context manager; its exit finalizes its region.

    Notes
    -----
    Requires an active ``if_`` frame and records the false branch. Invalid ordering or repeated
    branches raise native builder errors. The frame stores its source location; statements
    keep their individual locations. Lexical helpers take zero explicit arguments and run
    inside the entered frame.

    .. code:: python

        # Source
        if condition:
            pass
        else:
            pass
        # Generated builder, inside X.if_(condition)
        with X.Else():
            def branch():
                pass
            branch()
    """
    return _base.at_(span, _native.Else())


def return_(value: Any = None, *, span: _Span = None) -> None:
    """Record a dialect function return while continuing Python construction.

    Parameters
    ----------
    value : Any, optional
        Return operand. None (the default) means an empty tuple in Relax; TIRx requires
        an expression.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    None
        No Python control transfer or source-visible value.

    Notes
    -----
    Requires an active function/region. TIRx emits its native return operation; Relax
    records the function result. Unsupported missing/type/context combinations raise
    TypeError or native errors. A source macro retaining ordinary Python return does not
    call this hook.

    .. code:: python

        # Source
        return value
        # Generated builder
        X.return_(value)
    """
    if value is None:
        value = _relax.Tuple([])
    # Normalization may emit bindings, but an existing result keeps its own span.
    _base.with_at_group_(span, lambda: _native.func_ret_value(_builder._value(value)))


def unpack(value: Any) -> Any:
    """Expose elements for ordinary Python target unpacking.

    Parameters
    ----------
    value : Any
        Concrete IR tuple, typed tuple expression, or ordinary host iterable.

    Returns
    -------
    Any
        A Python tuple of IR fields/projections for an IR tuple; otherwise the original
        value.

    Notes
    -----
    No statement is emitted and no builder frame is entered. Known IR tuple types supply
    arity; ordinary Python performs target-count/starred-unpacking checks. Field identities
    are retained for concrete tuples. Direct-call results keep ordinary Python unpacking
    without this hook.

    .. code:: python

        # Source
        a, b = value
        # Generated builder
        left, right = X.unpack(value)
        a = X.bind_(left, name="a")
        b = X.bind_(right, name="b")
    """
    if isinstance(value, _relax.Tuple):
        return _python.tuple(value.fields)
    if isinstance(value, _relax.Expr) and isinstance(value.ty, _ir.TupleType):
        return _python.tuple(_relax.TupleGetItem(value, i) for i in range(len(value.ty.fields)))
    return value


def assert_(
    condition: Any,
    message: str | tuple[str, Sequence[Any]] | Sequence[Any] = "",
    *,
    span: _Span = None,
) -> None:
    """Emit a runtime assertion.

    Parameters
    ----------
    condition : Expr or bool
        Already-evaluated predicate.
    message : str or assertion metadata, optional
        Empty text by default. Relax requires construction-time text. TIRx also accepts
        message parts or an (error_kind, parts) pair.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    None
        No value; emits a native assertion.

    Notes
    -----
    Requires an active function/statement region. Malformed diagnostic metadata raises
    TypeError; native predicate checking propagates. Construction does not test an IR
    predicate as a host boolean.

    .. code:: python

        # Source
        assert condition, "failed"
        # Generated builder
        X.assert_(condition, "failed")
    """
    if not isinstance(message, _python.str):
        raise TypeError("An assertion message must be construction-time text")
    emit_(_base.at_(span, _native.assert_op(condition, format=message)), span=span)


def for_(
    iterable: Any, *, names: str | Sequence[str] | None = None, span: _Span = None
) -> NoReturn:
    """Configure and return the native iteration frame for a source for loop.

    Parameters
    ----------
    iterable : ForFrame or range
        Already-evaluated iteration specification. TIRx converts a Python range to its
        native serial frame.
    names : str or sequence of str, optional
        Source target names, including at most one ``*starred`` group. None (the default)
        retains constructor defaults. Native configuration expands/validates them before
        entry; names never control the entry return shape.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    NoReturn
        Always raises TypeError; Relax does not support this imperative operation.
        No frame is entered and no binding, store or statement is created.

    Notes
    -----
    TIRx requires an active primitive function before entry. Variables already have final
    names at entry, and Python performs single/multiple/starred unpacking. Invalid
    iterable/names raise TypeError, ValueError or native errors. Relax rejects imperative
    loops. A frame stores its location before deferred body finalization.

    .. code:: python

        # Source
        for i in range(n):
            T.evaluate(i)
        # Generated builder
        with X.for_(X.range_(n), names=("i",)) as (i,):
            X.emit_(X.evaluate(i))
    """
    raise TypeError("Relax does not support imperative for loops")


def break_(*, span: _Span = None) -> NoReturn:
    """Emit break for the enclosing dialect loop.

    Parameters
    ----------
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    NoReturn
        Always raises TypeError; Relax does not support this imperative operation.
        No frame is entered and no binding, store or statement is created.

    Notes
    -----
    TIRx emits the operation during construction; completed native well-formedness
    validation checks its loop placement. Relax raises TypeError. Explicit constexpr
    loops retain ordinary Python control flow.

    .. code:: python

        # Source
        break
        # Generated builder
        X.break_()
    """
    raise TypeError("Relax does not support break")


def continue_(*, span: _Span = None) -> NoReturn:
    """Emit continue for the enclosing dialect loop.

    Parameters
    ----------
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    NoReturn
        Always raises TypeError; Relax does not support this imperative operation.
        No frame is entered and no binding, store or statement is created.

    Notes
    -----
    TIRx emits the operation during construction; completed native well-formedness
    validation checks its loop placement. Relax raises TypeError. Explicit constexpr
    loops retain ordinary Python control flow.

    .. code:: python

        # Source
        continue
        # Generated builder
        X.continue_()
    """
    raise TypeError("Relax does not support continue")


def setitem(target: Any, index: Any, value: Any, *, span: _Span = None) -> NoReturn:
    """Apply an indexed assignment using already-evaluated operands.

    Parameters
    ----------
    target : buffer Var
        Destination buffer.
    index : Expr, int, slice or sequence
        Indices in written order; native buffer-store rules validate supported forms.
    value : Expr or scalar
        Once-evaluated stored value.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    NoReturn
        Always raises TypeError; Relax does not support this imperative operation.
        No frame is entered and no binding, store or statement is created.

    Notes
    -----
    TIRx requires an active statement region and preserves target/index evaluation order.
    Native shape/type/index errors propagate. Relax raises TypeError because indexed
    mutation is unsupported.

    .. code:: python

        # Source
        A[i] = value
        # Generated builder
        X.setitem(A, i, value)
    """
    raise TypeError("Relax does not support indexed assignment")


def if_then_else_(condition: Any, true_value: Any, false_value: Any) -> Any:
    """Construct conditional evaluation from eagerly constructed operands.

    Parameters
    ----------
    condition : Any
        Host boolean or scalar IR predicate.
    true_value : Any
        Already-constructed true arm.
    false_value : Any
        Already-constructed false arm.

    Returns
    -------
    Any
        The selected host value or native conditional expression.

    Notes
    -----
    No statement emission or frame entry occurs. Host predicates select an existing value;
    scalar IR uses conditional evaluation. Relax also accepts its expression arms. Operand
    construction remains eager; runtime evaluation follows dialect conditional semantics.
    Invalid combinations raise native type errors. Source-result handling supplies spans.

    .. code:: python

        # Source
        yes if condition else no
        # Generated builder
        X.if_then_else_(condition, yes, no)
    """
    if isinstance(condition, _ffi.ObjectConvertible):
        condition = condition.asobject()
    if not isinstance(condition, _ir.Expr):
        return true_value if condition else false_value
    true_value = (
        true_value.asobject() if isinstance(true_value, _ffi.ObjectConvertible) else true_value
    )
    false_value = (
        false_value.asobject() if isinstance(false_value, _ffi.ObjectConvertible) else false_value
    )
    if _ir.is_prim_expr(condition) and all(
        _ir.is_prim_expr(value)
        if isinstance(value, _ir.Expr)
        else isinstance(value, _numbers.Number)
        for value in (true_value, false_value)
    ):
        return _tir.if_then_else(condition, true_value, false_value)
    return _relax.If(condition, _builder._value(true_value), _builder._value(false_value))


def _chain_binding(variable: _ir.Var, value: _ir.Expr, body: _ir.Expr) -> _ir.Expr:
    if _ir.is_prim_expr(value) and _ir.is_prim_expr(body):
        return _tir.Let(variable, value, body)
    return _relax.SeqExpr([_relax.BindingBlock([_relax.VarBinding(variable, value)])], body)


def and_(*values: Any, chain: Sequence[Any] | None = None) -> Any:
    """Construct conjunction while preserving comparison-chain operand identity.

    Parameters
    ----------
    values : Any
        One or more eagerly constructed host/IR comparisons or boolean operands, in
        order.
    chain : sequence of Any, optional
        Original once-evaluated comparison operands. None (default) means ordinary
        conjunction. Otherwise length is len(values)+1; shared nontrivial IR operands
        receive lexical native bindings.

    Returns
    -------
    Any
        Host result or native conjunction expression.

    Notes
    -----
    No statement is emitted. TIRx supports scalar/vector conjunction; Relax supports host,
    primitive and tensor predicates. Empty operands or invalid chain arity raise
    TypeError/ValueError; native type errors propagate. Chained runtime operands are
    evaluated once and progressively short-circuited, without constructing replacement
    variables in the parser.

    .. code:: python

        # Source
        a < b < c
        # Generated builder; a, b and c have already evaluated once
        X.and_(X.lt(a, b), X.lt(b, c), chain=(a, b, c))
    """
    if chain is not None:
        from tvm.tirx.script.builder.comparison import _comparison_chain

        return _comparison_chain(values, chain, and_, _chain_binding)
    return _builder.logical_and(*values)


def or_(*values: Any) -> Any:
    """Construct a disjunction from already-evaluated operands.

    Parameters
    ----------
    values : Any
        One or more host or dialect boolean operands in source order.

    Returns
    -------
    Any
        Host result or dialect disjunction expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx supports scalar/vector predicates; Relax
    also supports tensor predicates. Empty operands raise TypeError; native type checks
    propagate. Explicit constexpr operands retain host short-circuit syntax in the generated
    program. Result spans are attached by shared source handling.

    .. code:: python

        # Source
        a or b
        # Generated builder
        X.or_(a, b)
    """
    return _builder.logical_or(*values)


def not_(value: Any) -> Any:
    """Negate a host or dialect boolean without coercing IR truth in Python.

    Parameters
    ----------
    value : Any
        Once-evaluated host boolean or primitive/tensor boolean expression.

    Returns
    -------
    Any
        Host boolean or native logical negation expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx constructs primitive negation; Relax also
    supports tensor negation. Invalid operand types propagate native errors. Shared source
    handling attaches the result span.

    .. code:: python

        # Source
        not value
        # Generated builder
        X.not_(value)
    """
    return _builder.logical_not(value)


def While(condition: Any, *, span: _Span = None) -> NoReturn:
    """Create a native while-loop frame.

    Parameters
    ----------
    condition : Expr or bool
        Loop predicate expression, constructed once and evaluated by the IR at runtime.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    NoReturn
        Always raises TypeError; Relax does not support this imperative operation.
        No frame is entered and no binding, store or statement is created.

    Notes
    -----
    TIRx requires an active function region. Invalid predicate/context produces native
    errors; Relax raises TypeError. This does not repeatedly execute the Python body. The
    frame owns its stored source span.

    .. code:: python

        # Source
        while condition:
            T.evaluate(1)
        # Generated builder
        with X.While(condition):
            X.emit_(X.evaluate(1))
    """
    raise TypeError("Relax does not support imperative while loops")


def range_(*args: Any, annotations: dict[str, Any] | None = None) -> NoReturn:
    """Normalize builtin range syntax to a native serial loop frame.

    Parameters
    ----------
    args : Expr or int
        One to three already-evaluated bounds: stop; start, stop; or start, stop, step.
        Omitted start is zero and omitted step uses the native default.
    annotations : dict, optional
        Native loop annotations. None (the default) adds none.

    Returns
    -------
    NoReturn
        Always raises TypeError; Relax does not support this imperative operation.
        No frame is entered and no binding, store or statement is created.

    Notes
    -----
    TIRx validates arity and rejects a literal zero step (TypeError/ValueError). Native
    bound/type diagnostics propagate. Relax raises TypeError. Only the resolved builtin
    range is normalized; unrelated host callables named range retain ordinary call behavior.

    .. code:: python

        # Source
        for i in range(2, n, 2):
            pass
        # Generated builder iteration specification
        X.range_(2, n, 2)
    """
    raise TypeError("Relax does not support imperative for loops")


def setattr(target: Any, name: str, value: Any, *, span: _Span = None) -> NoReturn:
    """Apply an attribute assignment using already-evaluated operands.

    Parameters
    ----------
    target : Any
        Object containing a scalar storage attribute or ordinary mutable Python
        metadata.
    name : str
        Attribute identifier, evaluated by source syntax before this hook.
    value : Any
        Once-evaluated replacement or stored value.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    NoReturn
        Always raises TypeError; Relax does not support this imperative operation.
        No frame is entered and no binding, store or statement is created.

    Notes
    -----
    TIRx stores through scalar storage attributes without replacing their handles; other
    attributes use ordinary Python setattr. Native store checks and Python attribute errors
    propagate. Relax raises TypeError. A native store needs an active function; ordinary
    host metadata updates do not.

    .. code:: python

        # Source
        state.count = value
        # Generated builder
        X.setattr(state, "count", value)
    """
    raise TypeError("Relax does not support attribute assignment")


def func_name(name: str) -> None:
    """Set the active native function's source name.

    Parameters
    ----------
    name : str
        Function identifier; used for the module reference and public symbol according
        to dialect privacy options.

    Returns
    -------
    None
        No value.

    Notes
    -----
    Requires a function frame. Mutates its signature metadata and emits no statement;
    invalid or repeated naming follows native diagnostics. No source span argument is needed
    because the frame owns its location.

    .. code:: python

        # Source
        def f():
            pass
        # Generated builder, inside the function frame
        X.func_name("f")
    """
    return _native.func_name(name)


def eq(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Construct the == comparison in written operand order.

    Parameters
    ----------
    lhs : Expr, IterVar or scalar
        Already-evaluated left operand.
    rhs : Expr, IterVar or scalar
        Already-evaluated right operand.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Expr
        A native comparison expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx compares primitive operands (including
    IterVar.var); Relax selects tensor comparison if either operand is a nonprimitive IR
    expression, otherwise primitive comparison. Native dtype/shape errors propagate. Do not
    use Python object equality to compare these operands.

    .. code:: python

        # Source
        lhs == rhs
        # Generated builder
        X.eq(lhs, rhs)
    """
    from . import comparison

    return comparison.eq(lhs, rhs, span=span)


def ne(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Construct the != comparison in written operand order.

    Parameters
    ----------
    lhs : Expr, IterVar or scalar
        Already-evaluated left operand.
    rhs : Expr, IterVar or scalar
        Already-evaluated right operand.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Expr
        A native comparison expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx compares primitive operands (including
    IterVar.var); Relax selects tensor comparison if either operand is a nonprimitive IR
    expression, otherwise primitive comparison. Native dtype/shape errors propagate. Do not
    use Python object equality to compare these operands.

    .. code:: python

        # Source
        lhs != rhs
        # Generated builder
        X.ne(lhs, rhs)
    """
    from . import comparison

    return comparison.ne(lhs, rhs, span=span)


def lt(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Construct the < comparison in written operand order.

    Parameters
    ----------
    lhs : Expr, IterVar or scalar
        Already-evaluated left operand.
    rhs : Expr, IterVar or scalar
        Already-evaluated right operand.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Expr
        A native comparison expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx compares primitive operands (including
    IterVar.var); Relax selects tensor comparison if either operand is a nonprimitive IR
    expression, otherwise primitive comparison. Native dtype/shape errors propagate. Do not
    use Python object equality to compare these operands.

    .. code:: python

        # Source
        lhs < rhs
        # Generated builder
        X.lt(lhs, rhs)
    """
    from . import comparison

    return comparison.lt(lhs, rhs, span=span)


def le(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Construct the <= comparison in written operand order.

    Parameters
    ----------
    lhs : Expr, IterVar or scalar
        Already-evaluated left operand.
    rhs : Expr, IterVar or scalar
        Already-evaluated right operand.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Expr
        A native comparison expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx compares primitive operands (including
    IterVar.var); Relax selects tensor comparison if either operand is a nonprimitive IR
    expression, otherwise primitive comparison. Native dtype/shape errors propagate. Do not
    use Python object equality to compare these operands.

    .. code:: python

        # Source
        lhs <= rhs
        # Generated builder
        X.le(lhs, rhs)
    """
    from . import comparison

    return comparison.le(lhs, rhs, span=span)


def gt(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Construct the > comparison in written operand order.

    Parameters
    ----------
    lhs : Expr, IterVar or scalar
        Already-evaluated left operand.
    rhs : Expr, IterVar or scalar
        Already-evaluated right operand.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Expr
        A native comparison expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx compares primitive operands (including
    IterVar.var); Relax selects tensor comparison if either operand is a nonprimitive IR
    expression, otherwise primitive comparison. Native dtype/shape errors propagate. Do not
    use Python object equality to compare these operands.

    .. code:: python

        # Source
        lhs > rhs
        # Generated builder
        X.gt(lhs, rhs)
    """
    from . import comparison

    return comparison.gt(lhs, rhs, span=span)


def ge(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Construct the >= comparison in written operand order.

    Parameters
    ----------
    lhs : Expr, IterVar or scalar
        Already-evaluated left operand.
    rhs : Expr, IterVar or scalar
        Already-evaluated right operand.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Expr
        A native comparison expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx compares primitive operands (including
    IterVar.var); Relax selects tensor comparison if either operand is a nonprimitive IR
    expression, otherwise primitive comparison. Native dtype/shape errors propagate. Do not
    use Python object equality to compare these operands.

    .. code:: python

        # Source
        lhs >= rhs
        # Generated builder
        X.ge(lhs, rhs)
    """
    from . import comparison

    return comparison.ge(lhs, rhs, span=span)


def check_well_formed_(function: _relax.Function) -> None:
    """Validate a completed standalone Relax function.

    Parameters
    ----------
    function : relax.Function
        Completed function after its native frame has exited.

    Returns
    -------
    None
        No value or mutation; success means the existing native validators accepted the
        program.

    Notes
    -----
    Requires completed IR, with no active construction frame. Shared I validation runs Relax
    whole-module checks, s_tir verification, then TIRx verification on each non-s_tir
    primitive function, preserving cross-function checks. Dialect X hooks validate a
    standalone completed function under that dialect policy. Invalid IR raises ValueError
    retaining native details; the parser propagates the exception unchanged.
    check_well_formed=False omits the generated call entirely.

    .. code:: python

        # Source
        @R.function
        def f():
            pass
        # Generated builder, after function frame exit
        X.check_well_formed_(function)
    """
    from tvm import s_tir

    message = (
        "Program is not well-formed. If this is deliberate, set "
        "check_well_formed=False in the top-level decorator."
    )
    if not _relax.analysis.check_well_formed(function):
        raise ValueError(message)
    try:
        s_tir.analysis.verify_well_formed(_ir.IRModule.from_expr(function))
    except Exception as error:
        raise ValueError(f"{message}\n{error}") from error


def scope_var_query_or_decl_(
    value: Any, *, name: str | None = None, span: _Span = None, name_span: _Span = None
) -> NoReturn:
    """Retain the identity of a scope query or declaration result.

    Parameters
    ----------
    value : Var, IterVar, list, tuple or Array
        The once-evaluated result of a registered scope variable operation: a native
        Var (including a pointer-typed Var), an IterVar, or a list, tuple or Array of
        these. The operation has already created or selected its variable.
    name : str, optional
        Source name for a scalar target. None (default) leaves its producer name.
        Aggregate target names do not prefix or rename individual members.
    span : SpanEntry, Span or source-location tuple, optional
        Source statement location, used for block-axis naming when name_span is
        omitted. None (default) leaves it unspecified. Other variables retain
        the producer location already supplied by source-call handling.
    name_span : SpanEntry, Span or source-location tuple, optional
        Location of the target identifier. None (the default) uses span; it can differ
        from the emitted statement location.

    Returns
    -------
    NoReturn
        Always raises TypeError; Relax does not support this imperative operation.
        No frame is entered and no binding, store or statement is created.

    Notes
    -----
    TIRx requires an active function and preserves variable identity without Bind,
    allocation, store or symbol-map canonicalization. Block axes receive source names and
    duplicate-name validation; unnamed scope variables receive a name while explicit
    producer names remain intact. Invalid result types raise TypeError and duplicate axis
    names raise ValueError. Relax rejects the category. This declaration category takes
    precedence over a same-named outer mutable storage target.

    .. code:: python

        # Source
        tid = T.thread_id_in_wg()
        # Generated builder
        tid = X.scope_var_query_or_decl_(X.thread_id_in_wg(), name="tid")
    """
    raise TypeError("Relax does not support scope variable declarations")
