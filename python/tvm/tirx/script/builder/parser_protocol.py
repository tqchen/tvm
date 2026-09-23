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
"""Parser-generated binding and statement operations for tirx."""

from __future__ import annotations

import builtins as _python
from collections.abc import Sequence
from functools import partial as _partial
from typing import Any

from tvm import ir as _ir
from tvm import tirx as _tir
from tvm.script.ir_builder import IRBuilder as _IRBuilder
from tvm.script.ir_builder import base as _base

from .. import builder as _builder
from . import _ffi_api
from . import frame as _frame
from . import ir as _native

_Span = _ir.Span | tuple[_ir.SourceName, int, int, int, int] | None


def _name(value: Any, name: str | None, span: _Span) -> Any:
    if name is not None:
        _IRBuilder.name(name, value)
    return _base.at_(span, value)


def _enter_concise(frame: _base.IRBuilderFrame) -> Any:
    # add_callback registers on the active parent before the child enters.
    # Later statements emit into the child; parent exit closes this scope.
    frame.add_callback(_partial(frame.__exit__, None, None, None))
    return frame.__enter__()


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
    span : Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.
    name_span : Span or source-location tuple, optional
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
    name_span = span if name_span is None else name_span
    if frame_value:
        if isinstance(value, _frame.SBlockFrame):
            raise TypeError("A block does not introduce an as-target value")
        if isinstance(value, _python.list | _python.tuple | _ir.Array):
            for index, item in enumerate(value):
                bind_(
                    item,
                    name=None if name is None else f"{name}_{index}",
                    span=span,
                    name_span=name_span,
                    frame_value=True,
                )
        elif isinstance(value, _ir.Var | _tir.IterVar | _tir.Layout):
            _name(value, name, name_span)
        elif isinstance(value, _ir.TensorLoad) and _tir.is_buffer_var(value.source):
            _name(value.source, name, name_span)
        return value
    if isinstance(ty, _native.LetAnnotation):
        if value is _base.MISSING:
            raise ValueError("An immutable binding requires an initializer")
        value = _builder._as_expr(value)
        variable = _name(ty.as_var(rhs_dtype=value.ty), name, name_span)
        _base.with_at_group_(span, lambda: _native.Bind(value, var=variable))
        return variable
    if ty is not None:
        annotation = ty() if callable(ty) and not isinstance(ty, _ir.Expr) else ty
        annotation = annotation.ty if isinstance(annotation, _ir.Expr) else annotation
        value = _builder._as_expr(value)
        variable = _ir.Var(name or "", annotation)
        return _name(
            _base.with_at_group_(span, lambda: _native.Bind(value, var=variable)), name, name_span
        )
    if value is _base.MISSING:
        raise ValueError("An uninitialized binding requires a scalar type annotation")
    if isinstance(value, _base.IRBuilderFrame):
        return _name(_enter_concise(_base.at_(span, value)), name, name_span)
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            bind_(item, name=None if name is None else f"{name}_{index}", span=span)
        return value
    if getattr(type(value), "_is_meta_class", False):
        return value
    if _tir.is_buffer_var(value) or isinstance(value, _tir.IterVar | _tir.Layout):
        return _name(value, name, name_span)
    if isinstance(value, _ir.Var) and not value.name:
        return _name(value, name, name_span)
    if isinstance(value, _ir.TensorRegion):
        return value
    if not isinstance(value, _ir.Expr | _python.int | _python.float | _python.bool | str):
        return value
    value = _builder._as_expr(value)
    return _name(_base.with_at_group_(span, lambda: _native.Bind(value)), name, name_span)


def emit_(value: Any, *, span: _Span = None) -> None:
    """Consume a source expression statement.

    Parameters
    ----------
    value : Any
        Once-evaluated expression result. AlreadyEmitted receipts and None produce no
        additional emission.
    span : Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

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
    ValueError. The receipt retains the exact previously emitted object. Ordinary source
    calls receive source handling before this hook; direct_call statements bypass it.

    .. code:: python

        # Source
        T.evaluate(1)
        # Generated builder
        X.emit_(X.evaluate(1))
    """
    if isinstance(value, _base.AlreadyEmitted):
        return None
    if value is None or isinstance(value, str | _ir.Var):
        return
    if isinstance(value, list | tuple | _ir.Array):
        for item in value:
            emit_(item, span=span)
        return
    if isinstance(value, _base.IRBuilderFrame):
        _enter_concise(_base.at_(span, value))
    elif hasattr(value, "frames"):
        for frame in value.frames:
            _enter_concise(_base.at_(span, frame))
    elif isinstance(value, _tir.Stmt):
        _native.add_to_parent(_base.at_(span, value))
    else:
        _base.at_(span, _native.evaluate(value))


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
    span : Span or source-location tuple, optional
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
    return _native._call_global(function, *args)


def decl_mutable_var_(
    value: Any = _base.MISSING,
    *,
    ty: Any = None,
    name: str | None = None,
    span: _Span = None,
    name_span: _Span = None,
) -> Any:
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
    span : Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.
    name_span : Span or source-location tuple, optional
        Location of the target identifier. None (the default) uses span; it can differ
        from the emitted statement location.

    Returns
    -------
    Any
        The same declared storage handle, or the handle allocated for the annotation.

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
    name_span = span if name_span is None else name_span
    if isinstance(ty, _native.LocalVectorAnnotation):
        if value is not _base.MISSING:
            raise ValueError("Vector annotation does not support an initializer")
        return _name(
            _base.with_at_group_(span, lambda: _native.alloc_local(ty.shape, ty.dtype)),
            name,
            name_span,
        )
    if ty is not None:
        annotation = ty() if callable(ty) and not isinstance(ty, _ir.Expr) else ty
        annotation = annotation.ty if isinstance(annotation, _ir.Expr) else annotation
        if not isinstance(annotation, _ir.PrimType) or str(annotation) == "handle":
            raise TypeError("Mutable scalar annotations require a primitive scalar type")
        storage = _base.with_at_group_(span, lambda: _native.local_scalar(str(annotation))).scalar
        if value is not _base.MISSING:
            set_mutable_var_(storage, value, span=span)
    else:
        storage = value.scalar if isinstance(value, _native.scalar_wrapper) else value
    if isinstance(storage, _ir.TensorLoad):
        _name(storage.source, name, name_span)
    elif _tir.is_buffer_var(storage):
        _name(storage, name, name_span)
    else:
        raise TypeError("A mutable declaration requires scalar or vector storage")
    return storage


def set_mutable_var_(
    target: _ir.TensorLoad | _ir.Var | _native.scalar_wrapper, value: Any, *, span: _Span = None
) -> None:
    """Emit an update through an existing mutable handle without rebinding it.

    Parameters
    ----------
    target : TensorLoad, scalar wrapper or one-element buffer Var
        Storage handle returned by an explicit mutable declaration. Its identity is
        retained; this argument is not a source name or a new declaration.
    value : Expr or scalar convertible to Expr
        Once-evaluated value to store. Native store checking validates its type and
        indices against the target.
    span : Span or source-location tuple, optional
        Location of the emitted store. None (the default) leaves explicit location
        unspecified; existing source-call provenance is retained.

    Returns
    -------
    None
        No value. The Python target continues to denote the original storage handle.

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
    if isinstance(target, _native.scalar_wrapper):
        target = target.scalar
    if isinstance(target, _ir.TensorLoad):
        _base.at_(span, _builder.buffer_store(target.source, value, list(target.indices)))
    elif (
        _tir.is_buffer_var(target)
        and len(target.ty.shape) == 1
        and isinstance(target.ty.shape[0], _tir.IntImm)
        and target.ty.shape[0].value == 1
    ):
        _base.at_(span, _builder.buffer_store(target, value, [0]))
    else:
        raise TypeError("A mutable assignment requires scalar storage")


def _register_declarations() -> None:
    from tvm.script.ir_builder.ir.parser_protocol import (
        direct_call,
        register_mutable_var_decl,
        register_result_members,
        register_scope_var_query_or_decl,
    )

    # Native axes and scope helpers already own their returned variables.
    # Declaration syntax shadows outer storage without adding another binding.
    for name in ("spatial", "reduce", "scan", "opaque", "remap"):
        register_scope_var_query_or_decl(getattr(_builder.axis, name))

    for name in (
        "scope_id",
        "cluster_id",
        "cta_id",
        "cta_id_in_cluster",
        "cta_id_in_pair",
        "warpgroup_id",
        "warp_id",
        "warp_id_in_wg",
        "lane_id",
        "thread_id",
        "thread_id_in_wg",
        "bind",
        "env_thread",
    ):
        register_scope_var_query_or_decl(getattr(_native, name))
    register_scope_var_query_or_decl(_builder.bind)

    from tvm.tirx import buffer as buffer_module
    from tvm.tirx import layout as layout_module

    for constructor in (
        _native.Layout,
        _native.TileLayout,
        _native.ComposeLayout,
        _native.IterVar,
        _native.iter_var,
    ):
        direct_call(constructor)
    for name in (
        "canonicalize",
        "tile",
        "direct_sum",
        "slice",
        "tile_to",
        "storage",
        "unpack",
        "broadcast",
        "pack",
    ):
        direct_call(getattr(layout_module.Layout, name))
    for name in (
        "from_iters",
        "group",
        "group_many",
        "trainium",
        "to_psum",
        "permute_dims",
        "permute_by_groups",
    ):
        method = getattr(layout_module.TileLayout, name)
        direct_call(getattr(method, "__func__", method))
    for name in (
        "tmem_datapath_layout",
        "tmem_mma_operand_layout",
        "wg_local_layout",
        "tcgen05_atom_layout",
    ):
        direct_call(getattr(layout_module, name))
    for name in (
        "get_flattened_buffer",
        "with_allocated_addr",
        "with_dtype",
        "view",
        "local",
        "permute",
        "rearrange",
        "tile",
        "chunk",
    ):
        direct_call(getattr(buffer_module._BufferMethods, name))

    # Static producer syntax identifies actual registered members on parameters
    # and unambiguous declarations; the parser never inspects returned IR types.
    from tvm.tirx import _buffer_view

    buffer_members = buffer_module._BufferMethods
    for constructor in (_builder.Buffer, _native.buffer):
        register_result_members(constructor, buffer_members)
    for namespace in (_builder, _native):
        for name in (
            "alloc_buffer",
            "alloc_local",
            "alloc_shared",
            "decl_buffer",
            "match_buffer",
        ):
            register_result_members(getattr(namespace, name), buffer_members)
    for name in (
        "get_flattened_buffer",
        "with_allocated_addr",
        "with_dtype",
        "view",
        "local",
        "permute",
        "rearrange",
    ):
        register_result_members(getattr(buffer_members, name), buffer_members)
    register_result_members(buffer_members.sub.fget, _buffer_view.SubIndexer)
    for name, indexer in (("tile", _buffer_view.TileIndexer), ("chunk", _buffer_view.ChunkIndexer)):
        register_result_members(getattr(buffer_members, name), indexer)
    for indexer in (_buffer_view.SubIndexer, _buffer_view.TileIndexer, _buffer_view.ChunkIndexer):
        direct_call(indexer.__getitem__)
        register_result_members(indexer.__getitem__, buffer_members)
    for constructor in (_native.Layout, _native.TileLayout, _native.ComposeLayout):
        register_result_members(constructor, constructor)
    for name in (
        "canonicalize",
        "tile",
        "direct_sum",
        "slice",
        "tile_to",
        "storage",
        "unpack",
        "broadcast",
        "pack",
    ):
        register_result_members(getattr(layout_module.Layout, name), layout_module.Layout)
    for name in ("from_iters", "trainium", "to_psum", "permute_dims", "permute_by_groups"):
        register_result_members(getattr(layout_module.TileLayout, name), layout_module.TileLayout)
    for name in (
        "tmem_datapath_layout",
        "tmem_mma_operand_layout",
        "wg_local_layout",
        "tcgen05_atom_layout",
    ):
        register_result_members(getattr(layout_module, name), layout_module.TileLayout)

    for constructor in vars(_native).values():
        if isinstance(constructor, _native.DtypeConstructor):
            register_mutable_var_decl(constructor, syntax="annotation")
    # These source calls explicitly declare storage. Naming preserves the
    # native handle; assignment to a scalar buffer later emits a store.
    for namespace in (_builder, _native):
        for name in (
            "local_scalar",
            "shared_scalar",
            "alloc_scalar",
            "decl_scalar",
            "alloc_buffer",
            "alloc_local",
            "alloc_shared",
            "decl_buffer",
            "match_buffer",
        ):
            register_mutable_var_decl(getattr(namespace, name), syntax="call")
    register_mutable_var_decl(_native.LocalVectorAnnotation, syntax="annotation")
    register_mutable_var_decl(_builder.Buffer, syntax="parameter")


def function(
    *,
    private: bool = False,
    s_tir: bool = False,
    persistent: bool = False,
    decl: bool = False,
    span: _Span = None,
) -> _frame.PrimFuncFrame:
    """Create the native function frame used for signature and body construction.

    Parameters
    ----------
    private : bool, optional
        False by default; True suppresses a public global symbol.
    s_tir : bool, optional
        False by default. True uses s_tir root-block and buffer-layout semantics.
    persistent : bool, optional
        False by default. True marks a persistent kernel.
    decl : bool, optional
        False (default) constructs a complete function on one entry. True collects a
        signature on the first entry and retains this frame for body re-entry.
    span : Span or source-location tuple, optional
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
    native = (
        _ffi_api.DeclFunction(private, s_tir, persistent)
        if decl
        else _native.prim_func(private=private, s_tir=s_tir, persistent=persistent)
    )
    return _base.at_(span, native)


def arg(name: str, annotation: Any, *, span: _Span = None) -> _ir.Var:
    """Add a parameter to the active native function signature.

    Parameters
    ----------
    name : str
        Source parameter name.
    annotation : Type, Var, Buffer or callable
        Concrete rewritten annotation or existing native parameter. A callable
        annotation is evaluated; an existing variable retains identity.
    span : Span or source-location tuple, optional
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
    if callable(annotation) and not isinstance(annotation, _ir.Expr):
        annotation = annotation()
    if isinstance(annotation, _ir.PrimType) or _ir.is_prim_var(annotation):
        annotation = resolve_type_var_(name, annotation, span=span)
    elif isinstance(annotation, _ir.Type):
        annotation = _ir.Var(name, annotation)
    if _tir.is_buffer_var(annotation) and annotation.ty.layout is not None:
        frames = _IRBuilder.current().frames
        if _python.any(isinstance(frame, _frame.PrimFuncFrame) and frame.s_tir for frame in frames):
            ty = annotation.ty
            annotation = _native.buffer(
                ty.shape,
                ty.dtype,
                strides=ty.strides,
                elem_offset=ty.elem_offset,
                scope=ty.storage_scope,
                align=ty.data_alignment,
                offset_factor=ty.offset_factor,
                layout=None,
                allocated_addr=list(ty.allocated_addr),
                buffer_name=name,
            )
    return _native.arg(name, _base.at_(span, annotation))


def func_ret_type(annotation: Any, *, span: _Span = None) -> None:
    """Set the active native function's return annotation.

    Parameters
    ----------
    annotation : Type, Expr or callable
        Rewritten return annotation; expression annotations supply their type. None
        denotes a void/empty tuple return as supported by the dialect.
    span : Span or source-location tuple, optional
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
    annotation = _base._return_annotation(annotation)
    if callable(annotation) and not isinstance(annotation, _ir.Expr | _ir.Type):
        annotation = annotation()
    if isinstance(annotation, _ir.Expr):
        annotation = annotation.ty
    return _native.func_ret(annotation)


def setitem(target: Any, key: Any, value: Any, *, span: _Span = None) -> None:
    """Apply an indexed assignment using already-evaluated operands.

    Parameters
    ----------
    target : buffer Var
        Destination buffer.
    key : Expr, int, slice or sequence
        Indices in written order; native buffer-store rules validate supported forms.
    value : Expr or scalar
        Once-evaluated stored value.
    span : Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    None
        No value; emits a store without rebinding the target.

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
    _base.at_(span, _builder.buffer_store(target, value, key))


def setattr(target: Any, name: str, value: Any, *, span: _Span = None) -> None:
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
    span : Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    None
        No value.

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
    previous = getattr(target, name, _base.MISSING)
    if isinstance(previous, _native.scalar_wrapper):
        previous = previous.scalar
    buffer = previous.source if isinstance(previous, _ir.TensorLoad) else previous
    if _tir.is_buffer_var(buffer):
        shape = buffer.ty.shape
        if len(shape) == 1 and _python.bool(shape[0] == 1):
            set_mutable_var_(previous, value, span=span)
            return
    _python.setattr(target, name, value)


def return_(value: Any = None, *, span: _Span = None) -> None:
    """Record a dialect function return while continuing Python construction.

    Parameters
    ----------
    value : Any, optional
        Return operand. None (the default) means an empty tuple in Relax; TIRx requires
        an expression.
    span : Span or source-location tuple, optional
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
        raise TypeError("A primitive function return requires an expression")
    _base.with_at_group_(span, lambda: _native.Return(_builder._as_expr(value)))


def _require_loop() -> None:
    for frame in reversed(_IRBuilder.current().frames):
        if isinstance(frame, _frame.ForFrame | _frame.WhileFrame):
            return
        if isinstance(frame, _frame.PrimFuncFrame):
            break
    raise ValueError("Loop control requires an enclosing primitive loop")


def break_(*, span: _Span = None) -> None:
    """Emit break for the enclosing dialect loop.

    Parameters
    ----------
    span : Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    None
        No Python control transfer.

    Notes
    -----
    TIRx requires an enclosing native ForFrame or WhileFrame inside the current primitive
    function, emits break for that loop and raises ValueError otherwise. Relax raises
    TypeError. Explicit constexpr loops retain ordinary Python control flow.

    .. code:: python

        # Source
        break
        # Generated builder
        X.break_()
    """
    _require_loop()
    _base.at_(span, _native.evaluate(_native.break_loop()))


def continue_(*, span: _Span = None) -> None:
    """Emit continue for the enclosing dialect loop.

    Parameters
    ----------
    span : Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    None
        No Python control transfer.

    Notes
    -----
    TIRx requires an enclosing native ForFrame or WhileFrame inside the current primitive
    function, emits continue for that loop and raises ValueError otherwise. Relax raises
    TypeError. Explicit constexpr loops retain ordinary Python control flow.

    .. code:: python

        # Source
        continue
        # Generated builder
        X.continue_()
    """
    _require_loop()
    _base.at_(span, _native.evaluate(_native.continue_loop()))


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
    span : Span or source-location tuple, optional
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
    kind = "RuntimeError"
    if isinstance(message, tuple):
        if len(message) != 2 or not isinstance(message[0], str):
            raise TypeError("Assertion metadata must be (error_kind, message_parts)")
        kind, message = message
    if isinstance(message, list | tuple):
        message = [str(part) for part in message]
    if not isinstance(message, list | tuple):
        message = [message]
    with _base.at_(span, _native.Assert(condition, message, error_kind=kind)):
        pass


def if_(condition: Any, *, span: _Span = None) -> _frame.IfFrame:
    """Create the native conditional frame for a source if statement.

    Parameters
    ----------
    condition : Expr or bool
        Already-evaluated predicate; both branch bodies construct IR without testing it
        in Python.
    span : Span or source-location tuple, optional
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
    span : Span or source-location tuple, optional
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
    span : Span or source-location tuple, optional
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


def for_(
    iterable: Any, *, names: str | Sequence[str] | None = None, span: _Span = None
) -> _frame.ForFrame:
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
    span : Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    IRBuilderFrame
        The same configured ForFrame; entry always returns its native variable sequence.

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
    if isinstance(iterable, _python.range):
        iterable = _native.serial(iterable.start, iterable.stop, step=iterable.step)
    if not isinstance(iterable, _frame.ForFrame):
        raise TypeError("A primitive for loop requires an iteration specification")
    iterable.set_names(names)
    return _base.at_(span, iterable)


def While(condition: Any, *, span: _Span = None) -> _frame.WhileFrame:
    """Create a native while-loop frame.

    Parameters
    ----------
    condition : Expr or bool
        Loop predicate expression, constructed once and evaluated by the IR at runtime.
    span : Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    IRBuilderFrame
        A context manager whose exit completes the loop body.

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
    return _base.at_(span, _native.While(condition))


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
    if isinstance(value, _ir.Tuple):
        return _python.tuple(value.fields)
    if isinstance(value, _ir.Expr) and isinstance(value.ty, _ir.TupleType):
        return _python.tuple(_ir.TupleGetItem(value, i) for i in range(len(value.ty.fields)))
    return value


def range_(*args: Any, annotations: dict[str, Any] | None = None) -> _frame.ForFrame:
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
    IRBuilderFrame
        An unentered serial loop frame; ``for_`` configures names and source span.

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
    if len(args) == 1:
        args = (0, args[0], None)
    elif len(args) == 2:
        args = (*args, None)
    elif len(args) != 3:
        raise TypeError("range expects one to three arguments")
    if isinstance(args[2], _python.int) and args[2] == 0:
        raise ValueError("range step cannot be zero")
    return _native.serial(args[0], args[1], step=args[2], annotations=annotations)


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
    return _builder.select(condition, true_value, false_value)


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
        from .comparison import _comparison_chain

        return _comparison_chain(values, chain, and_, _tir.Let)
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
    span : Span or source-location tuple, optional
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
    span : Span or source-location tuple, optional
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
    span : Span or source-location tuple, optional
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
    span : Span or source-location tuple, optional
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
    span : Span or source-location tuple, optional
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
    span : Span or source-location tuple, optional
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


def check_well_formed_(function: _tir.PrimFunc) -> None:
    """Validate a completed standalone primitive function.

    Parameters
    ----------
    function : PrimFunc
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
        @T.prim_func
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
    try:
        s_tir.analysis.verify_well_formed(_ir.IRModule.from_expr(function))
        if not function.attrs.get("s_tir", False):
            _tir.analysis.verify_tirx_well_formed(function)
    except Exception as error:
        raise ValueError(f"{message}\n{error}") from error


def scope_var_query_or_decl_(
    value: Any, *, name: str | None = None, span: _Span = None, name_span: _Span = None
) -> Any:
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
    span : Span or source-location tuple, optional
        Source statement location, used for block-axis naming when name_span is
        omitted. None (default) leaves it unspecified. Other variables retain
        the producer location already supplied by source-call handling.
    name_span : Span or source-location tuple, optional
        Location of the target identifier. None (the default) uses span; it can differ
        from the emitted statement location.

    Returns
    -------
    Any
        The exact input object, including the original sequence for aggregate results.

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
    name_span = span if name_span is None else name_span
    if isinstance(value, list | tuple | _ir.Array):
        for item in value:
            scope_var_query_or_decl_(
                item,
                span=span,
                name_span=name_span,
            )
        return value
    variable = value.var if isinstance(value, _tir.IterVar) else value
    if not isinstance(variable, _ir.Var):
        raise TypeError("A scope variable declaration must return a native variable")
    for frame in reversed(_IRBuilder.current().frames):
        if isinstance(frame, _frame.SBlockFrame) and _python.any(
            axis.var.same_as(variable) for axis in frame.iter_vars
        ):
            if name is not None and _python.any(
                axis.var.name == name and not axis.var.same_as(variable) for axis in frame.iter_vars
            ):
                raise ValueError(f"Duplicate block axis name {name!r}")
            _name(variable, name, name_span)
            return value
    if not variable.name and name is not None:
        _IRBuilder.name(name, variable)
    # The source call already owns the producer span; naming must not relocate it.
    return value
