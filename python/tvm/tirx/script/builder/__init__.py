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
"""Concrete TIRx construction operations over the shared native IRBuilder stack."""

import builtins as _python
from functools import wraps as _wraps

import tvm_ffi as _ffi

from tvm import ir as _ir
from tvm import tirx as _tir
from tvm.script.ir_builder import IRBuilder as _IRBuilder
from tvm.script.ir_builder import ir as _I
from tvm.script.ir_builder.base import MISSING as _MISSING
from tvm.script.ir_builder.base import BypassBind as _BypassBind
from tvm.script.ir_builder.base import _construction_span, _return_annotation
from tvm.script.ir_builder.base import at as _at
from tvm.script.ir_builder.base import source_span as _source_span
from tvm.script.parser.protocol import constexpr as constexpr
from tvm.script.parser.protocol import expr_str_args as _expression_args
from tvm.script.parser.protocol import register_type_var_decl as _register_type_var_decl
from tvm.tirx.lang.alloc_pool import SMEMPool as SMEMPool
from tvm.tirx.lang.alloc_pool import TMEMPool as TMEMPool

from . import _ffi_api
from . import frame as _frame
from . import ir as _native
from . import tirx as tile
from .comparison import eq as eq
from .comparison import ge as ge
from .comparison import gt as gt
from .comparison import le as le
from .comparison import lt as lt
from .comparison import ne as ne
from .ir import *
from .ir import Bind as bind
from .ir import boolean as bool  # pylint: disable=redefined-builtin
from .protocol import bind_ as bind_
from .protocol import call_global_var_ as call_global_var_
from .protocol import decl_mutable_var_ as decl_mutable_var_
from .protocol import emit_ as emit_
from .protocol import resolve_type_var_ as resolve_type_var_
from .protocol import set_mutable_var_ as set_mutable_var_
from .tirx import cluster as cluster
from .tirx import cta as cta
from .tirx import thread as thread
from .tirx import warp as warp
from .tirx import warpgroup as warpgroup
from .tirx import wg as wg
from .utils import buffer_proxy as buffer_proxy
from .utils import frame_scope as frame_scope
from .utils import seq_scope as seq_scope

# Syntax capability: mutable declaration policies apply only in this dialect.
supports_mutable_declarations = True

is_type_var = _ir.is_prim_var


def type_var(name, *, dtype=None, span=None):
    """Construct an explicit standalone primitive symbol."""
    return _ir.Var(name, "int64" if dtype is None else dtype, _source_span(span))


@_expression_args(
    "shape",
    "strides",
    "elem_offset",
    "byte_offset",
    "allocated_addr",
    introduce=True,
    compound_declarations=True,
    as_type=True,
)
def Buffer(
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    byte_offset=None,
    scope="global",
    align=0,
    offset_factor=0,
    layout="default",
    allocated_addr=None,
    buffer_name="",
    *,
    span=None,
):
    """The buffer declaration function.

    Parameters
    ----------
    shape : Union[List[Expr], Tuple[Expr], Expr, Integral]
        The shape of the buffer prior to flattening.

    dtype : str
        The data type in the content of the buffer.

    data : Var
        The pointer to the head of the data.

    strides : List[Expr]
        The strides of each dimension.

    elem_offset : Expr
        The offset in terms of number of dtype elements (including lanes).

    byte_offset : Expr, optional
        The offset in bytes, as an alternative to elem_offset.

    scope : str
        The optional storage scope of buffer data pointer.

    align : int
        The alignment requirement of data pointer in bytes.

    offset_factor : int
        The factor of elem_offset field.

    layout : str or Layout, optional
        The buffer layout; "default" selects the layout for the buffer scope.

    allocated_addr : int or tuple of int, optional
        Addresses assigned to the buffer allocation.

    buffer_name : str
        The name of the buffer.

    span : Span or source location, optional
        Source location attached to the constructed IR.

    Returns
    -------
    res : Buffer
        The declared buffer.
    """
    with _construction_span(span):
        return _at(
            span,
            _native.buffer(
                shape,
                dtype,
                data,
                strides,
                elem_offset,
                byte_offset,
                scope,
                align,
                offset_factor,
                layout,
                allocated_addr,
                buffer_name,
            ),
        )


buffer = Buffer


def Ptr(dtype, storage_scope="global", *, span=None):
    """The pointer declaration function.

    Parameters
    ----------
    dtype : str, Type or callable
        The data type of the pointer.

    storage_scope : str
        The storage scope of the pointer.

    span : Span or source location, optional
        Source location attached to the constructed IR.

    Returns
    -------
    res : Var
        The pointer.
    """
    if callable(dtype) and not isinstance(dtype, _ir.Expr):
        dtype = dtype()
    if isinstance(dtype, _ir.Expr):
        dtype = dtype.ty
    if isinstance(dtype, _ir.PrimType):
        dtype = dtype.dtype
    with _construction_span(span):
        return _at(span, _native.ptr(dtype, storage_scope))


def function(*, private=False, s_tir=False, persistent=False, decl=False, span=None):
    """Create a primitive-function definition frame.

    Parameters
    ----------
    decl : bool
        Collect a signature and retain this frame for a later body entry.
    private : bool
        Whether the function is private. Defaults to False.
    s_tir : bool
        Whether to use s_tir semantics: buffers use layout=None and completion
        wraps the body in a root SBlock. Defaults to False for tirx semantics.
    persistent : bool
        Whether this is a persistent kernel.
    span : Span or source location, optional
        Source location attached to the function frame.

    Returns
    -------
    frame : context manager
        A primitive-function construction context retaining source metadata.
    """
    with _construction_span(span):
        native = (
            _ffi_api.DeclFunction(private, s_tir, persistent)
            if decl
            else _native.prim_func(private=private, s_tir=s_tir, persistent=persistent)
        )
        return _at(span, native)


def arg(name, annotation, *, span=None):
    """The PrimFunc arguments adding function.

    Parameters
    ----------
    name : str
        The name of the argument.

    annotation : Type, Var, Buffer or callable
        The argument annotation. Type annotations construct a named variable;
        supplied variables and buffers retain their native identity.

    span : Span or source location, optional
        Source location attached to the constructed IR.

    Returns
    -------
    res : Union[Var, Buffer]
        The argument.
    """
    if callable(annotation) and not isinstance(annotation, _ir.Expr):
        annotation = annotation()
    if isinstance(annotation, _ir.PrimType) or _ir.is_prim_var(annotation):
        annotation = resolve_type_var_(name, annotation, span=span)
    elif isinstance(annotation, _ir.Type):
        annotation = _ir.Var(name, annotation)
    with _construction_span(span):
        if _tir.is_buffer_var(annotation) and annotation.ty.layout is not None:
            frames = _IRBuilder.current().frames
            if _python.any(
                isinstance(frame, _frame.PrimFuncFrame) and frame.s_tir for frame in frames
            ):
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
        return _native.arg(name, _at(span, annotation))


def func_ret_type(annotation, *, span=None):
    """Set the active primitive function's return type."""
    annotation = _return_annotation(annotation)
    if callable(annotation) and not isinstance(annotation, _ir.Expr | _ir.Type):
        annotation = annotation()
    if isinstance(annotation, _ir.Expr):
        annotation = annotation.ty
    with _construction_span(span):
        return _native.func_ret(annotation)


def _as_expr(value):
    if isinstance(value, _ffi.ObjectConvertible):
        value = value.asobject()
    if isinstance(value, _ir.Expr):
        return value
    if isinstance(value, str):
        return _ir.StringImm(value)
    if isinstance(value, list | tuple):
        return _ir.Tuple([_as_expr(item) for item in value])
    return _tir.const(value)


def emit(value):
    """Emit a standalone value, preserving already-emitted statement receipts."""
    from tvm.script.ir_builder.base import BypassEmit

    if isinstance(value, BypassEmit):
        return None
    return emit_(value)


def setitem(target, key, value, *, span=None):
    """Emit an indexed store using already evaluated operands."""
    with _construction_span(span):
        buffer_store(target, value, key)


def setattr(target, name, value, *, span=None):
    """Store through a scalar attribute or update Python metadata."""
    if isinstance(value, _I.meta_var):
        _python.setattr(target, name, value.value)
        return
    previous = getattr(target, name, _MISSING)
    if isinstance(previous, _native.scalar_wrapper):
        previous = previous.scalar
    buffer = previous.source if isinstance(previous, _ir.TensorLoad) else previous
    if _tir.is_buffer_var(buffer):
        shape = buffer.ty.shape
        if len(shape) == 1 and bool(shape[0] == 1):
            set_mutable_var_(previous, value, span=span)
            return
    _python.setattr(target, name, value)


def return_(value=None, *, span=None):
    """Emit a primitive-function return expression."""
    if value is None:
        raise TypeError("A primitive function return requires an expression")
    with _construction_span(span):
        _native.Return(_as_expr(value))


def _require_loop():
    for frame in reversed(_IRBuilder.current().frames):
        if isinstance(frame, _frame.ForFrame | _frame.WhileFrame):
            return
        if isinstance(frame, _frame.PrimFuncFrame):
            break
    raise ValueError("Loop control requires an enclosing primitive loop")


def break_(*, span=None):
    """Emit a break targeting the nearest primitive loop."""
    _require_loop()
    with _construction_span(span):
        _native.evaluate(_native.break_loop())


def continue_(*, span=None):
    """Emit a continue targeting the nearest primitive loop."""
    _require_loop()
    with _construction_span(span):
        _native.evaluate(_native.continue_loop())


def assert_(condition, message="", *, span=None):
    """Emit a flat native assertion with its source location."""
    kind = "RuntimeError"
    if isinstance(message, tuple):
        if len(message) != 2 or not isinstance(message[0], str):
            raise TypeError("Assertion metadata must be (error_kind, message_parts)")
        kind, message = message
    if isinstance(message, list | tuple):
        message = [str(part) for part in message]
    if not isinstance(message, list | tuple):
        message = [message]
    with _construction_span(span):
        with _native.Assert(condition, message, error_kind=kind):
            pass


def If(condition, *, span=None):
    """Create an if node.

    Parameters
    ----------
    condition : Expr
        The condition of if statement, executes the true branch if the condition is true,
        otherwise jump into the false branch.

    span : Span or source location, optional
        Source location attached to the constructed IR.

    Returns
    -------
    frame : context manager
        Construction context for the native frame, retaining source metadata.
    """
    with _construction_span(span):
        return _at(span, _native.If(condition))


def Then(*, span=None):
    """Create a native then statement region."""
    with _construction_span(span):
        return _at(span, _native.Then())


def Else(*, span=None):
    """Create a native else statement region."""
    with _construction_span(span):
        return _at(span, _native.Else())


def grid(*extents, dtype=None):
    """Create a native Cartesian loop frame.

    Parameters
    ----------
    extents : Tuple[Union[Expr, Tuple[Expr, Expr]]]
        If a single Expr is provided, it is used as the extent of the iteration.
        If a tuple of two Expr is provided, the first is the start of the iteration,
        and the second is the extent of the iteration.

    dtype : str, optional
        The dtype of every loop variable, either ``"int32"`` or ``"uint32"``. When
        omitted each loop variable takes the dtype of its own extent.

    Returns
    -------
    res : context manager
        The native loop frame; entering it constructs the loop body.
    """
    return _native.grid(*extents, dtype=dtype)


def for_(iterable, *, names=None, span=None):
    """Create native loop scope from a concrete iteration specification."""
    if names is not None and not isinstance(names, str):
        if not isinstance(names, tuple) or not _python.all(isinstance(name, str) for name in names):
            raise TypeError("Loop names must be a source identifier or tuple of identifiers")
        if _python.sum(name.startswith("*") for name in names) > 1:
            raise ValueError("Loop targets may contain only one starred group")
    with _construction_span(span):
        if isinstance(iterable, _python.range):
            iterable = _native.serial(iterable.start, iterable.stop, step=iterable.step)
        if not isinstance(iterable, _frame.ForFrame):
            raise TypeError("A primitive for loop requires an iteration specification")
        iterable.names = names
        return _at(span, iterable)


For = for_


def While(condition, *, span=None):
    """Create a while node.

    Parameters
    ----------
    condition : Expr
        The termination condition of the loop.

    span : Span or source location, optional
        Source location attached to the constructed IR.

    Returns
    -------
    frame : context manager
        Construction context for the native frame, retaining source metadata.
    """
    with _construction_span(span):
        return _at(span, _native.While(condition))


def unpack(value):
    """Project a concrete IR tuple while preserving Python iteration."""
    if isinstance(value, _BypassBind):
        wrapper = (
            _native._ScopeIdResult if isinstance(value, _native._ScopeIdResult) else _BypassBind
        )
        return _python.tuple(wrapper(item) for item in unpack(value.value))
    if isinstance(value, _ir.Tuple):
        return _python.tuple(value.fields)
    if isinstance(value, _ir.Expr) and isinstance(value.ty, _ir.TupleType):
        return _python.tuple(_ir.TupleGetItem(value, i) for i in range(len(value.ty.fields)))
    return value


def alloc_scalar(dtype="float32", scope="global"):
    """Allocate scalar storage and return its load expression."""
    value = _native.alloc_scalar(dtype, scope)
    return value.scalar if isinstance(value, _native.scalar_wrapper) else value


def local_scalar(dtype="float32"):
    """Allocate scalar storage in local memory."""
    return alloc_scalar(dtype, "local")


def shared_scalar(dtype="float32"):
    """Allocate scalar storage in shared memory."""
    return alloc_scalar(dtype, "shared")


@_expression_args(
    "shape",
    "strides",
    "elem_offset",
    "allocated_addr",
    introduce=True,
    compound_declarations=True,
)
@_wraps(_native.match_buffer)
def match_buffer(*args, **kwargs):
    """The buffer match function.

    Note
    ----
    This function will perform different behavior, depending on the type of param.
    If the param is a var in function parameter, it will create a buffer from DLTensor.
    Else if the param is a subregion of other buffers, then create a subregion match inside a block.

    Example
    -------
    Match buffer from function parameter

    .. code-block:: python

        A = T.match_buffer(a, (128, 128), dtype="float32")

    Match buffer from Buffer subregion

    .. code-block:: python

        A = T.match_buffer(B[0:128, i * 128 : i * 128 + 128], (128, 128), dtype="float32")

    Parameters
    ----------
    param : Union[Var, TensorLoad, TensorRegion]
        The parameter of the PrimFunc to match.

    shape : Union[List[Expr], Tuple[Expr], Expr, Integral]
        The type of the buffer prior to flattening.

    dtype : str
        The data type in the content of the buffer.

    data : Var
        The pointer to the head of the data.

    strides : List[Expr]
        The strides of each dimension.

    elem_offset : Expr
        The offset in terms of number of dtype elements (including lanes).

    scope : str
        The optional storage scope of buffer data pointer.

    align : int
        The alignment requirement of data pointer in bytes.

    offset_factor : int
        The factor of elem_offset field.

    layout: Optional[Union[str, Layout]]
        The layout of the buffer.

    allocated_addr : Expr or int or tuple of Expr or int, optional
        Addresses assigned to the buffer allocation.

    Returns
    -------
    res : Buffer
        The matched buffer.

    Notes
    -----
    Shape, stride, element-offset and allocation-address expression strings are
    resolved by the construction protocol before the native buffer match is created.
    """
    return _native.match_buffer(*args, **kwargs)


# Constructor identities carry syntax policy; aliases share it without wrappers.
for _constructor in vars(_native).values():
    if isinstance(_constructor, _native.DtypeConstructor):
        _register_type_var_decl(_constructor, dtype=_constructor._dtype_str)
del _constructor


def range_(*args, annotations=None):
    """Construct a native serial loop frame from Python-style bounds."""
    if len(args) == 1:
        args = (0, args[0], None)
    elif len(args) == 2:
        args = (*args, None)
    elif len(args) != 3:
        raise TypeError("range expects one to three arguments")
    if isinstance(args[2], _python.int) and args[2] == 0:
        raise ValueError("range step cannot be zero")
    return _native.serial(args[0], args[1], step=args[2], annotations=annotations)


def logical_and(*values):
    """Construct scalar or vector conjunction from eager operands."""
    if not values:
        raise TypeError("logical_and requires at least one operand")
    values = [
        value.asobject() if isinstance(value, _ffi.ObjectConvertible) else value for value in values
    ]
    result = values[0]
    for value in values[1:]:
        if not isinstance(result, _ir.Expr) and not isinstance(value, _ir.Expr):
            result = result and value
        else:
            lhs, rhs = _as_expr(result), _as_expr(value)
            result = _tir.And(lhs, rhs) if lhs.ty.is_scalar() and rhs.ty.is_scalar() else lhs & rhs
    return result


def logical_or(*values):
    """Construct scalar or vector disjunction from eager operands."""
    if not values:
        raise TypeError("logical_or requires at least one operand")
    values = [
        value.asobject() if isinstance(value, _ffi.ObjectConvertible) else value for value in values
    ]
    result = values[0]
    for value in values[1:]:
        if not isinstance(result, _ir.Expr) and not isinstance(value, _ir.Expr):
            result = result or value
        else:
            lhs, rhs = _as_expr(result), _as_expr(value)
            result = _tir.Or(lhs, rhs) if lhs.ty.is_scalar() and rhs.ty.is_scalar() else lhs | rhs
    return result


def logical_not(value):
    """Negate a host or IR value without testing IR truth in Python."""
    if isinstance(value, _ffi.ObjectConvertible):
        value = value.asobject()
    return _tir.Not(value) if isinstance(value, _ir.Expr) else not value


def select(condition, true_value, false_value):
    """Construct a scalar conditional whose runtime evaluates one arm."""
    if isinstance(condition, _ffi.ObjectConvertible):
        condition = condition.asobject()
    if not isinstance(condition, _ir.Expr):
        return true_value if condition else false_value
    return _tir.if_then_else(condition, true_value, false_value)


def if_then_else_(condition, true_value, false_value):
    """Construct a scalar conditional whose runtime evaluates one arm."""
    return select(condition, true_value, false_value)


def and_(*values, chain=None):
    """Construct the dialect's logical conjunction from evaluated values."""
    if chain is not None:
        from .comparison import _comparison_chain

        return _comparison_chain(values, chain, and_, _tir.Let)
    return logical_and(*values)


def or_(*values):
    """Construct the dialect's logical disjunction from evaluated values."""
    return logical_or(*values)


def not_(value):
    """Negate a host or IR value without testing IR truth in Python."""
    return logical_not(value)


def __getattr__(name):
    """Expose registered backend construction namespaces."""
    return _native._get_script_namespace(name)


# Registration executes after constructor exports are initialized; protocol owns
# the declaration policies used by the syntax-only prescan.
from .protocol import _register_declarations

_register_declarations()
