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

from functools import partial as _partial
from functools import wraps as _wraps

import tvm_ffi as _ffi

from tvm import ir as _ir
from tvm import tirx as _tir
from tvm.script.ir_builder.base import at as _at
from tvm.script.ir_builder.base import source_span as _source_span
from tvm.script.parser.protocol_registry import args_policy as _args_policy
from tvm.script.parser.protocol_registry import constexpr as constexpr
from tvm.script.parser.protocol_registry import (
    register_mutable_var_decl as _register_mutable_var_decl,
)
from tvm.script.parser.protocol_registry import register_result_members as _register_result_members
from tvm.tirx.buffer import _BufferMethods
from tvm.tirx.lang.alloc_pool import SMEMPool as SMEMPool
from tvm.tirx.lang.alloc_pool import TMEMPool as TMEMPool

from . import ir as _native
from . import tirx as tile
from .ir import *
from .ir import Bind as bind
from .ir import boolean as bool  # pylint: disable=redefined-builtin
from .parser_protocol import Else as Else
from .parser_protocol import Then as Then
from .parser_protocol import While as While
from .parser_protocol import and_ as and_
from .parser_protocol import arg as arg
from .parser_protocol import assert_ as assert_
from .parser_protocol import bind_ as bind_
from .parser_protocol import break_ as break_
from .parser_protocol import call_global_var_ as call_global_var_
from .parser_protocol import check_well_formed_ as check_well_formed_
from .parser_protocol import continue_ as continue_
from .parser_protocol import decl_mutable_var_ as decl_mutable_var_
from .parser_protocol import emit_ as emit_
from .parser_protocol import eq as eq
from .parser_protocol import for_ as for_
from .parser_protocol import func_name as func_name
from .parser_protocol import func_ret_type as func_ret_type
from .parser_protocol import function as function
from .parser_protocol import ge as ge
from .parser_protocol import gt as gt
from .parser_protocol import if_ as if_
from .parser_protocol import if_then_else_ as if_then_else_
from .parser_protocol import le as le
from .parser_protocol import lt as lt
from .parser_protocol import ne as ne
from .parser_protocol import not_ as not_
from .parser_protocol import or_ as or_
from .parser_protocol import range_ as range_
from .parser_protocol import resolve_type_var_ as resolve_type_var_
from .parser_protocol import return_ as return_
from .parser_protocol import set_mutable_var_ as set_mutable_var_
from .parser_protocol import setattr as setattr
from .parser_protocol import setitem as setitem
from .parser_protocol import unpack as unpack
from .tirx import cluster as cluster
from .tirx import cta as cta
from .tirx import thread as thread
from .tirx import warp as warp
from .tirx import warpgroup as warpgroup
from .tirx import wg as wg
from .utils import buffer_proxy as buffer_proxy
from .utils import frame_scope as frame_scope
from .utils import seq_scope as seq_scope

If = if_
For = for_

# Syntax capability: mutable declaration policies apply only in this dialect.
supports_mutable_declarations = True

is_type_var = _ir.is_prim_var


def type_var(name, *, dtype=None, span=None):
    """Construct an explicit standalone primitive symbol."""
    return _ir.Var(name, "int64" if dtype is None else dtype, _source_span(span))


@_partial(_register_mutable_var_decl, syntax="parameter")
@_partial(_register_result_members, members=_BufferMethods)
@_args_policy(
    {
        "shape": "expr_str",
        "strides": "expr_str",
        "elem_offset": "expr_str",
        "byte_offset": "expr_str",
        "allocated_addr": "expr_str",
    },
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
    return _at(span, _native.ptr(dtype, storage_scope))


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
    from tvm.script.ir_builder.base import AlreadyEmitted

    if isinstance(value, AlreadyEmitted):
        return None
    return emit_(value)


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


@_register_mutable_var_decl
def alloc_scalar(dtype="float32", scope="global"):
    """Allocate scalar storage and return its load expression."""
    value = _native.alloc_scalar(dtype, scope)
    return value.scalar if isinstance(value, _native.scalar_wrapper) else value


@_register_mutable_var_decl
def local_scalar(dtype="float32"):
    """Allocate scalar storage in local memory."""
    return alloc_scalar(dtype, "local")


@_register_mutable_var_decl
def shared_scalar(dtype="float32"):
    """Allocate scalar storage in shared memory."""
    return alloc_scalar(dtype, "shared")


@_register_mutable_var_decl
@_partial(_register_result_members, members=_BufferMethods)
@_args_policy(
    {
        "shape": "expr_str",
        "strides": "expr_str",
        "elem_offset": "expr_str",
        "allocated_addr": "expr_str",
    }
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


def __getattr__(name):
    """Expose registered backend construction namespaces."""
    return _native._get_script_namespace(name)
