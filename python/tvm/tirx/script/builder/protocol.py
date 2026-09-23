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

import builtins as _python
from functools import partial as _partial

from tvm import ir as _ir
from tvm import tirx as _tir
from tvm.script.ir_builder import IRBuilder as _IRBuilder
from tvm.script.ir_builder import ir as _I
from tvm.script.ir_builder.base import MISSING as _MISSING
from tvm.script.ir_builder.base import BypassBind as _BypassBind
from tvm.script.ir_builder.base import IRBuilderFrame as _NativeFrame
from tvm.script.ir_builder.base import _construction_span
from tvm.script.ir_builder.base import at as _at

from .. import builder as _builder
from . import frame as _frame
from . import ir as _native


def _name(value, name, span):
    if name is not None:
        _IRBuilder.name(name, value)
    return _at(span, value)


def _enter_concise(frame):
    frame.add_callback(_partial(frame.__exit__, None, None, None))
    return frame.__enter__()


def bind_(
    value=_MISSING,
    *,
    ty=None,
    name=None,
    span=None,
    name_span=None,
    frame_value=False,
):
    """Construct a named binding under the active primitive function's policy."""
    if isinstance(value, _BypassBind):
        # Scope declarations already own their native binding.  Supply only the
        # missing source name; generic bypass values retain the immediate path.
        if isinstance(value, _native._ScopeIdResult) and _ir.is_prim_var(value.value):
            if not value.value.name and name is not None:
                _IRBuilder.name(name, value.value)
        return value.value
    name_span = span if name_span is None else name_span
    # Axis and environment-thread variables already belong to native frames;
    # naming an assignment must preserve their registration identities.
    if not frame_value and _ir.is_prim_var(value):
        for frame in reversed(_IRBuilder.current().frames):
            if isinstance(frame, _frame.SBlockFrame) and _python.any(
                axis.var.same_as(value) for axis in frame.iter_vars
            ):
                if name is not None and _python.any(
                    axis.var.name == name and not axis.var.same_as(value)
                    for axis in frame.iter_vars
                ):
                    raise ValueError(f"Duplicate block axis name {name!r}")
                return _name(value, name, name_span)
            if isinstance(frame, _frame.PrimFuncFrame) and _python.any(
                thread.same_as(value) for thread in frame.env_threads
            ):
                return _name(value, name, name_span)
    with _construction_span(span):
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
        if isinstance(value, _I.meta_var):
            return value.value
        if isinstance(ty, _native.LetAnnotation):
            if value is _MISSING:
                raise ValueError("An immutable binding requires an initializer")
            value = _builder._as_expr(value)
            variable = _name(ty.as_var(rhs_dtype=value.ty), name, name_span)
            _native.Bind(value, var=variable)
            return variable
        if ty is not None:
            annotation = ty() if callable(ty) and not isinstance(ty, _ir.Expr) else ty
            annotation = annotation.ty if isinstance(annotation, _ir.Expr) else annotation
            value = _builder._as_expr(value)
            variable = _ir.Var(name or "", annotation)
            return _name(_native.Bind(value, var=variable), name, name_span)
        if value is _MISSING:
            raise ValueError("An uninitialized binding requires a scalar type annotation")
        if isinstance(value, _NativeFrame):
            return _name(_enter_concise(value), name, name_span)
        if isinstance(value, list | tuple):
            for index, item in enumerate(value):
                bind_(item, name=None if name is None else f"{name}_{index}", span=span)
            return value
        if getattr(type(value), "_is_meta_class", False):
            if name is not None:
                _native.name_meta_class_value(name, value)
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
        return _name(_native.Bind(value), name, name_span)


def emit_(value, *, span=None):
    """Consume an expression statement, including effect-only calls."""
    from tvm.script.ir_builder.base import BypassEmit

    if isinstance(value, _BypassBind):
        # Binding bypass does not imply emission bypass. Consume the declared
        # value normally, including each result of a multi-axis declaration.
        values = value.value if isinstance(value.value, (list, tuple)) else (value.value,)
        for item in values:
            emit_(item, span=span)
        return None
    if isinstance(value, BypassEmit):
        return None
    if value is None or isinstance(value, str | _ir.Var):
        return
    with _construction_span(span):
        if isinstance(value, _NativeFrame):
            _enter_concise(value)
        elif hasattr(value, "frames"):
            for frame in value.frames:
                _enter_concise(frame)
        elif isinstance(value, _tir.Stmt):
            _native.add_to_parent(value)
        else:
            _native.evaluate(value)


def resolve_type_var_(name, dtype=None, *, value=None, span=None):
    """Resolve a symbol using the nearest native primitive-function frame."""
    from tvm.script.ir_builder.base import _current_function_frame

    return _current_function_frame().resolve_type_var(name, dtype, value=value, span=span)


def call_global_var_(function, args):
    """Build a primitive call using the declared callee's exact result type."""
    return _native._call_global(function, *args)


def decl_mutable_var_(value=_MISSING, *, ty=None, name=None, span=None, name_span=None):
    """Name explicit scalar/vector storage and optionally initialize an annotation declaration."""
    name_span = span if name_span is None else name_span
    with _construction_span(span):
        if isinstance(ty, _native.LocalVectorAnnotation):
            if value is not _MISSING:
                raise ValueError("Vector annotation does not support an initializer")
            return _name(_native.alloc_local(ty.shape, ty.dtype), name, name_span)
        if ty is not None:
            annotation = ty() if callable(ty) and not isinstance(ty, _ir.Expr) else ty
            annotation = annotation.ty if isinstance(annotation, _ir.Expr) else annotation
            if not isinstance(annotation, _ir.PrimType) or str(annotation) == "handle":
                raise TypeError("Mutable scalar annotations require a primitive scalar type")
            storage = _native.local_scalar(str(annotation)).scalar
            if value is not _MISSING:
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


def set_mutable_var_(target, value, *, span=None):
    """Store through an explicitly declared handle without rebinding it."""
    with _construction_span(span):
        if isinstance(target, _native.scalar_wrapper):
            target = target.scalar
        if isinstance(target, _ir.TensorLoad):
            _builder.buffer_store(target.source, value, list(target.indices))
        elif (
            _tir.is_buffer_var(target)
            and len(target.ty.shape) == 1
            and isinstance(target.ty.shape[0], _tir.IntImm)
            and target.ty.shape[0].value == 1
        ):
            _builder.buffer_store(target, value, [0])
        else:
            raise TypeError("A mutable assignment requires scalar storage")


def _register_declarations():
    from tvm.script.parser.protocol import register_mutable_var_decl

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
