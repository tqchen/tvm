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
"""Parser-owned syntax policies and function-decorator registration.

This module stores only host callable identities and flat syntax metadata.
Concrete eager annotation behavior is delegated to a builder-owned adapter;
no IR definition, concrete annotation result, symbol or frame is stored here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from inspect import signature
from types import MappingProxyType, MethodType
from typing import Any, NamedTuple, NoReturn, TypeVar

_Callable = TypeVar("_Callable", bound=Callable[..., Any])


def constexpr(value: object) -> NoReturn:
    """Mark a host control value in source syntax; consumed by the transpiler.

    The same callable may annotate a JIT specialization parameter. Generated
    programs never call this marker: they evaluate its operand as Python.
    """
    raise TypeError("constexpr is a parser syntax marker, not a runtime operation")


class ExprStrPolicy(NamedTuple):
    """Immutable syntax policy for registered constructor arguments.

    Parameters
    ----------
    fields : tuple of str
        Parameter names whose string values represent expressions.
    dtype : object, optional
        Dtype spelling passed to builder symbol resolution. Default is None.
    scalar_strings : bool, optional
        Interpret bare strings as expressions. Default is True. Nested
        strings in marked fields are always treated as expressions.

    Notes
    -----
    This tuple record lives with its registered callable across transpilation
    passes. It stores no symbols, eager results, or function-local state.
    Dtype spellings are forwarded as builder arguments; the transpiler never
    constructs or validates symbols. Builders retain already-declared symbol
    identity and type. Construction enters no frame.
    """

    fields: tuple[str, ...]
    dtype: object = None
    scalar_strings: bool = True


# Process-wide registry: callable identity -> immutable syntax policy. Dialect
# imports register once; aliases share identities. No per-function entries or
# evaluation results are cached, and transpilers only read this table.
_ARGS_POLICIES: dict[object, ArgsPolicy] = {}


class ArgsPolicy(NamedTuple):
    """Immutable per-parameter syntax policies and eager expression metadata."""

    fields: Mapping[str, str]
    expression: ExprStrPolicy


def get_args_policy(constructor: object) -> ArgsPolicy | None:
    """Read a registered policy without evaluating the constructor.

    Unregistered and unhashable host values have no argument policy.
    """
    if isinstance(constructor, MethodType):
        constructor = constructor.__func__
    try:
        return _ARGS_POLICIES.get(constructor)
    except TypeError:
        return None


def args_policy(
    fields: Mapping[str, str],
    *,
    scalar_strings: bool = True,
    dtype: object = None,
    as_type: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register ``expr_str`` and ``global_info`` policies by parameter name.

    Expression strings are rewritten as expressions; global-info arguments
    become shared builder lookup calls with their original content. Unmarked
    parameters, including dtype and placement, keep literal strings. Registered
    mappings are copied and immutable. Both positional and keyword arguments
    use the same parameter policy.

    ``scalar_strings=False`` preserves a bare string in an expression field
    (such as ``Tensor("float32")``), while nested shape strings are expressions.
    Builders own the eager expression adapter: unresolved annotations outside
    builder scope return MissingType, and concrete arguments construct normally.
    ``dtype`` is opaque syntax metadata forwarded to builder symbol resolution;
    its default of None leaves the dtype choice to the builder.
    ``as_type=True`` preserves the Python annotation-class surface, including
    Python type unions; construction still returns the wrapped callable's result.

    Both original and wrapped callable identities share the registered policy,
    including bound methods. Concrete arguments call the original constructor.
    Outside a builder, marked strings and ``typing.TypeVar`` values return
    MissingType. Inside a builder, TypeVar values resolve through the current
    function frame and unresolved expression strings raise TypeError. Calls and
    annotation results are never cached; annotations must be safe to re-evaluate.

    Raises
    ------
    ValueError
        If a policy kind or parameter name is unknown, or the constructor's
        signature cannot be inspected.
    TypeError
        If the constructor cannot be inspected, a wrapped call cannot bind its
        signature, or expression strings remain unresolved inside a builder.

    Examples
    --------
    Register shape expressions and module-owned device references::

        @args_policy({"shape": "expr_str", "vdevice": "global_info"},
                     scalar_strings=False)
        def tensor(shape=None, dtype=None, vdevice=None):
            ...
    """
    fields = dict(fields)
    unsupported = set(fields.values()).difference(("expr_str", "global_info"))
    if unsupported:
        raise ValueError(f"Unknown argument policies: {sorted(unsupported)}")

    def decorate(constructor: Callable[..., Any]) -> Callable[..., Any]:
        call_signature = signature(constructor)
        unknown = set(fields).difference(call_signature.parameters)
        if unknown:
            raise ValueError(f"Unknown argument policy fields: {sorted(unknown)}")
        expression_fields = tuple(name for name, kind in fields.items() if kind == "expr_str")
        expression = ExprStrPolicy(expression_fields, dtype, bool(scalar_strings))
        if expression_fields or as_type:
            # Builders own eager construction and active-frame/MissingType decisions.
            from tvm.script.ir_builder.base import wrap_expression_constructor

            result = wrap_expression_constructor(
                constructor, call_signature, expression, as_type=as_type
            )
        else:
            result = constructor
        policy = ArgsPolicy(MappingProxyType(fields.copy()), expression)
        _ARGS_POLICIES[constructor] = policy
        _ARGS_POLICIES[result] = policy
        return result

    return decorate


class DeclarationArguments(NamedTuple):
    """Immutable metadata for scalar-constructor predeclarations.

    Parameters
    ----------
    value_parameter : str
        Optional value parameter whose absence denotes a declaration.
    dtype : object, optional
        Explicit primitive dtype metadata. Default is None.

    Notes
    -----
    `register_type_var_decl` attaches this record to its callable. Aliases
    and transpilation passes share it for the callable's lifetime; it stores
    no symbols or construction results and enters no builder frame.
    """

    value_parameter: str
    dtype: object = None


def register_type_var_decl(
    constructor: _Callable, *, value_parameter: str = "expr", dtype: object = None
) -> _Callable:
    """Register a constructor that can declare a type variable.

    Parameters
    ----------
    constructor : callable
        Scalar constructor supporting attribute assignment.
    value_parameter : str, optional
        Optional value parameter. Default is ``"expr"``; omission of this
        parameter identifies declaration syntax.
    dtype : object, optional
        Explicit primitive dtype metadata. Default is None. A string dtype
        also permits predeclaration of direct zero-argument body calls.

    Returns
    -------
    callable
        The original constructor with attached `DeclarationArguments`.

    Raises
    ------
    AttributeError
        If the constructor cannot store the registration attribute.

    Notes
    -----
    The signature prepass reserves explicitly declared parameter types before
    resolving earlier shape strings. The shared name prescan also recognizes
    direct unconditional zero-argument body calls when dtype is a string,
    emitting the spelling as builder predeclaration data. It excludes nested
    control scopes and argument-bearing or effectful expressions. Original
    calls and bindings retain their body order.

    Registration persists with the callable. The constructor is neither
    wrapped nor evaluated, and no builder frame is entered.
    """
    constructor.__tvm_type_var_decl__ = DeclarationArguments(value_parameter, dtype)
    return constructor


def register_binding_decl(constructor: _Callable) -> _Callable:
    """Mark a call that explicitly introduces an ordinary source binding.

    Its scalar or unpacked targets bind the returned values even when an outer
    declaration uses the same name for mutable storage. Builders still own the
    returned values and their identity; this flag only selects assignment syntax.
    Callable aliases share the metadata, without a separate registry.
    """
    constructor.__tvm_binding_decl__ = True
    return constructor


def register_mutable_var_decl(constructor: _Callable, *, syntax: str = "call") -> _Callable:
    """Register mutable storage in call, annotation or parameter position.

    The immutable syntax set belongs to the callable across translations. It
    describes source forms only; builders own the resulting storage and stores.
    """
    if syntax not in ("call", "annotation", "parameter"):
        raise ValueError("Mutable declaration syntax must be call, annotation or parameter")
    kinds: frozenset[str] = getattr(constructor, "__tvm_mutable_var_decl__", frozenset())
    constructor.__tvm_mutable_var_decl__ = kinds | frozenset((syntax,))
    return constructor


def is_mutable_var_decl(constructor: object, *, syntax: str) -> bool:
    """Read a registered mutable declaration without evaluating source values."""
    return syntax in getattr(constructor, "__tvm_mutable_var_decl__", ())


class FunctionDecoratorInfo(NamedTuple):
    """Flat syntax registration for a source function decorator.

    Parameters
    ----------
    builder : object or None
        Opaque construction namespace, or None for ordinary Python functions.
    option_map : dict of str to str, optional
        Public option names mapped to builder option names. Default is None,
        interpreted as empty. `register_function` copies supplied mappings.
    defaults : dict of str to object, optional
        Default builder keyword values. Default is None, interpreted as
        empty. `register_function` copies supplied mappings.
    python : bool, optional
        Preserve the source body as ordinary Python. Default is False.

    Notes
    -----
    These are the complete supported fields; no generic metadata dictionary
    is retained. Records live with decorators across compilations. Consumers
    must treat mappings as read-only. Records retain no frame, function
    result, annotation result, or symbol state and enter no builder frames.
    """

    builder: object
    option_map: dict[str, str] | None = None
    defaults: dict[str, Any] | None = None
    python: bool = False


def register_function(
    decorator: _Callable,
    builder: object,
    *,
    option_map: Mapping[str, str] | None = None,
    defaults: Mapping[str, Any] | None = None,
    python: bool = False,
) -> _Callable:
    """Register a decorator with explicit supported syntax options.

    Parameters
    ----------
    decorator : callable
        Function decorator supporting attribute assignment.
    builder : object or None
        Opaque construction namespace, or None when ``python=True``.
    option_map : mapping of str to str, optional
        Public option names mapped to builder keyword names. Default is None,
        interpreted as an empty mapping. Entries are copied.
    defaults : mapping of str to object, optional
        Default builder keyword values. Default is None, interpreted as an
        empty mapping. Entries are copied.
    python : bool, optional
        Preserve the source body as ordinary Python. Default is False.

    Returns
    -------
    callable
        The original decorator with attached `FunctionDecoratorInfo`.

    Raises
    ------
    AttributeError
        If the decorator cannot store the registration attribute.
    TypeError or ValueError
        If supplied option mappings cannot be converted to dictionaries.

    Notes
    -----
    Registration replaces ``__tvm_function_info__`` for the callable's
    lifetime. It executes no constructor or annotation and enters no frame.
    Unknown keyword arguments are rejected by the explicit signature.
    """
    decorator.__tvm_function_info__ = FunctionDecoratorInfo(
        builder, dict(option_map or {}), dict(defaults or {}), bool(python)
    )
    return decorator


def function_info(decorator: object) -> FunctionDecoratorInfo | None:
    """Read registered construction metadata without calling the decorator.

    The returned option mappings are shared registration state. This lookup
    enters no builder frame and propagates custom attribute-access errors.
    """
    return getattr(decorator, "__tvm_function_info__", None)
