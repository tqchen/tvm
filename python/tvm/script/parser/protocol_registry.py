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
"""Parser-owned syntax registration, metadata and lookup helpers.

Dialect builders register their callables here, and prescan, argument rewriting
and transpilation read the resulting dictionary entries. Registration leaves
callables unchanged. Aliases, bound methods and property getters share normalized
callable identities. Explicit syntax registrations own their callables and records
for registry lifetime. Copied source-function metadata and applied options use
separate weak storage, releasing temporary functions and closures after parsing.
Records contain syntax facts and opaque builder namespaces, never active frames,
constructed results or parse contexts.

Importing this module needs only the standard library. Registration does not
initialize parser entry points or concrete dialects. Expression-argument policies
request the shared builder's eager annotation adapter only when applied; that
adapter retains native construction and MissingType decisions.

The complete customization overview and generated-operation contract are in
``tvm.script.ir_builder.ir.parser_protocol``. Its documentation shows direct
imports from this module; it does not own or re-export the registry.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from inspect import signature
from types import MappingProxyType, MethodType
from typing import Any, NamedTuple, NoReturn, TypeVar
from weakref import ReferenceType, ref

_Callable = TypeVar("_Callable", bound=Callable[..., Any])

_Metadata = TypeVar("_Metadata")
_Registry = dict[int, tuple[ReferenceType[object] | None, object, _Metadata]]


def _callable_key(target: object) -> object:
    """Normalize descriptors without evaluating a property or producer."""
    if issubclass(type(target), property):
        target = property.fget.__get__(target)
    if type(target) is MethodType:
        target = target.__func__
    return target


def _register(table: _Registry[_Metadata], target: object, value: _Metadata) -> None:
    """Own explicit syntax registrations for the registry lifetime."""
    target = _callable_key(target)
    table[id(target)] = (None, target, value)


def _remember_source(table: _Registry[_Metadata], target: object, value: _Metadata) -> None:
    """Keep copied parsing metadata without owning a temporary source function."""
    target = _callable_key(target)
    key = id(target)

    def discard(reference: ReferenceType[object]) -> None:
        entry = table.get(key)
        if entry is not None and entry[0] is reference:
            del table[key]

    try:
        reference = ref(target, discard)
        owner = None
    except TypeError:
        # Built-ins/extensions may expose neither writable attributes nor weakrefs.
        reference = None
        owner = target
    table[key] = (reference, owner, value)


def _lookup(table: _Registry[_Metadata], target: object) -> _Metadata | None:
    """Read by identity, without invoking user hashing, equality or descriptors."""
    target = _callable_key(target)
    entry = table.get(id(target))
    if entry is None:
        return None
    reference, owner, value = entry
    if reference is not None:
        owner = reference()
    return value if owner is target else None


_RESULT_SPAN: _Registry[bool] = {}
_DIRECT_CALL: _Registry[bool] = {}
_TYPE_VAR_DECL: _Registry[DeclarationArguments] = {}
_BINDING_DECL: _Registry[bool] = {}
_MUTABLE_VAR_DECL: _Registry[frozenset[str]] = {}
_FUNCTION_INFO: _Registry[FunctionDecoratorInfo] = {}
_RESULT_MEMBERS: _Registry[object] = {}
_MODULE_DECORATOR: _Registry[bool] = {}
_PARAMETER_DTYPE: _Registry[str] = {}
# Parsing a source function does not make it a persistent syntax registration.
_COPIED_FUNCTION_INFO: _Registry[FunctionDecoratorInfo] = {}
_FUNCTION_OPTIONS: _Registry[Mapping[str, Any]] = {}


def result_span(constructor: _Callable) -> _Callable:
    """Declare that a callable's complete IR effect is represented by its result.

    Parameters
    ----------
    constructor : callable
        Constructor whose returned node or emission receipt owns all source attribution.
        It must not emit unrelated statements requiring a caller context.

    Returns
    -------
    _Callable
        The unchanged callable with static syntax metadata.

    Notes
    -----
    Register at the concrete definition or creation site after establishing this
    contract. Unlike direct_call, this marker keeps ordinary binding and emission.
    It only permits result attachment instead of a construction context. Arguments
    still receive their own source instrumentation. Registration executes no IR
    construction and stores no per-parse state.

    .. code:: python

        @result_span
        def make_node(value):
            return Node(value)

        # Source: make_node(x)
        # Builder: X.emit_(make_node(x), span=_S[i])
        # Nested expression: _S[i](make_node(x))
    """
    _register(_RESULT_SPAN, constructor, True)
    return constructor


def is_result_span(constructor: object) -> bool:
    """Read result-attribution metadata without evaluating arbitrary properties.

    Parameters
    ----------
    constructor : object
        Resolved callable or bound method; never invoked by this lookup.

    Returns
    -------
    bool
        Whether the callable explicitly declares complete returned-result attribution.

    Notes
    -----
    Bound methods and properties share their underlying function's registration.
    Callable attributes are not consulted; unregistered values retain ordinary handling.
    This lookup creates no IR, enters no frame and retains no construction state.

    .. code:: python

        # Source: make_node(x)
        # Rewrite-time classification:
        if is_result_span(make_node):
            # Generated: _S[i](make_node(x))
            pass
    """
    return _lookup(_RESULT_SPAN, constructor) is True


def constexpr(value: object) -> NoReturn:
    """Mark a host control expression or a JIT specialization annotation.

    Parameters
    ----------
    value : object
        Source operand evaluated as ordinary Python after rewriting; never passed to
        this marker at runtime.

    Returns
    -------
    NoReturn
        The marker is removed from generated execution; direct Python invocation raises
        TypeError.

    Notes
    -----
    No builder context or IR effects. Recognition uses callable identity, including aliases,
    rather than name spelling or return values. The same marker can annotate a
    specialization parameter. Host operators/control flow retain Python behavior while their
    source calls still receive ordinary source-call instrumentation.

    .. code:: python

        # Source
        if I.constexpr(enabled):
            T.evaluate(1)
        # Generated builder
        if enabled:
            X.emit_(X.evaluate(1))
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


# Argument policies use the same normalized identity and lifetime as other syntax facts.
_ARGS_POLICIES: _Registry[ArgsPolicy] = {}


class ArgsPolicy(NamedTuple):
    """Immutable per-parameter policies shared by a callable and its adapter.

    Parameters
    ----------
    fields : Mapping[str, str]
        Immutable parameter-to-policy mapping containing expr_str/global_info.
    expression : ExprStrPolicy
        Expression fields, scalar-string behavior and opaque symbolic dtype.

    Notes
    -----
    This process-wide syntax record stores no evaluated arguments, scopes,
    frames or results. Registration copies its mappings; parses only read it.
    """

    fields: Mapping[str, str]
    expression: ExprStrPolicy


def get_args_policy(constructor: object) -> ArgsPolicy | None:
    """Read argument policy metadata without evaluating a constructor.

    Parameters
    ----------
    constructor : object
        Resolved host callable or bound method; bound methods use their underlying
        function identity.

    Returns
    -------
    ArgsPolicy | None
        Shared immutable policy, or None for unregistered values.

    Notes
    -----
    No builder context, IR construction or source-span changes occur. The returned policy
    belongs to callable registration, not a particular parse. Identity lookup does not
    call user hashing or equality methods.

    .. code:: python

        from tvm.script.parser.protocol_registry import get_args_policy

        # Source
        tensor(("n",))
        # Rewrite-time lookup; generated code receives decoded arguments
        policy = get_args_policy(tensor)
        tensor((X.resolve_type_var_("n"),))
    """
    return _lookup(_ARGS_POLICIES, constructor)


def handle_call_args_policy(
    node: ast.Call, resolve: Callable[[ast.expr], object]
) -> tuple[ArgsPolicy, list[str]] | None:
    """Select source-argument policy before the main visitor traverses children.

    Parameters
    ----------
    node : ast.Call
        Original source call, whose children have not yet been rewritten.
    resolve : Callable[[ast.expr], object]
        Fixed syntax lookup for the callee; does not evaluate source expressions.

    Returns
    -------
    tuple[ArgsPolicy, list[str]] or None
        Registered policy and positional parameter names, or None for an unmatched call.

    Notes
    -----
    Unmatched calls need no normalization. The returned positional names and
    syntax policy guide that same visitor; no generated nodes are inserted into
    an unvisited tree and no per-node provenance markers are needed.
    No builder context, IR construction or per-parse registry state is created.

    .. code:: python

        # Source: tensor(("n",))
        policy = handle_call_args_policy(source_call, resolve_syntax)
        # The existing visitor generates:
        # tensor((X.resolve_type_var_("n"),))
    """
    constructor = resolve(node.func)
    policy = get_args_policy(constructor)
    if policy is None:
        return None
    parameters = [
        parameter.name
        for parameter in signature(constructor).parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    return policy, parameters


def args_policy(
    fields: Mapping[str, str],
    *,
    scalar_strings: bool = True,
    dtype: object = None,
    as_type: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register per-argument source policies without storing construction results.

    Parameters
    ----------
    fields : mapping of str to str
        Parameter names mapped to expr_str or global_info. Copied into immutable
        registration data; positional and keyword calls use the same policy.
    scalar_strings : bool, optional
        True by default. False preserves a bare string in an expression field; nested
        strings are still expression syntax.
    dtype : object, optional
        Opaque dtype passed to symbol resolution for expression strings. None (default)
        leaves the builder default.
    as_type : bool, optional
        False by default. True preserves the annotation-class surface, including Python
        type unions, while construction returns the underlying result.

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        Decorator returning the callable or its eager annotation adapter. Original and
        adapted identities share one argument policy.

    Notes
    -----
    Registration enters no frame, constructs no IR and retains no per-function values.
    Concrete arguments call the original constructor. Outside construction, unresolved
    expression annotations yield MissingType; inside construction, unresolved expression
    strings raise TypeError and TypeVar values use the nearest function resolver. Unknown
    policy kinds/parameter names raise ValueError; signature/argument-binding errors
    propagate. The adapter belongs to the builder, not parser execution.

    .. code:: python

        from tvm.script.parser.protocol_registry import args_policy

        # Source
        @args_policy({"shape": "expr_str", "device": "global_info"})
        def tensor(shape, device=None):
            ...
        # Source
        tensor(("n",), device="cuda:0")
        # Generated builder
        tensor((X.resolve_type_var_("n"),), device=I.resolve_global_info_("cuda:0"))
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
        _register(_ARGS_POLICIES, constructor, policy)
        _register(_ARGS_POLICIES, result, policy)
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
    `register_type_var_decl` stores this record in the parser-owned table. Aliases
    and transpilation passes share it for the registry's lifetime; it stores
    no symbols or construction results and enters no builder frame.
    """

    value_parameter: str
    dtype: object = None


def register_type_var_decl(
    constructor: _Callable, *, value_parameter: str = "expr", dtype: object = None
) -> _Callable:
    """Mark a constructor whose omitted value argument declares a symbolic variable.

    Parameters
    ----------
    constructor : callable
        Constructor whose identity is retained; writable attributes are unnecessary.
    value_parameter : str, optional
        Value parameter whose absence means declaration; defaults to expr.
    dtype : object, optional
        Opaque primitive dtype, default None. A string dtype also allows prescan to
        recognize direct zero-argument body declarations.

    Returns
    -------
    _Callable
        The original callable with DeclarationArguments metadata.

    Notes
    -----
    No constructor executes and no frame is entered. Syntax predeclaration precedes
    dependent signature shapes while original body ordering remains. Only direct declaration
    forms are collected, not nested/effectful expressions. Aliases share the same
    dictionary-owned syntax record; registration leaves callable attributes untouched.

    .. code:: python

        from tvm.script.parser.protocol_registry import register_type_var_decl

        # Source
        register_type_var_decl(T.int32, dtype="int32")
        # Source
        n = T.int32()
        # Generated builder
        n = X.resolve_type_var_("n", dtype="int32")
    """
    _register(_TYPE_VAR_DECL, constructor, DeclarationArguments(value_parameter, dtype))
    return constructor


def register_binding_decl(constructor: _Callable) -> _Callable:
    """Mark a call that explicitly introduces an ordinary source binding.

    Parameters
    ----------
    constructor : callable
        Producer, including built-in or slotted callables without writable attributes.

    Returns
    -------
    _Callable
        The unchanged producer with its declaration marker.

    Notes
    -----
    No evaluation, frame entry or source attachment occurs. Its assignment targets shadow an
    outer mutable declaration instead of storing through it; ordinary dialect ``bind_`` still
    owns result construction. Scalar and unpacked targets follow the same precedence.
    Attribute errors propagate.

    .. code:: python

        from tvm.script.parser.protocol_registry import register_binding_decl

        # Source
        register_binding_decl(make_value)
        # Source, even if an outer x is mutable
        x = make_value()
        # Generated builder
        x = X.bind_(make_value(), name="x")
    """
    _register(_BINDING_DECL, constructor, True)
    return constructor


def get_type_var_decl(constructor: object) -> DeclarationArguments | None:
    """Read scalar-declaration metadata without evaluating the constructor.

    Parameters
    ----------
    constructor : object
        Resolved source constructor or annotation.

    Returns
    -------
    DeclarationArguments or None
        The registered record, or None when no declaration is registered.

    Notes
    -----
    Static metadata lookup preserves callable aliases without evaluating properties.
    No frame, symbol or per-parse state is created.

    .. code:: python

        from tvm.script.parser.protocol_registry import get_type_var_decl

        # Source
        n = X.int32()
        # Rewrite-time lookup and generated builder
        declaration = get_type_var_decl(X.int32)
        n = X.resolve_type_var_("n", dtype=declaration.dtype)
    """
    declaration = _lookup(_TYPE_VAR_DECL, constructor)
    return declaration if isinstance(declaration, DeclarationArguments) else None


def is_binding_decl(constructor: object) -> bool:
    """Read whether a producer explicitly introduces an ordinary binding.

    Parameters
    ----------
    constructor : object
        Resolved source callable; it is not invoked.

    Returns
    -------
    bool
        True only for an explicitly registered binding marker.

    Notes
    -----
    Static lookup never invokes a property or a truth-conversion hook.
    No native value, frame or source location is inspected.

    .. code:: python

        from tvm.script.parser.protocol_registry import is_binding_decl

        # Source
        x = make_value()
        # Rewrite-time decision and generated builder
        is_binding_decl(make_value)
        x = X.bind_(make_value(), name="x")
    """
    return _lookup(_BINDING_DECL, constructor) is True


def register_mutable_var_decl(constructor: _Callable, *, syntax: str = "call") -> _Callable:
    """Advertise explicit mutable storage declaration syntax.

    Parameters
    ----------
    constructor : callable
        Storage/annotation constructor; no writable attribute or slot is required.
    syntax : str, optional
        One of call (default), annotation or parameter. Repeated registration adds to
        its immutable syntax set.

    Returns
    -------
    _Callable
        The same callable with its mutable syntax metadata.

    Notes
    -----
    No value is constructed, cached or inspected. Builders own storage and stores; prescan
    reads only syntax category. Unknown syntax raises ValueError. Registration leaves
    callable attributes untouched. Declaration patterns take precedence over a prior
    target binding.

    .. code:: python

        from tvm.script.parser.protocol_registry import register_mutable_var_decl

        # Source
        register_mutable_var_decl(T.int32, syntax="annotation")
        # Source
        x: T.int32 = 0
        # Generated builder
        x = X.decl_mutable_var_(0, ty=X.int32, name="x")
    """
    if syntax not in ("call", "annotation", "parameter"):
        raise ValueError("Mutable declaration syntax must be call, annotation or parameter")
    kinds = _lookup(_MUTABLE_VAR_DECL, constructor) or frozenset()
    _register(_MUTABLE_VAR_DECL, constructor, kinds | frozenset((syntax,)))
    return constructor


def is_mutable_var_decl(constructor: object, *, syntax: str) -> bool:
    """Read whether a callable advertises a mutable declaration position.

    Parameters
    ----------
    constructor : object
        Resolved host constructor or annotation value; it is never called.
    syntax : str
        Source position to test, normally call, annotation or parameter.

    Returns
    -------
    bool
        True exactly when the position is registered; False for absent metadata.

    Notes
    -----
    No frame, IR effect or source attachment. Static metadata lookup does not evaluate
    properties. It does not inspect constructed storage or infer
    mutability from IR types.

    .. code:: python

        from tvm.script.parser.protocol_registry import is_mutable_var_decl

        # Source
        x = T.local_scalar("int32")
        # Rewrite-time decision and generated operation
        is_mutable_var_decl(T.local_scalar, syntax="call")
        x = X.decl_mutable_var_(X.local_scalar("int32"), name="x")
    """
    kinds = _lookup(_MUTABLE_VAR_DECL, constructor)
    return isinstance(kinds, frozenset) and syntax in kinds


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
    """Register a source decorator and its explicit builder options.

    Parameters
    ----------
    decorator : callable
        Function decorator, retained unchanged without writing attributes.
    builder : object or None
        Opaque dialect builder namespace, or None for ordinary Python functions.
    option_map : mapping of str to str, optional
        Source decorator keyword names mapped to builder keyword names. None (default)
        means empty; values are copied.
    defaults : mapping of str to Any, optional
        Default builder keyword values. None (default) means empty; values are copied.
    python : bool, optional
        False (default) requests script lowering. True preserves the original Python
        callable/body for module.__pyfuncs__.

    Returns
    -------
    _Callable
        The original decorator with its flat FunctionDecoratorInfo record.

    Notes
    -----
    No annotation, function or builder executes and no frame/span is retained. Registration
    owns the callable and its metadata for registry lifetime; mappings are read-only
    by convention. A namespace may intentionally refer back to its decorator.
    Invalid mappings raise TypeError or ValueError; writable attributes are not required.
    These records contain syntax options, never per-parse results.

    .. code:: python

        from tvm.script.parser.protocol_registry import register_function

        # Source
        register_function(prim_func, X, option_map={"private": "private"})
        # Source
        @prim_func(private=True)
        def f():
            pass
        # Generated builder entry
        with X.function(private=True):
            pass
    """
    info = FunctionDecoratorInfo(
        builder, dict(option_map or {}), dict(defaults or {}), bool(python)
    )
    _COPIED_FUNCTION_INFO.pop(id(_callable_key(decorator)), None)
    _register(_FUNCTION_INFO, decorator, info)
    return decorator


def function_info(decorator: object) -> FunctionDecoratorInfo | None:
    """Read registered function-decorator metadata.

    Parameters
    ----------
    decorator : object
        Resolved source decorator; it is not called by this lookup.

    Returns
    -------
    FunctionDecoratorInfo | None
        Shared explicit or copied source record, or None when neither is registered.

    Notes
    -----
    No context, construction effect or source attachment. Option mappings belong to
    registration and must not be mutated by a parse. Static lookup does not evaluate
    properties; unknown decorators are diagnosed by the parser consumer. Copied
    source metadata takes precedence without promoting that source function to a
    persistent syntax registration.

    .. code:: python

        from tvm.script.parser.protocol_registry import function_info

        # Source
        @T.prim_func
        def f():
            pass
        # Rewrite-time lookup and generated builder
        function_info(T.prim_func)
        with X.function():
            pass
    """
    info = _lookup(_COPIED_FUNCTION_INFO, decorator)
    return info if info is not None else _lookup(_FUNCTION_INFO, decorator)


def copy_function_info(source: Callable[..., Any], target: Callable[..., Any]) -> None:
    """Share an existing decorator registration with its source function.

    Parameters
    ----------
    source : callable
        Registered decorator whose function metadata is required.
    target : callable
        Source function that will share the exact registration record.

    Returns
    -------
    None
        The target receives a separate temporary entry sharing the exact syntax record.

    Notes
    -----
    Missing registration raises AttributeError. The record is read once and shared
    unchanged through a weak callable key, so copying does not retain a temporary
    source function or its closure. Applied options use register_function_options;
    definition scope and native construction remain entry-point concerns. No parse
    environment or result is retained by this operation.

    .. code:: python

        from tvm.script.parser.protocol_registry import copy_function_info

        # Source
        @prim_func
        def f():
            pass
        # Decorator preparation before the generated builder executes
        copy_function_info(prim_func, f)
    """
    info = function_info(source)
    if info is None:
        raise AttributeError("Callable has no registered function metadata")
    _remember_source(_COPIED_FUNCTION_INFO, target, info)


def direct_call(constructor: _Callable) -> _Callable:
    """Mark a call that owns its result without automatic source-result handling.

    Parameters
    ----------
    constructor : callable
        Producer, method or class, including immutable callables. Callable identity and
        ordinary return values are retained.

    Returns
    -------
    _Callable
        The same callable; no forwarding callable or result wrapper is created.

    Notes
    -----
    No builder context or constructor evaluation. Registered source calls omit ``bind_``, ``emit_``,
    result ``at_`` and the outer ``with_at_group_`` wrapper. Callee/arguments still evaluate once in
    order, and argument expressions follow their own syntax rules. The callable owns any IR
    effects or source attribution it needs. This is separate from AlreadyEmitted, which
    retains ordinary result-span attachment. Registration writes only parser-owned tables.

    .. code:: python

        from tvm.script.parser.protocol_registry import direct_call

        # Source
        @direct_call
        def identity(value):
            return value
        # Source and generated Python
        x = identity(value)
        identity(value)
    """
    _register(_DIRECT_CALL, constructor, True)
    return constructor


def is_direct_call(constructor: object) -> bool:
    """Read direct-call syntax metadata without invoking the callable.

    Parameters
    ----------
    constructor : object
        Resolved callable, class or bound method; methods read their underlying function
        metadata.

    Returns
    -------
    bool
        True if direct_call is registered, otherwise False.

    Notes
    -----
    No frame, IR effect, naming or span attachment. Static lookup does not evaluate properties.
    Recognition is by resolved callable metadata, never inferred from a constructed return
    type or an arbitrary method spelling.

    .. code:: python

        from tvm.script.parser.protocol_registry import is_direct_call

        # Source
        x = I.meta_var(value)
        # Rewrite-time decision; generated execution remains a direct call
        is_direct_call(I.meta_var)
        x = I.meta_var(value)
    """
    return _lookup(_DIRECT_CALL, constructor) is True


def register_result_members(constructor: _Callable, members: object) -> _Callable:
    """Advertise a producer's static member namespace for syntax recognition.

    Parameters
    ----------
    constructor : callable
        Registered annotation or value producer. Bound methods and properties register
        their underlying function; the supplied target is returned unchanged.
    members : object
        Opaque namespace of actual member callables/descriptors. It describes
        syntax available on the produced value, without describing an IR type.

    Returns
    -------
    _Callable
        The original producer with its dictionary-owned member-namespace registration.

    Notes
    -----
    No constructor executes and no result, frame or per-parse binding is stored.
    Parameter annotation and unambiguous declaration syntax can supply this
    namespace from existing prescan facts. A registered member producer may
    supply another namespace for chained views. Unknown or ambiguous aliases
    retain their ordinary policy; spelling alone never marks a method direct.
    Registration requires no writable attributes and performs no source
    attachment and does not itself mark a call as direct_call.

    .. code:: python

        from tvm.script.parser.protocol_registry import direct_call, register_result_members

        # Registration: BufferMethods.permute is the actual registered method.
        register_result_members(X.Buffer, BufferMethods)
        direct_call(BufferMethods.permute)
        # Source
        view = A.permute(1, 0)  # A has source annotation X.Buffer(...).
        # Generated Python preserves the producer's ordinary result.
        view = A.permute(1, 0)
    """
    _register(_RESULT_MEMBERS, constructor, members)
    return constructor


def get_result_members(constructor: object) -> object | None:
    """Read a producer's opaque syntax member namespace without evaluation.

    Parameters
    ----------
    constructor : object
        Resolved annotation/producer, bound method or property descriptor.
        Methods use their function and properties use their getter metadata.

    Returns
    -------
    object or None
        Registered namespace, or None when static member information is absent.

    Notes
    -----
    This reads syntax metadata only. It enters no frame, calls no producer and
    inspects no constructed IR result. Explicit metadata has registry lifetime; no
    function-local lookup state is cached. Static lookup does not evaluate properties.
    Member lookup still checks the actual member's direct_call or
    declaration metadata; a namespace does not make all its methods direct.

    .. code:: python

        from tvm.script.parser.protocol_registry import get_result_members, is_direct_call

        # Source
        view = A.permute(1, 0)
        # Rewrite-time lookup for A's registered source annotation.
        members = get_result_members(X.Buffer)
        is_direct_call(members.permute)
        # Generated Python
        view = A.permute(1, 0)
    """
    return _lookup(_RESULT_MEMBERS, constructor)


def module_decorator(decorator: _Callable) -> _Callable:
    """Register a module decorator without changing its callable identity.

    Parameters
    ----------
    decorator : callable
        Module decorator, including immutable or slotted callables.

    Returns
    -------
    _Callable
        The identical callable, with dictionary-owned syntax registration.

    Notes
    -----
    Member function decorators use this classification to defer parsing until
    the enclosing module is available. Registration calls no producer, writes
    no callable attributes and retains no parse scope or constructed result.

    .. code:: python

        @module_decorator
        def module(source):
            return parse(source)
        # Source: @module class M: ...
        # Member decorators defer; module(source) owns the shared parse.
    """
    _register(_MODULE_DECORATOR, decorator, True)
    return decorator


def is_module_decorator(decorator: object) -> bool:
    """Read module-decorator syntax without evaluating descriptors.

    Parameters
    ----------
    decorator : object
        Resolved callable, bound method or property getter descriptor.

    Returns
    -------
    bool
        True only for an explicitly registered module decorator.

    Notes
    -----
    Normalized identity lookup does not call the decorator or inspect its
    attributes. Unregistered values retain ordinary function-entry handling.

    .. code:: python

        # Source: @module class M: ...
        if is_module_decorator(module):
            # Defer member construction to the module boundary.
            pass
    """
    return _lookup(_MODULE_DECORATOR, decorator) is True


def register_parameter_dtype(constructor: _Callable, dtype: str) -> _Callable:
    """Register parameter dtype syntax for a constructor.

    Parameters
    ----------
    constructor : callable
        Parameter annotation constructor, returned unchanged.
    dtype : str
        Dtype spelling, or the constructor argument name that supplies it.

    Returns
    -------
    _Callable
        The identical constructor with a dictionary-owned dtype policy.

    Notes
    -----
    Prescan reads this syntax fact without evaluating annotations. Non-string
    metadata raises TypeError; registration constructs no IR, writes no callable
    attributes and stores no evaluated parameters.

    .. code:: python

        register_parameter_dtype(Prim, "dtype")
        # Source: def main(value: Prim("int32")): ...
        # Prescan records the literal dtype for the parameter declaration.
    """
    if not isinstance(dtype, str):
        raise TypeError("Parameter dtype metadata must be a string")
    _register(_PARAMETER_DTYPE, constructor, dtype)
    return constructor


def get_parameter_dtype(constructor: object) -> str | None:
    """Read a registered constructor's parameter dtype syntax.

    Parameters
    ----------
    constructor : object
        Resolved constructor, bound method or property getter descriptor.

    Returns
    -------
    str or None
        Registered dtype spelling/argument name, or None when unregistered.

    Notes
    -----
    Lookup normalizes callable identities and executes no descriptor, annotation
    or constructor. The receiving dialect owns parameter construction.

    .. code:: python

        # Source: def main(value: Prim("int32")): ...
        dtype_parameter = get_parameter_dtype(Prim)
        # Prescan reads the named source argument; no Prim call runs here.
    """
    return _lookup(_PARAMETER_DTYPE, constructor)


def register_function_options(function: _Callable, options: Mapping[str, Any]) -> _Callable:
    """Record applied decorator options without retaining the source function.

    Parameters
    ----------
    function : callable
        Source function receiving the already-applied decorator's syntax options.
    options : Mapping[str, Any]
        Applied keyword options, copied into a read-only mapping.

    Returns
    -------
    _Callable
        The identical function, with no attribute mutation or forwarding wrapper.

    Notes
    -----
    Values are syntax options, never parse scopes, frames or construction results.
    These are temporary source facts, separate from explicit syntax registrations.
    Weak callable ownership permits Python source functions and their closures
    to disappear after parsing, even when cyclic GC is disabled. Non-weak-referenceable
    native or slotted targets use a strong fallback; ordinary Python functions
    require no such fallback. The supplied mapping itself is not retained.

    .. code:: python

        # Source: @prim_func(private=True) def main(): ...
        register_function_options(main, {"private": True})
        # Entry later generates: with X.function(private=True): ...
    """
    _remember_source(_FUNCTION_OPTIONS, function, MappingProxyType(dict(options)))
    return function


def get_function_options(function: object) -> Mapping[str, Any]:
    """Read applied decorator options for a source function.

    Parameters
    ----------
    function : object
        Source function, normalized consistently with registration.

    Returns
    -------
    Mapping[str, Any]
        Registered read-only options, or an empty mapping when unregistered.

    Notes
    -----
    Entry reads these syntax options to reconstruct an explicitly applied
    decorator. Lookup does not evaluate the function, construct IR, attach
    spans or retrieve a captured environment.

    .. code:: python

        # Source: @prim_func(private=True) def main(): ...
        options = get_function_options(main)
        # Generated builder: with X.function(private=True): ...
    """
    options = _lookup(_FUNCTION_OPTIONS, function)
    return options if options is not None else {}
