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
and transpilation read the resulting metadata. The argument-policy table lives
across parses; other markers live on their callables. Aliases and bound methods
share the existing callable identities. Records contain syntax facts and opaque
builder namespaces, never active frames, constructed results or parse contexts.

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
from inspect import getattr_static, signature
from types import MappingProxyType, MethodType
from typing import Any, NamedTuple, NoReturn, TypeVar

_Callable = TypeVar("_Callable", bound=Callable[..., Any])


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
    constructor.__tvm_result_span__ = True
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
    Bound methods share their underlying function's marker. Only the literal
    boolean True declares the contract; descriptors and unregistered values do not.
    This lookup creates no IR, enters no frame and retains no construction state.

    .. code:: python

        # Source: make_node(x)
        # Rewrite-time classification:
        if is_result_span(make_node):
            # Generated: _S[i](make_node(x))
            pass
    """
    if isinstance(constructor, MethodType):
        constructor = constructor.__func__
    return getattr_static(constructor, "__tvm_result_span__", False) is True


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


# Process-wide registry: callable identity -> immutable syntax policy. Dialect
# imports register once; aliases share identities. No per-function entries or
# evaluation results are cached, and transpilers only read this table.
_ARGS_POLICIES: dict[object, ArgsPolicy] = {}


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
        Shared immutable policy, or None for unregistered/unhashable values.

    Notes
    -----
    No builder context, IR construction or source-span changes occur. The returned policy
    belongs to process-wide registration, not a particular parse. Custom hashing errors
    other than TypeError propagate.

    .. code:: python

        from tvm.script.parser.protocol_registry import get_args_policy

        # Source
        tensor(("n",))
        # Rewrite-time lookup; generated code receives decoded arguments
        policy = get_args_policy(tensor)
        tensor((X.resolve_type_var_("n"),))
    """
    if isinstance(constructor, MethodType):
        constructor = constructor.__func__
    try:
        return _ARGS_POLICIES.get(constructor)
    except TypeError:
        return None


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
    """Mark a constructor whose omitted value argument declares a symbolic variable.

    Parameters
    ----------
    constructor : callable
        Constructor supporting syntax attribute assignment. Identity is retained.
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
    forms are collected, not nested/effectful expressions. Aliases share callable metadata;
    unsupported attribute assignment raises AttributeError.

    .. code:: python

        from tvm.script.parser.protocol_registry import register_type_var_decl

        # Source
        register_type_var_decl(T.int32, dtype="int32")
        # Source
        n = T.int32()
        # Generated builder
        n = X.resolve_type_var_("n", dtype="int32")
    """
    constructor.__tvm_type_var_decl__ = DeclarationArguments(value_parameter, dtype)
    return constructor


def register_binding_decl(constructor: _Callable) -> _Callable:
    """Mark a call that explicitly introduces an ordinary source binding.

    Parameters
    ----------
    constructor : callable
        Producer supporting syntax attribute assignment.

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
    constructor.__tvm_binding_decl__ = True
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
        The attached record, or None when no declaration is registered.

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
    if isinstance(constructor, MethodType):
        constructor = constructor.__func__
    declaration = getattr_static(constructor, "__tvm_type_var_decl__", None)
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
    if isinstance(constructor, MethodType):
        constructor = constructor.__func__
    return getattr_static(constructor, "__tvm_binding_decl__", False) is True


def register_mutable_var_decl(constructor: _Callable, *, syntax: str = "call") -> _Callable:
    """Advertise explicit mutable storage declaration syntax.

    Parameters
    ----------
    constructor : callable
        Storage/annotation constructor supporting attribute assignment.
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
    reads only syntax category. Unknown syntax raises ValueError and unsupported attribute
    assignment raises AttributeError. Declaration patterns take precedence over a prior
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
    kinds: frozenset[str] = getattr(constructor, "__tvm_mutable_var_decl__", frozenset())
    constructor.__tvm_mutable_var_decl__ = kinds | frozenset((syntax,))
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
    if isinstance(constructor, MethodType):
        constructor = constructor.__func__
    kinds = getattr_static(constructor, "__tvm_mutable_var_decl__", frozenset())
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
        Function decorator supporting attribute assignment; retained unchanged.
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
    replaces callable metadata for its lifetime; mappings are read-only by convention.
    Unsupported attributes or invalid mappings raise AttributeError/TypeError/ValueError.
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
    decorator.__tvm_function_info__ = FunctionDecoratorInfo(
        builder, dict(option_map or {}), dict(defaults or {}), bool(python)
    )
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
        Shared flat record, or None for an unregistered decorator.

    Notes
    -----
    No context, construction effect or source attachment. Option mappings belong to
    registration and must not be mutated by a parse. Static lookup does not evaluate
    properties; unknown decorators are diagnosed by the parser consumer.

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
    if isinstance(decorator, MethodType):
        decorator = decorator.__func__
    info = getattr_static(decorator, "__tvm_function_info__", None)
    return info if isinstance(info, FunctionDecoratorInfo) else None


def copy_function_info(source: Callable[..., Any], target: Callable[..., Any]) -> None:
    """Attach an existing decorator registration to its source function.

    Parameters
    ----------
    source : callable
        Registered decorator whose function metadata is required.
    target : callable
        Source function that will share the exact registration record.

    Returns
    -------
    None
        The target is updated in place without constructing another record.

    Notes
    -----
    Missing registration or unsupported attribute assignment raises AttributeError.
    The source attribute is read once and assigned unchanged. Definition scope,
    applied decorator options and native construction remain entry-point concerns;
    no parse environment or result is retained by this operation.

    .. code:: python

        from tvm.script.parser.protocol_registry import copy_function_info

        # Source
        @prim_func
        def f():
            pass
        # Decorator preparation before the generated builder executes
        copy_function_info(prim_func, f)
    """
    target.__tvm_function_info__ = source.__tvm_function_info__


def register_scope_var_query_or_decl(constructor: _Callable) -> _Callable:
    """Mark a call that already owns its scope variable or immutable declaration.

    Parameters
    ----------
    constructor : callable
        Scope query/declaration producer, for example a block-axis constructor, scope-ID
        helper or T.bind.

    Returns
    -------
    _Callable
        The exact producer with scope_var_query_or_decl syntax metadata.

    Notes
    -----
    Registration neither executes the producer nor stores an IR result/frame. The RHS
    executes once; generated assignment preserves returned identities and delegates source
    naming/validation to the dialect hook. This category precedes ordinary binding or a
    store to an outer mutable name. Callable attribute errors propagate.

    .. code:: python

        from tvm.script.parser.protocol_registry import register_scope_var_query_or_decl

        # Source
        register_scope_var_query_or_decl(T.bind)
        # Source
        x = T.bind(value)
        # Generated builder
        x = X.scope_var_query_or_decl_(X.bind(value), name="x")
    """
    constructor.__tvm_scope_var_query_or_decl__ = True
    return constructor


def is_scope_var_query_or_decl(constructor: object) -> bool:
    """Read scope-query/declaration metadata from a resolved callable.

    Parameters
    ----------
    constructor : object
        Resolved producer or bound method; bound methods read the underlying function.

    Returns
    -------
    bool
        True for the registered syntax category, otherwise False.

    Notes
    -----
    No builder context/evaluation or span handling occurs. This is syntax recognition, not a
    runtime variable-type test. Static lookup does not evaluate properties.

    .. code:: python

        from tvm.script.parser.protocol_registry import is_scope_var_query_or_decl

        # Source
        x = T.bind(value)
        # Rewrite-time decision and generated operation
        is_scope_var_query_or_decl(T.bind)
        x = X.scope_var_query_or_decl_(X.bind(value), name="x")
    """
    if isinstance(constructor, MethodType):
        constructor = constructor.__func__
    return getattr_static(constructor, "__tvm_scope_var_query_or_decl__", False) is True


def direct_call(constructor: _Callable) -> _Callable:
    """Mark a call that owns its result without automatic source-result handling.

    Parameters
    ----------
    constructor : callable
        Producer, method or class supporting attribute assignment. Callable identity and
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
    retains ordinary result-span attachment. Attribute errors propagate.

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
    constructor.__tvm_direct_call__ = True
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
    if isinstance(constructor, MethodType):
        constructor = constructor.__func__
    return getattr_static(constructor, "__tvm_direct_call__", False) is True


def register_result_members(constructor: _Callable, members: object) -> _Callable:
    """Advertise a producer's static member namespace for syntax recognition.

    Parameters
    ----------
    constructor : callable
        Registered annotation or value producer. Bound methods attach metadata
        to their underlying function; the supplied callable is returned unchanged.
    members : object
        Opaque namespace of actual member callables/descriptors. It describes
        syntax available on the produced value, without describing an IR type.

    Returns
    -------
    _Callable
        The original producer with its member-namespace attribute.

    Notes
    -----
    No constructor executes and no result, frame or per-parse binding is stored.
    Parameter annotation and unambiguous declaration syntax can supply this
    namespace from existing prescan facts. A registered member producer may
    supply another namespace for chained views. Unknown or ambiguous aliases
    retain their ordinary policy; spelling alone never marks a method direct.
    Attribute-assignment errors propagate. Registration performs no source
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
    target = constructor.__func__ if isinstance(constructor, MethodType) else constructor
    target.__tvm_result_members__ = members
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
    inspects no constructed IR result. Metadata has callable lifetime; no
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
    if isinstance(constructor, property):
        constructor = constructor.fget
    if isinstance(constructor, MethodType):
        constructor = constructor.__func__
    return getattr_static(constructor, "__tvm_result_members__", None)
