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
references and results. Ordinary functions can use one ``X.function()`` entry.
When forward references are needed, ``X.function(decl=True)`` reserves each
signature before body entry, allowing sibling calls without another ownership
record. Re-entering that same frame completes its body.
Syntax registration is owned by ``tvm.script.parser.protocol_registry``. Dialects import
its APIs directly; this file documents the complete customization contract but
does not import or re-export those registration APIs. Registration state lives
across parses, while PrescanContext, ModuleContext and FunctionContext only
consume it. Callable flags and records live in parser-owned dictionaries, never
on the registered callables. The argument-policy table shares one immutable
policy between original and adapted callable
identities. No registry stores active frames, construction results or per-parse
environments. Module setup precedes all signatures, which precede all bodies.
Completed results are validated only after frame exit.

A dialect selects source syntax through these registration APIs:

* ``args_policy(fields, scalar_strings=True, dtype=None, as_type=False)`` marks
  individual parameters as ``expr_str`` or ``global_info``. Positional and
  keyword arguments use the same signature-based policy. Expression strings
  lower through ``X.resolve_type_var_``; global-info strings use
  ``I.resolve_global_info_``. Unmarked strings remain literal. The optional
  opaque dtype reaches symbol resolution unchanged. Concrete arguments still
  execute the constructor; eager expression/type adapters retain the builder's
  MissingType and active-function behavior. Unknown policies or parameter names
  are rejected at registration.
* ``register_type_var_decl`` marks an omitted-value scalar declaration and its
  optional dtype, selecting ``X.resolve_type_var_``. ``register_binding_decl``
  marks an explicit ordinary binding that shadows outer mutable storage and
  still uses ``X.bind_``. ``register_mutable_var_decl`` advertises call,
  annotation or parameter positions for ``X.decl_mutable_var_``; later stores
  use ``X.set_mutable_var_``.
* ``direct_call`` returns the registered callable unchanged. Source calls omit
  automatic binding, emission, result-span attachment and the outer source-call
  wrapper. Callee and arguments still evaluate once in order, with normal child
  rewriting. Scope-ID queries, block-axis producers and explicit ``T.bind`` use
  this same category, preserving producer identity, unpacking and explicit names.
  Direct calls precede stores to outer mutable names; assignment supplies no new
  resource name. The callable owns its effects. This differs from AlreadyEmitted,
  whose wrapped result retains normal source-result handling.
* ``result_span`` declares that the complete effect is represented by the returned
  node or emission receipt. It retains ordinary binding/emission while selecting
  result attachment instead of a source-call context. Register it beside a proven
  concrete producer; opaque helpers that emit unrelated statements still need
  their context. ``is_result_span`` reads this marker without evaluating properties.
* ``register_result_members`` associates a producer or annotation with an
  opaque namespace of actual member callables/descriptors. Prescan facts and
  chained producer syntax can expose those members without evaluating a value
  or inspecting its IR type. Each member still needs its own declaration or
  direct-call registration; a method name alone never selects a policy.
* ``register_function`` records an opaque builder namespace, copied option
  mapping, defaults and an optional Python-body flag for a source decorator.
  ``function_info`` reads that ``FunctionDecoratorInfo`` record. The flag keeps
  original Python callables in module ``__pyfuncs__``; other functions lower
  through the registered dialect's function/signature/body hooks.

``get_args_policy`` reads ``ArgsPolicy`` and its ``ExprStrPolicy`` record;
``handle_call_args_policy`` selects that policy before source argument traversal.
String decoding and physical source-range mapping live in ``parser.expr_str_handling``;
general expression rewriting remains in the main transpiler.
``get_type_var_decl`` reads ``DeclarationArguments``. ``is_binding_decl``,
``is_mutable_var_decl`` and ``is_direct_call``
read declaration/call markers. ``copy_function_info`` shares the exact registered
decorator record with a source function; ``register_function_options`` and
``get_function_options`` retain explicitly applied options. ``module_decorator``
and ``is_module_decorator`` classify module decorators, while
``register_parameter_dtype`` and ``get_parameter_dtype`` record parameter dtype
syntax. ``get_result_members`` reads member namespaces. Every registration and
lookup normalizes Python bound methods and property getters without evaluating
properties. ``DeclarationArguments`` carries omitted-value parameter and dtype
facts. These readers neither call constructors nor own native state. The shared
``constexpr`` marker is recognized by identity for host control expressions and
specialization annotations; dialect semantic exports refer to that same marker.

All callable syntax facts live in parser-owned dictionaries, with no callable
attribute writes or fallback reads. Decorators preserve identity, including built-in,
extension and slotted callables. Explicit registrations own their callable and
policy for registry lifetime, including namespaces that reference their own
decorator. Copied source-function information and applied options instead use
separate weak storage so parsing does not permanently retain each source function
or its closure. Tables store syntax metadata only, never active parse scopes,
native frames or constructed results.

A dialect registers policies beside the owning definitions. Generated constructors
receive metadata when created; native callables that cannot use a decorator receive
an explicit registration beside their exposure. Do not scan a completed namespace or
keep a separate inventory of concrete dialect registrations. For example:

.. code:: python

    from tvm.script.parser.protocol_registry import (
        args_policy,
        direct_call,
        register_binding_decl,
        register_function,
        register_mutable_var_decl,
        register_result_members,
        register_type_var_decl,
    )

    @register_binding_decl
    def alloc_buffer(shape, dtype):
        return make_buffer(shape, dtype)

    @direct_call
    def thread_id():
        return current_thread_variable()

    # At a generated dtype-constructor creation site:
    int32 = make_dtype_constructor("int32")
    register_type_var_decl(int32, dtype="int32")

    @args_policy({"shape": "expr_str", "device": "global_info"})
    def tensor(shape, device=None):
        return make_tensor(shape, device)

    @direct_call
    def identity(value):
        return value

    # Source
    value = tensor(("n",), device="cuda:0")
    kept = identity(value)
    # Generated builder
    value = X.bind_(
        tensor((X.resolve_type_var_("n"),), device=I.resolve_global_info_("cuda:0")),
        name="value",
    )
    kept = identity(value)

Importing the registration module does not initialize parser entry points or
concrete builders. Applying an expression/type argument policy requests the
shared eager annotation adapter when needed; native frames still own all IR
construction, symbol identity and validation. Definitions and detailed API
docstrings remain beside the parser-owned implementation.

Shared source support remains in ir_builder.base and is exported through I:
I.at_(location, value) annotates that same object (or AlreadyEmitted.value),
records a returned frame's deferred location, and preserves identity;
I.with_at_group_(location, thunk) evaluates one source call inside restored
caller/definition provenance and applies ``at_`` before returning. ``I.annotation_value_``
converts a rewritten annotation without creating a parser owner. Generated programs
use a table of ``SpanEntry`` objects created while rewriting: ``_S[i](value)``
attaches a materialized range, ``_S[i].ctx(thunk)`` evaluates under that range,
and known protocol operations accept ``span=_S[i]``. Entries retain fixed source
metadata only; dynamic caller ancestry is composed at invocation. Explicit locations
also accept Span objects or (SourceName, start_line, end_line, start_column, end_column)
tuples; None omits explicit attribution. Track-span disabling omits source
instrumentation. Registered direct_call syntax is the explicit exception: it
omits automatic binding, emission and result location handling, while arguments
still follow their own rewriting rules. meta_var is a direct identity call.
Module aliases use shared lexical assignment and retain the active module.

The dialect namespace advertises ``supports_mutable_declarations`` so a
primitive annotation imported from another dialect does not grant mutable
storage syntax to a dialect whose bindings are immutable.

For example, source functions can share a symbolic shape spelling while each
function retains its own symbol identity:

.. code:: python

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

.. code:: python

    with IRBuilder() as builder:
        with I.ir_module():
            first_reference = I.reserve_function("first")
            second_reference = I.reserve_function("second")
            with X.function(decl=True) as first:
                X.func_name("first")
                X.arg("a", X.Buffer((X.resolve_type_var_("n"),), "float32"))
            with X.function(decl=True) as second:
                X.func_name("second")
                X.arg("a", X.Buffer((X.resolve_type_var_("n"),), "float32"))
            with first:
                def first_body():
                    a = first.params[0]
                    n = X.resolve_type_var_("n", dtype="int64")
                    X.emit_(X.call_global_var_(second.reference, [a]))
                first_body()
            with second:
                def second_body():
                    a = second.params[0]
                    X.emit_(X.evaluate(a[0]))
                second_body()
        result = builder.get()
    I.check_well_formed_(result)

The function stubs below document dialect hooks, implemented in each dialect's
``builder.parser_protocol`` and exported through ``X``. They introduce no base class or
runtime dispatcher. Shared operations are implemented here; dialect stubs raise NotImplementedError.
They specify a structural contract, without a base class or runtime dispatcher.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from tvm import ir as _ir

from ..base import MISSING, IRBuilder, IRBuilderFrame, SpanEntry
from .frame import IRModuleFrame

_Span = SpanEntry | _ir.Span | tuple[_ir.SourceName, int, int, int, int] | None


def resolve_global_info_(content: Any) -> Any:
    """Resolve a module-owned selector or retain a concrete object.

    Parameters
    ----------
    content : str or Any
        Original global-info selector or concrete value. "mesh[0]" indexes a named list;
        "cuda:1" selects the second CUDA vdevice; "vdevice:0" selects by absolute index.
        A trailing memory-scope suffix is accepted without changing device selection.

    Returns
    -------
    Any
        The exact registered global-info object, or the unchanged non-string input.

    Notes
    -----
    String lookup requires the nearest active native module frame and creates no metadata.
    Missing context, malformed selectors or unmatched devices raise ValueError; missing map
    entries or out-of-range indices propagate KeyError/IndexError. Non-string values require
    no frame. No source span is attached to an existing metadata object.

    .. code:: python

        # Source
        R.Tensor((n,), "float32", vdevice="cuda:0")
        # Generated builder
        X.Tensor((n,), "float32", vdevice=I.resolve_global_info_("cuda:0"))
    """
    if not isinstance(content, str):
        return content
    if not IRBuilder.is_in_scope():
        raise ValueError("Global-info lookup requires an enclosing module frame")
    for frame in reversed(IRBuilder.current().frames):
        if isinstance(frame, IRModuleFrame):
            break
    else:
        raise ValueError("Global-info lookup requires an enclosing module frame")
    match = re.fullmatch(r"([^\[\]]+)\[(\d+)\]", content)
    if match:
        name, index = match.groups()
        return frame.global_infos[name][int(index)]
    selector = re.fullmatch(r"([^:\[\]]+)(?::(\d+)(?::([^:]+))?)?", content)
    if selector is None:
        raise ValueError(f"Invalid global-info reference: {content!r}")
    target, index, _scope = selector.groups()
    ordinal = int(index) if index is not None else 0
    devices = frame.global_infos.get("vdevice", ())
    if target == "vdevice":
        return devices[ordinal]
    for device in devices:
        if device.target.kind.name == target:
            if ordinal == 0:
                return device
            ordinal -= 1
    raise ValueError(f"Global-info device reference was not found: {content!r}")


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
    raise NotImplementedError


def bind_(
    value: Any = MISSING,
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
    raise NotImplementedError


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
    raise NotImplementedError


def decl_mutable_var_(
    value: Any = MISSING,
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
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.
    name_span : SpanEntry, Span or source-location tuple, optional
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
    raise NotImplementedError


def set_mutable_var_(target: Any, value: Any, *, span: _Span = None) -> None:
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
    raise NotImplementedError


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
    raise NotImplementedError


def module_member_(name: str, value: Any) -> Any:
    """Register a concrete function member or retain ordinary class setup.

    Parameters
    ----------
    name : str
        Source class member identifier.
    value : Any
        Already-evaluated class member value; a BaseFunc is declared and defined as a
        module function.

    Returns
    -------
    Any
        The reserved GlobalVar for a concrete BaseFunc, otherwise the exact original
        value.

    Notes
    -----
    Concrete function registration requires an active module frame and follows native
    duplicate/type checks. Other values need no frame and cause no IR emission or renaming.
    Shared source class setup runs before function signatures so module attributes/global
    info are available.

    .. code:: python

        # Source
        class Module:
            helper = existing_function
        # Generated builder, inside I.ir_module()
        helper = I.module_member_("helper", existing_function)
    """
    from tvm.ir import BaseFunc

    from .ir import decl_function, def_function

    if isinstance(value, BaseFunc):
        reference = decl_function(name, value)
        def_function(name, value)
        return reference
    return value


def check_well_formed_(module: _ir.IRModule) -> None:
    """Validate a completed module after every body has been finalized.

    Parameters
    ----------
    module : IRModule
        Completed native module, including all resolved forward references and mixed-
        dialect members.

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
        @I.ir_module
        class Module:
            pass
        # Generated builder, after module frame exit
        I.check_well_formed_(module)
    """
    from tvm import relax, s_tir, tirx

    message = (
        "Program is not well-formed. If this is deliberate, set "
        "check_well_formed=False in the top-level decorator."
    )
    if not relax.analysis.check_well_formed(module):
        raise ValueError(message)
    try:
        s_tir.analysis.verify_well_formed(module)
        for function in module.functions.values():
            if isinstance(function, tirx.PrimFunc) and not function.attrs.get("s_tir", False):
                tirx.analysis.verify_tirx_well_formed(function)
    except Exception as error:
        raise ValueError(f"{message}\n{error}") from error


def function(*, decl: bool = False, span: _Span = None, **options: Any) -> IRBuilderFrame:
    """Create the native function frame used for signature and body construction.

    Parameters
    ----------
    decl : bool, optional
        False (default) constructs a complete function on one entry. True collects a
        signature on the first entry and retains this frame for body re-entry.
    span : SpanEntry, Span or source-location tuple, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.
    options : Any
        Dialect options: TIRx private, s_tir and persistent default to False. Relax
        is_pure defaults to True, is_private/local to False; local=True requires a
        declared reference when building its body.

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
    raise NotImplementedError


def arg(name: str, annotation: Any, *, span: _Span = None) -> _ir.Var:
    """Add a parameter to the active native function signature.

    Parameters
    ----------
    name : str
        Source parameter name.
    annotation : Type, Var, Buffer or callable
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    span : SpanEntry, Span or source-location tuple, optional
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
    raise NotImplementedError


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
    raise NotImplementedError


def break_(*, span: _Span = None) -> None:
    """Emit break for the enclosing dialect loop.

    Parameters
    ----------
    span : SpanEntry, Span or source-location tuple, optional
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
    raise NotImplementedError


def continue_(*, span: _Span = None) -> None:
    """Emit continue for the enclosing dialect loop.

    Parameters
    ----------
    span : SpanEntry, Span or source-location tuple, optional
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
    raise NotImplementedError


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
    raise NotImplementedError


def if_(condition: Any, *, span: _Span = None) -> IRBuilderFrame:
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
    raise NotImplementedError


def Then(*, span: _Span = None) -> IRBuilderFrame:
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
    raise NotImplementedError


def Else(*, span: _Span = None) -> IRBuilderFrame:
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
    raise NotImplementedError


def for_(
    iterable: Any, *, names: str | Sequence[str] | None = None, span: _Span = None
) -> IRBuilderFrame:
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
    IRBuilderFrame
        The same configured ForFrame. Entry returns the native variable for one
        dimension, or the original variable sequence otherwise.

    Notes
    -----
    TIRx requires an active primitive function before entry. Variables already have final
    names at entry. Simple scalar targets use the entry result; generated tuple,
    list or starred targets use the stable frame.vars sequence for unpacking.
    Invalid iterable/names raise TypeError, ValueError or native errors. Relax rejects imperative
    loops. A frame stores its location before deferred body finalization.

    .. code:: python

        # Source
        for i in range(n):
            T.evaluate(i)
        # Generated builder
        with X.for_(X.range_(n), names=("i",)) as i:
            X.emit_(X.evaluate(i))
    """
    raise NotImplementedError


def While(condition: Any, *, span: _Span = None) -> IRBuilderFrame:
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
    raise NotImplementedError


def range_(*args: Any, annotations: dict[str, Any] | None = None) -> IRBuilderFrame:
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
