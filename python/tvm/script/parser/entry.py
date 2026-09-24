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
"""Source acquisition and declaration/body execution for registered builders."""

from __future__ import annotations

import ast
import dis
import inspect
import linecache
import sys
from collections import ChainMap
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from types import FrameType, FunctionType
from typing import TYPE_CHECKING, Any, TypeVar

from tvm.ir import SourceName, Span
from tvm.script.ir_builder import base
from tvm.script.ir_builder import ir as builder_ir
from tvm.script.ir_builder.base import SpanEntry

from . import jit_support
from . import protocol_registry as syntax_protocol
from .inspect_source import (
    acquire_source,
    capture_annotation_bindings,
    capture_definition_scope,
    capture_lexical_bindings,
)
from .prescan import PrescanCollector, resolve_syntax
from .transpile import FunctionContext, IRBuilderTranspiler, ModuleContext

if TYPE_CHECKING:
    from tvm.ir import IRModule


_Callable = TypeVar("_Callable", bound=Callable[..., Any])

# Executed Python bodies and registered dialects may supply arbitrary host/IR values.
_NAMESPACES: dict[str, object] = {}


def register_namespace(alias: str, namespace: object) -> None:
    """Register a host namespace for source-text entry points.

    Parameters
    ----------
    alias : str
        Python identifier used to refer to the namespace in source text.
    namespace : object
        Module or object bound to ``alias``.

    Returns
    -------
    None

    Notes
    -----
    Registration replaces the process-wide alias entry. Each parse copies this
    table; existing compilations are unaffected. This operation
    enters no builder frame.
    """
    _NAMESPACES[alias] = namespace


def _read_closure_values(function: FunctionType) -> dict[str, Any]:
    values = {}
    for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
        try:
            values[name] = cell.cell_contents
        except ValueError:
            # Recursive and later-bound locals are empty until the helper is used.
            pass
    return values


def _read_lexical_environment(obj: FunctionType | type) -> dict[str, Any]:
    """Retain Python globals and closure bindings without inspecting callers."""
    if inspect.isfunction(obj):
        return {**obj.__globals__, **_read_closure_values(obj)}
    module = inspect.getmodule(obj)
    return dict(vars(module)) if module is not None else {}


def _recompose_builder(
    translated: ast.Module,
    *,
    source_fn: str | FunctionType | type,
    definition_scope: Mapping[str, Any],
    filename: str,
    flags: int,
    name: str,
    fresh: Callable[[str], str],
    environment: Mapping[str, Any],
    result: str | None = None,
    definition_scope_name: str | None = None,
    body_sources: Sequence[tuple[ast.FunctionDef, str, set[str]]] = (),
) -> Callable[..., Any]:
    """Compile one builder callable with source lexical and annotation scopes.

    Python code objects identify the source's globals and closure cells. The
    generated body keeps those lexical bindings, while annotation expressions
    execute in separate definition-site scopes inside builder declaration frames.
    """
    namespace = dict(environment)
    originals = (
        {key: value for key, value in vars(source_fn).items() if inspect.isfunction(value)}
        if inspect.isclass(source_fn)
        else {source_fn.__name__: source_fn}
        if inspect.isfunction(source_fn)
        else {}
    )
    if definition_scope_name is not None:
        namespace[definition_scope_name] = definition_scope

    for body, source_name, retained in body_sources:
        original = originals.get(source_name)
        if original is None:
            continue
        parameters = {argument.arg for argument in body.args.args}
        # co_names also contains attribute spellings. Only actual global
        # instructions describe a Python global binding; an attribute may have
        # the same spelling as a captured closure cell (for example C.dtype).
        global_names = {
            instruction.argval
            for instruction in dis.get_instructions(original)
            if instruction.opname in ("LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL")
        }
        global_names -= retained | parameters
        source_closure = _read_closure_values(original)
        for captured in sorted(source_closure.keys() - retained - parameters):
            value = (
                environment.get(captured, source_closure[captured])
                if captured in source_closure and inspect.isfunction(source_fn)
                else source_closure.get(captured, environment.get(captured, base.MISSING))
            )
            # Builtin defaults remain ordinary Python lookup when no explicit
            # lexical binding exists. Missing global names are never supplied by
            # the annotation definition scope.
            if value is base.MISSING:
                import builtins

                value = getattr(builtins, captured, base.MISSING)
            # Standalone lexical inputs already have their original names in the
            # execution environment. A global declaration prevents generated
            # enclosing scopes from redirecting those body references.
            if captured in namespace and namespace[captured] is value:
                global_names.add(captured)
                continue
            # A class method may close over a different value than its class's
            # same-named member. Only that concrete conflict needs an injected
            # binding; the body itself still reads the original source name.
            alias = fresh("_lexical")
            namespace[alias] = value
            reference = ast.copy_location(ast.Name(alias, ast.Load()), body)
            body.args.kwonlyargs.append(ast.arg(captured))
            body.args.kw_defaults.append(reference)
        if global_names:
            body.body.insert(0, ast.copy_location(ast.Global(sorted(global_names)), body))

    if result is not None:
        location = translated.body[-1]
        definition = ast.copy_location(
            ast.FunctionDef(
                name,
                ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                [
                    *translated.body,
                    ast.copy_location(ast.Return(ast.Name(result, ast.Load())), location),
                ],
                [],
                None,
            ),
            location,
        )
        if "type_params" in ast.FunctionDef._fields:
            definition.type_params = []
        translated = ast.Module([definition], [])
    exec(
        compile(
            ast.fix_missing_locations(translated), filename, "exec", flags=flags, dont_inherit=True
        ),
        namespace,
    )
    # Pop the generated callable: its globals must never own it in return.
    return namespace.pop(name)


def _is_inside_class(function: FunctionType, frame: FrameType) -> bool:
    """Defer only in the exact class frame of a registered module decorator."""
    local = frame.f_locals
    if local.get("__module__") != function.__module__ or "__qualname__" not in local:
        return False
    text = "".join(linecache.getlines(frame.f_code.co_filename))
    if not text:
        return False
    classes = [
        node
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.ClassDef)
        and node.name == frame.f_code.co_name
        and node.lineno <= frame.f_lineno <= node.end_lineno
    ]
    if not classes:
        return False
    node = min(classes, key=lambda item: item.end_lineno - item.lineno)
    environment = dict(frame.f_globals)
    if frame.f_back is not None:
        environment.update(frame.f_back.f_locals)

    return any(
        syntax_protocol.is_module_decorator(
            resolve_syntax(item.func if isinstance(item, ast.Call) else item, environment)
        )
        for item in node.decorator_list
    )


def make_decorator(
    builder: object,
    *,
    option_map: Mapping[str, str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> Callable[..., Any]:
    """Create and register a function decorator for a construction namespace.

    Parameters
    ----------
    builder : object
        Namespace implementing the function construction protocol.
    option_map : mapping of str to str, optional
        Public option names mapped to builder keyword names. Default is None,
        interpreted as an empty mapping; supplied entries are copied.
    defaults : mapping of str to object, optional
        Default builder keyword values. Default is None, interpreted as an
        empty mapping; supplied entries are copied.

    Returns
    -------
    decorator : callable
        Callable supporting ``@decorator``, ``@decorator(**options)``, and
        ``decorator(function)``.

    Raises
    ------
    ValueError
        When the returned decorator receives a non-function positional value.
    SyntaxError
        When standalone source violates a parser restriction.

    Notes
    -----
    Class members retain their Python functions until module construction.
    Standalone functions immediately transpile and execute a builder program.
    Annotations must be safe to re-evaluate: eager MissingType placeholders
    are not cached, and source annotations execute in declaration frames.
    Construction errors propagate unchanged through `parse`. Registration
    persists for the lifetime of the returned decorator.
    """
    mapping, default_options = dict(option_map or {}), dict(defaults or {})

    def decorator(function: FunctionType | None = None, **options: Any) -> Any:
        """Parse a Python function into a function of the selected IR dialect.

        Parameters
        ----------
        function : Callable, optional
            The function to be parsed. May be omitted to use the decorator with
            keyword options, such as ``@T.prim_func(private=True)``.
        private : bool, optional
            Whether the function should be treated as private. A private
            function has no global symbol attribute; a public function has a
            global symbol matching its name. Defaults to False.
        check_well_formed : bool, optional
            Whether to check that the constructed function is well formed.
            Defaults to True.
        **options
            Additional dialect options. ``T.prim_func`` accepts ``s_tir`` and
            ``persistent``; ``R.function`` accepts ``pure``.

        Returns
        -------
        result : PrimFunc or relax.Function or Callable
            The parsed function, or a decorator when ``function`` is omitted.
            Class members retain their Python functions until the enclosing
            module is constructed.
        """
        if function is not None and not inspect.isfunction(function):
            raise ValueError("Construction decorators require a function or keyword options")

        def apply(function: FunctionType) -> Any:
            frame = inspect.currentframe().f_back
            try:
                if frame.f_code is decorator.__code__:
                    frame = frame.f_back
                deferred = _is_inside_class(function, frame)
                # The module root supplies one scope for all deferred members.
                definition_scope = {} if deferred else capture_definition_scope(frame)
            finally:
                del frame
            syntax_protocol.copy_function_info(decorator, function)
            syntax_protocol.register_function_options(function, options)
            if deferred:
                return function
            result = parse(
                function,
                definition_scope=definition_scope,
                check_well_formed=options.get("check_well_formed", True),
            )
            result.__name__ = function.__name__
            return result

        return apply(function) if function is not None else apply

    return syntax_protocol.register_function(
        decorator, builder, option_map=mapping, defaults=default_options
    )


def make_macro_decorator(
    builder: object, *, preserve_return: bool = True, late_binding: bool = False
) -> Callable[..., Callable[..., Any]]:
    """Create a decorator for helpers executed in a caller's builder frames.

    Parameters
    ----------
    builder : object
        Namespace implementing construction operations for the helper body.
    preserve_return : bool, optional
        Keep helper returns as ordinary Python control flow. Default is True.
    late_binding : bool, optional
        Refresh captured closure cells on each call. Default is False.

    Returns
    -------
    decorator : callable
        Accepts a function directly or keyword options. The ``hygienic``
        option defaults to True and snapshots the definition environment;
        False captures the calling environment on each invocation. Other
        options remain metadata for the namespace consumer.

    Raises
    ------
    ValueError
        When the returned decorator receives a non-function positional value.
    TypeError
        When a helper invocation cannot bind its Python signature.

    Notes
    -----
    Each invocation binds arguments and defaults, transpiles the original
    body, and returns its result. It shares the caller's active construction
    frames instead of declaring an IR function. Source acquisition,
    compilation, and builder exceptions propagate to the caller.
    """

    def decorator(function: FunctionType | None = None, **options: Any) -> Callable[..., Any]:
        """Decorate a helper that constructs IR in its caller's active frames.

        Parameters
        ----------
        function : Callable, optional
            The helper function. May be omitted to supply keyword options.
        hygienic : bool, optional
            Whether the helper resolves symbols in its definition environment
            instead of its calling environment. Defaults to True. ``T.macro``
            and ``R.macro`` capture values at definition time; ``T.inline``
            refreshes captured closure cells when called.

        Returns
        -------
        result : Callable
            The construction helper, or a decorator when ``function`` is omitted.

        Notes
        -----
        ``T.inline`` follows Python lexical scoping with late binding of captured
        closure cells. Its return statements produce Python values, as do those
        of ``R.macro``. ``T.macro`` emits returns in the active primitive function.

        Examples
        --------
        An inline helper can read values from its enclosing scope::

            import tvm
            from tvm.script import tirx as T

            x_value = 128

            @T.inline
            def capture(A, B):
                B[()] = A[x_value]  # x_value resolved from enclosing scope

            @T.prim_func(s_tir=True)
            def use(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
                capture(A, B)       # Produces B[()] = A[128]
        """
        if function is not None and not inspect.isfunction(function):
            raise ValueError("Construction decorators require a function or keyword options")

        def apply(function: FunctionType) -> Callable[..., Any]:
            frame = inspect.currentframe().f_back
            try:
                if frame.f_code is decorator.__code__:
                    frame = frame.f_back
                definition_scope = capture_definition_scope(frame)
            finally:
                del frame
            definition_env = capture_lexical_bindings(function)
            definition_scope = capture_annotation_bindings(function, definition_scope)

            @wraps(function)
            def invoke(*args: Any, **kwargs: Any) -> Any:
                bound = inspect.signature(function).bind(*args, **kwargs)
                bound.apply_defaults()
                environment = (
                    {
                        **function.__globals__,
                        **definition_env,
                        **(_read_closure_values(function) if late_binding else {}),
                    }
                    if options.get("hygienic", True)
                    else {**function.__globals__, **inspect.currentframe().f_back.f_locals}
                )
                return _run_statements(
                    function,
                    builder,
                    {**environment, **bound.arguments},
                    set(bound.arguments),
                    preserve_return=preserve_return,
                    definition_scope=definition_scope,
                )

            return invoke

        return apply(function) if function is not None else apply

    return decorator


def pyfunc(function: _Callable) -> _Callable:
    """Keep an ordinary Python callable for collection in a module.

    Parameters
    ----------
    function : callable
        Python function to retain in the module.

    Returns
    -------
    callable
        The same function, unchanged.

    Notes
    -----
    The function body remains ordinary Python. Shared module parsing attaches
    it to the result's ``__pyfuncs__`` mapping. This decorator enters no frame and
    preserves callable identity for the function's lifetime.
    """
    return function


syntax_protocol.register_function(pyfunc, None, python=True)


def _prepare_transpiler(
    tree: ast.Module,
    source: str | FunctionType | type,
    environment: Mapping[str, Any],
    definition_scope: Mapping[str, Any],
    filename: str,
    *,
    track_span: bool = True,
    specialize: bool = False,
    **options: Any,
) -> tuple[IRBuilderTranspiler, dict[str, Any]]:
    """Prescan an owned tree and inject collision-free execution bindings.

    The lexical environment is copied per invocation. Descriptor-safe metadata
    lookup retains namespace owners, including annotation-only definition
    bindings; ordinary body values remain opaque. Prescan facts are read-only, while the
    local name map allocates fresh identifiers across the complete source unit.
    No builder frame or expression is created here.
    """
    namespace = {
        "TypeVar": TypeVar,
        "tvm": sys.modules.get("tvm"),
        **_NAMESPACES,
        **environment,
    }
    # Imports establish source-text namespace metadata before prescan. Their
    # original AST nodes remain owned here and execute only once.
    imports: list[ast.stmt] = [
        node for node in tree.body[:-1] if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    if imports:
        exec(compile(ast.Module(imports, []), filename, "exec", dont_inherit=True), namespace)
    # Fixed namespace lookup references the one root definition scope directly.
    # A class namespace has precedence, matching its source definition context.
    metadata = ChainMap(
        vars(source) if inspect.isclass(source) else {}, definition_scope, namespace
    )
    # Direct decorator application may have no decorator in the inspected AST.
    # Give prescan its registered syntax policy before allocating injected names.
    root_info = syntax_protocol.function_info(source) if inspect.isfunction(source) else None
    prescan = PrescanCollector(metadata, filename=filename).collect(
        tree, root_function_info=root_info
    )
    names = dict.fromkeys([*namespace, *prescan.reserved_names], 0)

    def fresh(prefix: str = "_t") -> str:
        """Allocate a name without changing any source identifier."""
        counter = names.get(prefix, 0)
        while f"{prefix}{counter}" in names:
            counter += 1
        name = f"{prefix}{counter}"
        names[prefix], names[name] = counter + 1, 0
        return name

    # A directly applied decorator has no registered decorator syntax in its
    # original function. Inject its metadata and opaque option bindings only.
    if inspect.isfunction(source) and syntax_protocol.function_info(source) is not None:
        decorator_name = fresh()
        namespace[decorator_name] = source
        keywords = []
        for key, value in syntax_protocol.get_function_options(source).items():
            option_name = fresh()
            namespace[option_name] = value
            keywords.append(ast.keyword(key, ast.Name(option_name, ast.Load())))
        root = tree.body[-1]
        root.decorator_list = [
            ast.copy_location(ast.Call(ast.Name(decorator_name, ast.Load()), [], keywords), root)
        ]
    builder_name, infrastructure_name = fresh("_X"), fresh("_I")
    definition_scope_name = fresh("_definition_scope")
    namespace[infrastructure_name] = builder_ir
    span_table_name = fresh("_S") if track_span else None
    if track_span:
        # Entries contain fixed native metadata only. The existing rewrite creates
        # them on demand; there is no location collection pass or retained AST.
        source_name = SourceName(filename)
        span_entries: list[SpanEntry] = []
        span_indices: dict[tuple[int, int, int, int], int] = {}
        namespace[span_table_name] = span_entries

    def span(node: ast.AST) -> ast.expr:
        """Materialize a needed location and emit its injected table reference."""
        if not track_span:
            return ast.copy_location(ast.Constant(None), node)
        coordinates = (
            node.lineno,
            node.end_lineno,
            node.col_offset + 1,
            node.end_col_offset + 1,
        )
        index = span_indices.get(coordinates)
        if index is None:
            index = len(span_entries)
            span_indices[coordinates] = index
            span_entries.append(SpanEntry(Span(source_name, *coordinates)))
        location = ast.Subscript(
            ast.Name(span_table_name, ast.Load()), ast.Constant(index), ast.Load()
        )
        return ast.copy_location(location, node)

    context = ModuleContext(
        filename,
        metadata,
        infrastructure_name,
        span,
        fresh,
        track_span=track_span,
        specialize=specialize,
        definition_scope=definition_scope,
        definition_scope_name=definition_scope_name,
        source_functions=(
            {key: value for key, value in vars(source).items() if inspect.isfunction(value)}
            if inspect.isclass(source)
            else {source.__name__: source}
            if inspect.isfunction(source)
            else {}
        ),
        prescan=prescan,
        bindings=namespace,
    )
    transformer = IRBuilderTranspiler(
        context, FunctionContext(options.pop("current_scope", None), builder_name), **options
    )
    return transformer, namespace


def _run_statements(
    source: FunctionType,
    builder: object,
    environment: Mapping[str, Any],
    bound_names: set[str],
    *,
    preserve_return: bool = False,
    definition_scope: Mapping[str, Any] | None = None,
) -> Any:
    """Execute a macro body in its caller's active builder frames.

    Argument binding precedes this call. The helper owns the freshly acquired source AST,
    keeps Python parameter names and optionally keeps ordinary Python returns.
    Compilation uses the original coordinates without unparse/reparse. Builder
    and host exceptions propagate unchanged to the caller.
    """
    tree, filename, flags = acquire_source(source)
    definition_scope = {} if definition_scope is None else definition_scope
    transformer, namespace = _prepare_transpiler(
        tree,
        source,
        environment,
        definition_scope,
        filename,
        preserve_return=preserve_return,
        current_scope=tree.body[-1],
    )
    namespace[transformer.function.dialect_prefix] = builder
    node = tree.body[-1]
    statements = transformer.transform_statements(node.body)
    names = sorted(name for name in bound_names if name in namespace)
    helper_name = transformer.module.fresh("_macro")
    helper = ast.copy_location(
        ast.FunctionDef(
            helper_name,
            ast.arguments(
                posonlyargs=[],
                args=[ast.arg(name) for name in names],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            statements or [ast.Pass()],
            [],
            None,
        ),
        node,
    )
    if "type_params" in ast.FunctionDef._fields:
        helper.type_params = []
    runnable = _recompose_builder(
        ast.Module([helper], []),
        source_fn=source,
        definition_scope=transformer.module.definition_scope,
        body_sources=transformer.module.body_sources,
        filename=filename,
        flags=flags,
        name=helper_name,
        fresh=transformer.module.fresh,
        environment=namespace,
    )
    return runnable(*(namespace[name] for name in names))


def parse(
    source: str | FunctionType | type,
    extra_vars: Mapping[str, Any] | None = None,
    *,
    filename: str | None = None,
    track_span: bool = True,
    definition_scope: Mapping[str, Any] | None = None,
    **options: Any,
) -> Any:
    """Transpile and execute a source string, Python function, or Python class.

    Parameters
    ----------
    source : str or function or type
        Original source text or inspectable Python object.
    extra_vars : mapping of str to object, optional
        Lexical bindings overriding captured values. Default is None,
        interpreted as an empty mapping.
    filename : str, optional
        Source filename override. Default is None, which uses the inspected
        filename for objects and ``"<str>"`` for text.
    track_span : bool, optional
        Enable shared source metadata and IR location instrumentation.
        Default is True. False retains Python source locations only.
    definition_scope : mapping of str to object, optional
        Temporary definition-site bindings for annotation reconstruction. None
        adds no external scope; parse never inspects its caller for bindings.
    **options
        ``_specialization_bindings`` carries selected constexpr values and
        explicit optional-parameter absence in one mapping. ``check_well_formed``
        controls completed-result validation; construction policy otherwise
        comes from the registered source decorators.

    Returns
    -------
    object
        Opaque result of the generated builder program.

    Raises
    ------
    OSError
        If source inspection cannot recover the supplied object's text.
    TypeError
        If the source object cannot be inspected.
    SyntaxError
        If source parsing fails or transpilation detects a syntax restriction.
        Parser restrictions retain their source filename and range.

    Notes
    -----
    Each call owns its freshly acquired AST and a fresh lexical environment. Declaration and
    definition frames are entered only during generated execution. Source
    acquisition, host and builder errors propagate with their original type,
    identity and traceback. Temporary captures are released even when execution
    fails.
    """
    # - Recover source and explicit lexical/definition inputs.
    # - Collect source syntax facts on this invocation's freshly acquired AST.
    # - Rewrite syntax into a builder program and recompose its lexical bindings.
    # - Execute the private builder immediately and release temporary captures.
    # Definition scope is a per-root input; it never replaces body globals/closures.
    env = {} if isinstance(source, str) else _read_lexical_environment(source)
    env.update(extra_vars or {})
    definition_scope = {} if definition_scope is None else definition_scope
    tree, filename, flags = acquire_source(
        source, filename, definition_source=options.pop("_definition_source", None)
    )
    # Acquisition returns a fresh tree; prescan and rewriting own it directly.
    _builder = None
    definition_scope_name = None
    try:
        root = tree.body[-1]
        root_name = root.name if isinstance(root, ast.FunctionDef) else None
        specialization = options.get("_specialization_bindings")
        check_well_formed = options.get("check_well_formed")
        if check_well_formed is None:
            check_well_formed = True
            for decorator in getattr(root, "decorator_list", ()):
                if isinstance(decorator, ast.Call):
                    for keyword in decorator.keywords:
                        if keyword.arg == "check_well_formed":
                            check_well_formed = eval(
                                compile(ast.Expression(keyword.value), filename, "eval"),
                                {**_NAMESPACES, **env},
                            )
        # Prescan and rewrite consume only the owned syntax and fixed metadata.
        transformer, namespace = _prepare_transpiler(
            tree,
            source,
            env,
            definition_scope,
            filename,
            track_span=track_span,
            specialize=specialization is not None and root_name is not None,
        )
        transformed, result_name = transformer.rewrite_module(
            tree, check_well_formed=check_well_formed
        )
        definition_scope_name = transformer.module.definition_scope_name
        # Recomposition preserves original source ranges and body globals/closures.
        _builder = _recompose_builder(
            transformed,
            source_fn=source,
            definition_scope=transformer.module.definition_scope,
            definition_scope_name=definition_scope_name,
            body_sources=transformer.module.body_sources,
            filename=filename,
            flags=flags,
            name=transformer.module.fresh("_builder"),
            fresh=transformer.module.fresh,
            environment=namespace,
            result=result_name,
        )
        # Important: do not retain _builder. Its globals and closures may keep
        # values from the enclosing scope alive.
        with jit_support.use_specialization(root_name, specialization):
            result = _builder()
        return result
    finally:
        # Release temporary captures on both successful and exceptional exits.
        if _builder is not None and definition_scope_name is not None:
            _builder.__globals__.pop(definition_scope_name, None)
        _builder = None
        definition_scope = None


def ir_module(module: type | None = None, **options: Any) -> IRModule | Callable[[type], IRModule]:
    """Decorate a Python class with two-phase module construction.

    Parameters
    ----------
    module : type, optional
        Class to compile immediately. Default is None, which returns a
        decorator awaiting a class.
    **options
        Keyword arguments forwarded to `parse`.

    Returns
    -------
    object or callable
        Generated module result when a class is supplied, otherwise a class
        decorator.

    Raises
    ------
    SyntaxError
        If module source violates a parser restriction.

    Notes
    -----
    Class host bindings are captured before transpilation. Generated execution
    declares all registered signatures before defining their bodies. Source
    acquisition and builder errors propagate unchanged; frame lifetime follows
    `parse`.
    """

    def apply(module: type) -> IRModule:
        if not inspect.isclass(module):
            raise TypeError(f"Expect a class, but got: {module}")
        frame = inspect.currentframe().f_back
        try:
            if frame.f_code is ir_module.__code__:
                frame = frame.f_back
            definition_scope = capture_definition_scope(frame)
            definition_source = (frame.f_code.co_filename, frame.f_lineno)
        finally:
            del frame
        result = parse(
            module,
            definition_scope=definition_scope,
            _definition_source=definition_source,
            **options,
        )

        result.__name__ = module.__name__
        return result

    return apply(module) if module is not None else apply


syntax_protocol.module_decorator(ir_module)

from_source = parse
