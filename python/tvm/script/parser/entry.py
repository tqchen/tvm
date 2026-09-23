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
import __future__

import ast
import copy
import dis
import inspect
import linecache
import sys
import textwrap
from collections.abc import Callable, Mapping
from functools import wraps
from types import FrameType, FunctionType
from typing import TYPE_CHECKING, Any, TypeVar

from tvm.error import DiagnosticError
from tvm.ir import SourceName
from tvm.script.ir_builder import base
from tvm.script.ir_builder import ir as builder_ir

from . import jit_support
from . import protocol as syntax_protocol
from .diagnostics import diagnostic_error
from .prescan import PrescanCollector
from .transpile import IRBuilderTranspiler

if TYPE_CHECKING:
    from tvm.ir import IRModule, Span
    from tvm.relax import ExternFunc
    from tvm.relax.base_py_module import BasePyModule
    from tvm.runtime import Device
    from tvm.target import Target


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


def _closure_values(function: FunctionType) -> dict[str, Any]:
    values = {}
    for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
        try:
            values[name] = cell.cell_contents
        except ValueError:
            # Recursive and later-bound locals are empty until the helper is used.
            pass
    return values


def _lexical_environment(obj: FunctionType | type) -> dict[str, Any]:
    """Retain Python globals and closure bindings without inspecting callers."""
    if inspect.isfunction(obj):
        return {**obj.__globals__, **_closure_values(obj)}
    module = inspect.getmodule(obj)
    return dict(vars(module)) if module is not None else {}


def _definition_scope(frame: FrameType) -> dict[str, Any]:
    """Snapshot immediate locals and active enclosing Python function scopes.

    Postponed annotations do not necessarily create closure cells. Retain their
    active lexical ancestors only when each caller directly owns the child's
    code object. An unrelated caller ends this chain, even in the same file.
    Class locals belong only to the immediate definition context; enclosing
    classes are not Python lexical scopes. No frame survives this snapshot.
    """
    scopes = [dict(frame.f_locals)]
    while frame.f_back is not None:
        parent = frame.f_back
        if parent.f_globals is not frame.f_globals or not any(
            constant is frame.f_code for constant in parent.f_code.co_consts
        ):
            break
        if parent.f_code.co_flags & inspect.CO_NEWLOCALS:
            scopes.append(dict(parent.f_locals))
        frame = parent
    return {name: value for scope in reversed(scopes) for name, value in scope.items()}


def recompose_builder(
    translated: ast.Module,
    *,
    source_fn: str | FunctionType | type,
    definition_scope: Mapping[str, Any],
    filename: str,
    flags: int,
    name: str,
    environment: Mapping[str, Any],
    result: str | None = None,
    definition_scopes_name: str | None = None,
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
    scopes = {
        key: {
            **_closure_values(function),
            **definition_scope,
            **getattr(function, "__tvm_definition_scope__", {}),
        }
        for key, function in originals.items()
    }
    if definition_scopes_name is not None:
        namespace[definition_scopes_name] = scopes

    reserved = set(namespace)
    reserved.update(node.id for node in ast.walk(translated) if isinstance(node, ast.Name))
    reserved.update(node.arg for node in ast.walk(translated) if isinstance(node, ast.arg))
    for body in ast.walk(translated):
        original = originals.get(getattr(body, "_tvm_source_name", None))
        if original is None:
            continue
        retained = body._tvm_signature_names
        parameters = {argument.arg for argument in body.args.args}
        captures = {argument.arg for argument in body.args.kwonlyargs}
        # co_names also contains attribute spellings. Only actual global
        # instructions describe a Python global binding; an attribute may have
        # the same spelling as a captured closure cell (for example C.dtype).
        global_names = {
            instruction.argval
            for instruction in dis.get_instructions(original)
            if instruction.opname in ("LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL")
        }
        global_names -= retained | captures | parameters
        if global_names:
            body.body.insert(0, ast.copy_location(ast.Global(sorted(global_names)), body))
        source_closure = _closure_values(original)
        for captured in sorted((captures | source_closure.keys()) - retained - parameters):
            alias = f"{name}_lexical_{len(namespace)}"
            while alias in reserved:
                alias += "_"
            reserved.add(alias)
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
            namespace[alias] = value
            reference = ast.copy_location(ast.Name(alias, ast.Load()), body)
            if captured in captures:
                index = next(i for i, arg in enumerate(body.args.kwonlyargs) if arg.arg == captured)
                default = body.args.kw_defaults[index]

                class LexicalDefault(ast.NodeTransformer):
                    def visit_Name(self, node: ast.Name) -> ast.Name:
                        return (
                            ast.copy_location(ast.Name(alias, ast.Load()), node)
                            if node.id == captured
                            else node
                        )

                body.args.kw_defaults[index] = LexicalDefault().visit(default)
            else:
                body.args.kwonlyargs.append(ast.arg(captured))
                body.args.kw_defaults.append(reference)

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
    return namespace[name]


def _inside_class(function: FunctionType, frame: FrameType) -> bool:
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

    def resolve(expr: ast.expr) -> object:
        if isinstance(expr, ast.Name):
            return environment.get(expr.id)
        if isinstance(expr, ast.Attribute):
            return getattr(resolve(expr.value), expr.attr, None)
        return None

    from tvm.relax.script.builder.ir import rewriter

    return any(
        resolve(item.func if isinstance(item, ast.Call) else item) in (ir_module, rewriter)
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
    DiagnosticError
        When standalone construction fails during `parse` execution.

    Notes
    -----
    Class members retain their Python functions until module construction.
    Standalone functions immediately transpile and execute a builder program.
    Annotations must be safe to re-evaluate: eager MissingType placeholders
    are not cached, and source annotations execute in declaration frames.
    Registration persists for the lifetime of the returned decorator.
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
                function.__tvm_definition_scope__ = _definition_scope(frame)
                deferred = _inside_class(function, frame)
            finally:
                del frame
            function.__tvm_function_info__ = decorator.__tvm_function_info__
            function.__tvm_function_options__ = options
            if deferred:
                return function
            result = parse(
                function,
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
                function.__tvm_definition_scope__ = _definition_scope(frame)
            finally:
                del frame
            definition_env = _lexical_environment(function)

            @wraps(function)
            def invoke(*args: Any, **kwargs: Any) -> Any:
                bound = inspect.signature(function).bind(*args, **kwargs)
                bound.apply_defaults()
                environment = (
                    {**definition_env, **(_closure_values(function) if late_binding else {})}
                    if options.get("hygienic", True)
                    else {**function.__globals__, **inspect.currentframe().f_back.f_locals}
                )
                return _run_statements(
                    function,
                    builder,
                    {**environment, **bound.arguments},
                    set(bound.arguments),
                    preserve_return=preserve_return,
                )

            invoke.__tvm_construction_helper__ = (builder, options)
            return invoke

        return apply(function) if function is not None else apply

    return decorator


def pyfunc(function: _Callable) -> _Callable:
    """Mark a Python function for opaque registration in a module.

    Parameters
    ----------
    function : callable
        Python function supporting attribute assignment.

    Returns
    -------
    callable
        The same function, with its registration marker attached.

    Raises
    ------
    AttributeError
        If the supplied object does not support the marker attribute.

    Notes
    -----
    The function body remains ordinary Python. Module construction delegates
    registration to the builder runtime. This decorator enters no frame and
    preserves callable identity for the function's lifetime.
    """
    function.__tvm_python_function__ = True
    return function


syntax_protocol.register_function(pyfunc, None, python=True)


def _source_lines(
    source: FunctionType | type, definition_source: tuple[str, int] | None
) -> tuple[list[str], int, str | None]:
    """Recover a class from its exact decoration site when module inspection fails."""
    try:
        lines, start = inspect.getsourcelines(source)
        return lines, start, inspect.getsourcefile(source)
    except OSError:
        if not inspect.isclass(source) or definition_source is None:
            raise
        filename, lineno = definition_source
        lines = linecache.getlines(filename)
        # Gallery runners may execute the class in a temporary __main__ module
        # without __file__. Its decorator still has the original code location.
        tree = ast.parse("".join(lines), filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == source.__name__:
                start = min([node.lineno, *(item.lineno for item in node.decorator_list)])
                if start <= lineno <= node.lineno:
                    return lines[start - 1 : node.end_lineno], start, filename
        raise


def acquire_source(
    source: str | FunctionType | type,
    filename: str | None = None,
    *,
    definition_source: tuple[str, int] | None = None,
) -> tuple[ast.Module, str, int]:
    """Read source into a location-preserving AST, filename and compiler flags.

    Text uses ``<str>`` unless a filename is supplied. Function/class source
    retains its original file, line and UTF-8 column offsets, including the
    decoration-site fallback used by gallery runners. Source inspection and
    parsing errors propagate to the caller before diagnostic conversion.
    The returned AST is the source snapshot; entry copies it once for rewriting.
    """
    members = vars(source).values() if inspect.isclass(source) else (source,)
    flags = 0
    for member in members:
        code = getattr(member, "__code__", None)
        if code is not None:
            flags |= code.co_flags & __future__.annotations.compiler_flag
    if isinstance(source, str):
        text = source
        filename = filename or "<str>"
        start, indent = 1, 0
        linecache.cache[filename] = (len(text), None, text.splitlines(keepends=True), filename)
    else:
        lines, start, source_filename = _source_lines(source, definition_source)
        text = "".join(lines)
        filename = filename or source_filename
        indent = len(lines[0]) - len(lines[0].lstrip())
    tree = ast.parse(textwrap.dedent(text), filename)
    if start != 1:
        ast.increment_lineno(tree, start - 1)
    if indent:
        for node in ast.walk(tree):
            if hasattr(node, "col_offset"):
                node.col_offset += indent
                node.end_col_offset += indent
    return tree, filename, flags


def _prepare_transpiler(
    tree: ast.Module,
    source: str | FunctionType | type,
    environment: Mapping[str, Any],
    definition_scope: Mapping[str, Any],
    filename: str,
    *,
    track_span: bool = True,
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
    metadata_environment = {**namespace, **definition_scope}
    if inspect.isclass(source):
        metadata_environment.update(vars(source))
    # Namespace owners may be modules, classes or user instances with registered
    # methods. Keep their identity for descriptor-safe static policy lookup;
    # transpilation never evaluates or classifies ordinary lexical values.
    metadata = dict(metadata_environment)
    # Direct decorator application may have no decorator in the inspected AST.
    # Give prescan its registered syntax policy before allocating injected names.
    if inspect.isfunction(source):
        tree.body[-1]._tvm_function_info = syntax_protocol.function_info(source)
    prescan = PrescanCollector(metadata, filename=filename).collect(tree)
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
        namespace[decorator_name] = metadata[decorator_name] = source
        keywords = []
        for key, value in getattr(source, "__tvm_function_options__", {}).items():
            option_name = fresh()
            namespace[option_name] = value
            keywords.append(ast.keyword(key, ast.Name(option_name, ast.Load())))
        root = tree.body[-1]
        root.decorator_list = [
            ast.copy_location(ast.Call(ast.Name(decorator_name, ast.Load()), [], keywords), root)
        ]
    builder_name, infrastructure_name = fresh("_X"), fresh("_I")
    definition_scopes_name = fresh("_definition_scopes")
    namespace[infrastructure_name] = builder_ir
    source_name = fresh("_source") if track_span else None
    if track_span:
        # One shared SourceName is metadata, not an IR construction result.
        namespace[source_name] = SourceName(filename)

    def span(node: ast.AST) -> ast.expr:
        """Retain source ranges for builders to materialize during execution."""
        if not track_span:
            return ast.copy_location(ast.Constant(None), node)
        location = ast.Tuple(
            [
                ast.Name(source_name, ast.Load()),
                *[
                    ast.Constant(value)
                    for value in (
                        node.lineno,
                        node.end_lineno,
                        node.col_offset + 1,
                        node.end_col_offset + 1,
                    )
                ],
            ],
            ast.Load(),
        )
        return ast.copy_location(location, node)

    transformer = IRBuilderTranspiler(
        filename,
        metadata,
        builder_name,
        infrastructure_name,
        span,
        fresh,
        track_span=track_span,
        definition_scopes_name=definition_scopes_name,
        prescan=prescan,
        bindings=namespace,
        **options,
    )
    return transformer, namespace


def _run_statements(
    source: FunctionType,
    builder: object,
    environment: Mapping[str, Any],
    bound_names: set[str],
    *,
    preserve_return: bool = False,
) -> Any:
    """Execute a macro body in its caller's active builder frames.

    Argument binding precedes this call. The helper owns one source AST copy,
    keeps Python parameter names and optionally keeps ordinary Python returns.
    Compilation uses the original coordinates without unparse/reparse. Builder
    and host exceptions propagate unchanged to the caller.
    """
    tree, filename, flags = acquire_source(source)
    tree = copy.deepcopy(tree)
    definition_scope = getattr(source, "__tvm_definition_scope__", {})
    transformer, namespace = _prepare_transpiler(
        tree,
        source,
        environment,
        definition_scope,
        filename,
        preserve_return=preserve_return,
        current_scope=tree.body[-1],
    )
    namespace[transformer.dialect_prefix] = builder
    node = tree.body[-1]
    statements = transformer.transform_statements(node.body)
    names = sorted(name for name in bound_names if name in namespace)
    helper_name = transformer.fresh("_macro")
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
    runnable = recompose_builder(
        ast.Module([helper], []),
        source_fn=source,
        definition_scope=definition_scope,
        filename=filename,
        flags=flags,
        name=helper_name,
        environment=namespace,
    )
    return runnable(*(namespace[name] for name in names))


def _build(
    tree: ast.Module,
    source: str | FunctionType | type,
    environment: Mapping[str, Any],
    definition_scope: Mapping[str, Any],
    filename: str,
    flags: int,
    *,
    track_span: bool,
) -> Any:
    """Translate an owned AST and execute its direct native builder program.

    Source expressions and annotations execute only in the generated program.
    Recomposition restores the source's body and annotation scopes, then Python
    compilation retains its original file and full AST ranges. Injected bindings
    and name allocation live only for this invocation.
    """
    transformer, namespace = _prepare_transpiler(
        tree, source, environment, definition_scope, filename, track_span=track_span
    )
    transformed, result = transformer.program(tree)
    runnable = recompose_builder(
        transformed,
        source_fn=source,
        definition_scope=definition_scope,
        definition_scopes_name=transformer.definition_scopes_name,
        filename=filename,
        flags=flags,
        name=transformer.fresh("_builder"),
        environment=namespace,
        result=result,
    )
    return runnable()


def make_opaque_function(
    name: str,
    function: Callable[..., Any],
    source: str,
    location: tuple[SourceName, int, int, int, int] | Span | None = None,
) -> ExternFunc:
    """Represent a Python module member without executing its body."""
    from tvm import relax

    return relax.ExternFunc(name, span=base.source_span(location)).with_attrs(
        {
            "is_pyfunc": True,
            "function_type": "python",
            "python_function_name": name,
            "python_source": source,
            "python_packed_func": function,
        }
    )


def parse(
    source: str | FunctionType | type,
    extra_vars: Mapping[str, Any] | None = None,
    *,
    filename: str | None = None,
    track_span: bool = True,
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
    **options
        ``absent_params`` transports the legacy JIT's explicit mapping of
        absent parameter names to None for the root function. Bare constexpr
        annotations consume their captured bindings during builder execution.
        Other options are accepted for entry-point compatibility; construction
        policy comes from registered source decorators.

    Returns
    -------
    object
        Opaque result of the generated builder program.

    Raises
    ------
    DiagnosticError
        If transpilation or host/builder execution fails. An existing
        DiagnosticError is preserved; other execution errors gain original
        source ranges.
    OSError
        If source inspection cannot recover the supplied object's text.
    TypeError
        If the source object cannot be inspected.
    SyntaxError
        If initial source parsing fails before program execution.

    Notes
    -----
    Each call owns one AST copy and a fresh lexical environment. Declaration and
    definition frames are entered only during generated execution. Source
    acquisition errors propagate directly, before diagnostic conversion.
    """
    env = {} if isinstance(source, str) else _lexical_environment(source)
    env.update(extra_vars or {})
    definition_scope = options.pop(
        "_definition_scope", getattr(source, "__tvm_definition_scope__", {})
    )
    tree, filename, flags = acquire_source(
        source, filename, definition_source=options.pop("_definition_source", None)
    )
    owned_tree = copy.deepcopy(tree)
    try:
        root = owned_tree.body[-1]
        root_name = root.name if isinstance(root, ast.FunctionDef) else None
        specialization = options.get("_specialization_bindings")
        if specialization is None and options.get("absent_params") is not None:
            specialization = {}
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
        with jit_support.specialization_context(root_name, specialization):
            with jit_support.absent_parameters(root_name, options.get("absent_params")):
                result = _build(
                    owned_tree,
                    source,
                    env,
                    definition_scope,
                    filename,
                    flags,
                    track_span=track_span,
                )
        if check_well_formed:
            _check_well_formed(result)
        return result
    except DiagnosticError:
        raise
    except Exception as error:
        raise diagnostic_error(error, filename, tree) from error


def _check_well_formed(result: object) -> None:
    """Apply the public entry point's default validation to constructed IR."""
    from tvm import ir, relax, s_tir, tirx

    message = (
        "Program is not well-formed. If this is deliberate, set "
        "check_well_formed=False in the top-level decorator."
    )
    if isinstance(result, ir.IRModule | relax.Function):
        if not relax.analysis.check_well_formed(result):
            raise ValueError(message)
    if not isinstance(result, ir.IRModule | relax.Function | tirx.PrimFunc):
        return
    module = result if isinstance(result, ir.IRModule) else ir.IRModule.from_expr(result)
    try:
        s_tir.analysis.verify_well_formed(module)
        for function in module.functions.values():
            if isinstance(function, tirx.PrimFunc) and not function.attrs.get("s_tir", False):
                tirx.analysis.verify_tirx_well_formed(function)
    except Exception as error:
        raise ValueError(f"{message}\n{error}") from error


class _PyModuleFactory:
    """Keep executable Python attachments on each fresh module instance."""

    def __init__(self, module: IRModule, original_class: type) -> None:
        self.ir_module: IRModule = module
        self.original_class: type = original_class
        self.pyfunc_methods: list[str] = list(getattr(module, "pyfuncs", {}))
        self.__name__: str = original_class.__name__

    def __call__(self, device: Device | None = None, target: Target | None = None) -> BasePyModule:
        from tvm import cpu, ir
        from tvm.relax.base_py_module import BasePyModule

        source = self.ir_module
        instance_module = ir.IRModule(
            source.functions, attrs=source.attrs, global_infos=source.global_infos
        )
        instance = BasePyModule(instance_module, device or cpu(0), target)
        for name in self.pyfunc_methods:
            instance.add_python_function(name, getattr(self.original_class, name))
        return instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self.ir_module, name)


def ir_module(
    module: type | None = None, **options: Any
) -> IRModule | _PyModuleFactory | Callable[[type], IRModule | _PyModuleFactory]:
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
    DiagnosticError
        If module transpilation or builder execution fails.

    Notes
    -----
    Class host bindings are captured before transpilation. Generated execution
    declares all registered signatures before defining their bodies. Source
    acquisition errors and frame lifetime follow `parse`.
    """

    def apply(module: type) -> IRModule | _PyModuleFactory:
        if not inspect.isclass(module):
            raise TypeError(f"Expect a class, but got: {module}")
        frame = inspect.currentframe().f_back
        try:
            if frame.f_code is ir_module.__code__:
                frame = frame.f_back
            definition_scope = _definition_scope(frame)
            definition_source = (frame.f_code.co_filename, frame.f_lineno)
        finally:
            del frame
        result = parse(
            module,
            _definition_scope=definition_scope,
            _definition_source=definition_source,
            **options,
        )
        from tvm.relax.base_py_module import BasePyModule

        if issubclass(module, BasePyModule):
            return _PyModuleFactory(result, module)
        result.__name__ = module.__name__
        return result

    return apply(module) if module is not None else apply


from_source = parse
