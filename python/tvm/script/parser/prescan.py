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
"""One syntax prescan for generated names, declarations and region outputs."""

import ast
import builtins
import inspect
from dataclasses import dataclass
from types import GetSetDescriptorType, MappingProxyType, MemberDescriptorType

from . import protocol


def resolve_syntax(node, environment):
    """Read fixed namespace metadata without executing source descriptors."""
    if isinstance(node, ast.Name):
        return environment.get(node.id, getattr(builtins, node.id, None))
    if not isinstance(node, ast.Attribute):
        return None
    owner = resolve_syntax(node.value, environment)
    if owner is None:
        return None
    value = inspect.getattr_static(owner, node.attr, None)
    if isinstance(value, staticmethod):
        return value.__func__
    if inspect.isfunction(value):
        if inspect.ismodule(owner) or inspect.isclass(owner):
            return value
        dictionary = inspect.getattr_static(owner, "__dict__", None)
        if isinstance(dictionary, GetSetDescriptorType | MemberDescriptorType):
            if node.attr in dictionary.__get__(owner):
                return value
        return value.__get__(owner)
    return None if hasattr(type(value), "__get__") else value


@dataclass(frozen=True)
class Binding:
    """A source binding site, retained through rewriting without value state."""

    name: str
    node: ast.AST
    kind: str
    annotation: ast.AST | None = None
    dtype: object = None
    direct: bool = False


@dataclass(frozen=True)
class PrescanContext:
    """Read-only syntax facts; nodes refer to the entry-owned AST.

    ``reserved_names`` seeds name allocation for the complete translation.
    ``bindings`` preserves each lexical scope's declaration order. ``sites``
    selects the assignment rewrite for a source target. ``mutable_names`` is a
    derived index of those bindings, not independently updated name state.
    ``conditional_outputs`` supplies the one explicit native-frame result name.
    """

    reserved_names: frozenset
    bindings: object
    sites: object
    mutable_names: object
    conditional_outputs: object
    namespaces: frozenset
    # Explicit source dataflow outputs preserve native export identity on exit.
    with_outputs: object
    # Only source functions referencing themselves need early standalone refs.
    recursive_functions: frozenset


class PrescanCollector(ast.NodeVisitor):
    """Collect binding syntax once, then discard all traversal accumulators."""

    def __init__(self, environment, *, filename="<str>"):
        # Fixed lookup inputs last for this scan; never updated by assignments.
        self.environment = environment
        self.filename = filename
        # Accumulators are frozen by collect(); no native values are stored.
        self.names = set(environment)
        self.bindings = {}
        self.sites = {}
        self.outputs = {}
        self.namespaces = set()
        self.exports = {}
        self.recursive = set()
        # Active source declarations, used only to recognize self references.
        self.functions = []
        self.module_name = None
        # Temporary lexical with-stack routes explicit output calls, then resets.
        self.regions = []
        # Lexical scope and dialect restore on function/class exit. direct marks
        # unconditional function-body declarations eligible before signatures.
        self.scope = None
        self.builder = None
        self.direct = False

    def collect(self, tree):
        """Scan the owned tree and return immutable facts for its rewrite."""
        # Decorator roots establish namespace meaning for this translation.
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    target = decorator.func if isinstance(decorator, ast.Call) else decorator
                    if protocol.function_info(resolve_syntax(target, self.environment)):
                        while isinstance(target, ast.Attribute):
                            target = target.value
                        if isinstance(target, ast.Name):
                            self.namespaces.add(target.id)
        for name, value in self.environment.items():
            if inspect.ismodule(value) and (
                value.__name__.startswith("tvm.script.ir_builder")
                or value.__name__ == "tvm.script.parser.ir"
                or value.__name__.startswith("tvm.tirx.script")
                or value.__name__.startswith("tvm.relax.script")
            ):
                self.namespaces.add(name)
        self.scope = tree
        self.bindings[tree] = []
        self.visit(tree)
        return PrescanContext(
            frozenset(self.names),
            MappingProxyType({scope: tuple(items) for scope, items in self.bindings.items()}),
            MappingProxyType(dict(self.sites)),
            MappingProxyType(
                {
                    scope: frozenset(
                        item.name for item in items if item.kind in ("mutable", "mutable_parameter")
                    )
                    for scope, items in self.bindings.items()
                }
            ),
            MappingProxyType(dict(self.outputs)),
            frozenset(self.namespaces),
            MappingProxyType({node: tuple(names) for node, names in self.exports.items()}),
            frozenset(self.recursive),
        )

    def _error(self, node, message):
        raise SyntaxError(message, (self.filename, node.lineno, node.col_offset + 1, None))

    def _binding(self, name, node, kind="ordinary", annotation=None, dtype=None):
        if name in self.namespaces:
            self._error(node, f"Script namespace {name!r} cannot be rebound or shadowed")
        self.names.add(name)
        item = Binding(name, node, kind, annotation, dtype, self.direct)
        self.bindings[self.scope].append(item)
        self.sites[node] = item

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store) and node.id in self.namespaces:
            self._error(node, f"Script namespace {node.id!r} cannot be rebound or shadowed")
        self.names.add(node.id)
        if isinstance(node.ctx, ast.Load):
            for function in reversed(self.functions):
                if node.id == function.name:
                    self.recursive.add(function)
                    break

    def visit_arg(self, node):
        if node.arg in self.namespaces:
            self._error(node, f"Script namespace {node.arg!r} cannot be rebound or shadowed")
        self.names.add(node.arg)
        self.generic_visit(node)

    def visit_alias(self, node):
        name = node.asname or node.name.split(".")[0]
        if isinstance(self.scope, ast.FunctionDef) and name in self.namespaces:
            self._error(node, f"Script namespace {name!r} cannot be rebound or shadowed")
        self.names.add(name)

    def visit_ClassDef(self, node):
        self.names.add(node.name)
        old, old_module = self.scope, self.module_name
        self.scope, self.module_name = node, node.name
        self.bindings[node] = []
        self.generic_visit(node)
        self.scope, self.module_name = old, old_module

    def visit_FunctionDef(self, node):
        self._binding(node.name, node, "function")
        old_scope, old_builder, old_direct = self.scope, self.builder, self.direct
        self.scope, self.direct = node, True
        self.functions.append(node)
        original_kind = getattr(node, "_tvm_function_info", None)
        if original_kind is not None:
            self.builder = original_kind.builder
        self.bindings[node] = []
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            kind = protocol.function_info(resolve_syntax(target, self.environment))
            if kind is not None:
                self.builder = kind.builder
                break
        for parameter in getattr(node, "type_params", ()):
            self._binding(parameter.name, parameter, "symbol", dtype="int64")
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            annotation = arg.annotation
            constructor = resolve_syntax(
                annotation.func if isinstance(annotation, ast.Call) else annotation,
                self.environment,
            )
            dtype = getattr(constructor, "__tvm_parameter_dtype__", None)
            declaration = getattr(constructor, "__tvm_type_var_decl__", None)
            self._binding(
                arg.arg,
                arg,
                "mutable_parameter"
                if getattr(self.builder, "supports_mutable_declarations", True)
                and protocol.is_mutable_var_decl(constructor, syntax="parameter")
                else "parameter",
                arg.annotation,
                (declaration.dtype if declaration is not None else dtype)
                if not isinstance(annotation, ast.Call)
                else None,
            )
        for statement in node.body:
            self.visit(statement)
            if isinstance(statement, ast.Return | ast.Raise):
                self.direct = False
        for decorator in node.decorator_list:
            self.visit(decorator)
        if node.returns:
            self.visit(node.returns)
        self.functions.pop()
        self.scope, self.builder, self.direct = old_scope, old_builder, old_direct

    visit_AsyncFunctionDef = visit_FunctionDef

    def _target(self, target, value=None, annotation=None):
        if isinstance(target, ast.Name):
            constructor = (
                resolve_syntax(value.func, self.environment)
                if isinstance(value, ast.Call)
                else None
            )
            declaration = getattr(constructor, "__tvm_type_var_decl__", None)
            if declaration is not None and not value.args and not value.keywords:
                self._binding(target.id, target, "symbol", value, declaration.dtype)
            elif getattr(self.builder, "supports_mutable_declarations", True) and (
                protocol.is_mutable_var_decl(constructor, syntax="call")
                or (
                    annotation is not None
                    and protocol.is_mutable_var_decl(
                        resolve_syntax(
                            annotation.value
                            if isinstance(annotation, ast.Subscript)
                            else annotation,
                            self.environment,
                        ),
                        syntax="annotation",
                    )
                )
            ):
                self._binding(target.id, target, "mutable", annotation)
            elif isinstance(value, ast.Name) and value.id == self.module_name:
                self._binding(target.id, target, "module_alias")
            else:
                self._binding(target.id, target, annotation=annotation)
        elif isinstance(target, ast.Tuple | ast.List):
            values = (
                value.elts
                if isinstance(value, ast.Tuple | ast.List) and len(value.elts) == len(target.elts)
                else [None] * len(target.elts)
            )
            for child, rhs in zip(target.elts, values):
                self._target(child, rhs)
        elif isinstance(target, ast.Starred):
            self._target(target.value)
        self.visit(target)

    def visit_Assign(self, node):
        for target in node.targets:
            self._target(target, node.value)
        self.visit(node.value)

    def visit_AnnAssign(self, node):
        self._target(node.target, node.value, node.annotation)
        self.visit(node.annotation)
        if node.value:
            self.visit(node.value)

    def visit_If(self, node):
        old_direct, self.direct = self.direct, False
        self.generic_visit(node)
        marker = (
            isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Attribute)
            and node.test.func.attr == "constexpr"
        )
        if not marker and getattr(self.builder, "__tvm_value_if__", False):

            def ending(body):
                last = body[-1] if body else None
                if (
                    isinstance(last, ast.Assign)
                    and len(last.targets) == 1
                    and isinstance(last.targets[0], ast.Name)
                ):
                    return last.targets[0].id
                if isinstance(last, ast.AnnAssign) and isinstance(last.target, ast.Name):
                    return last.target.id
                return self.outputs.get(last)

            then, otherwise = ending(node.body), ending(node.orelse)
            # Effect-only branches have no Python output. Native branch frames
            # still reject a non-void expression used as an effect-only ending.
            effects = bool(
                node.body
                and node.orelse
                and isinstance(node.body[-1], ast.Expr)
                and isinstance(node.orelse[-1], ast.Expr)
            )
            if not effects:
                if then is None or then != otherwise:
                    location = node.orelse[-1] if node.orelse else node
                    self._error(
                        location, "IR conditional branches must end with the same named output"
                    )
                self.outputs[node] = then
        self.direct = old_direct

    def visit_For(self, node):
        old_direct, self.direct = self.direct, False
        self._target(node.target)
        self.visit(node.iter)
        for statement in node.body + node.orelse:
            self.visit(statement)
        self.direct = old_direct

    def visit_While(self, node):
        old_direct, self.direct = self.direct, False
        self.generic_visit(node)
        self.direct = old_direct

    def visit_With(self, node):
        old_direct, self.direct = self.direct, False
        self.regions.append(node)
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self._target(item.optional_vars)
        for statement in node.body:
            self.visit(statement)
        self.regions.pop()
        self.direct = old_direct

    def visit_Call(self, node):
        if self.regions and isinstance(node.func, ast.Attribute) and node.func.attr == "output":
            output = getattr(self.builder, "output", None)
            if output is not None and resolve_syntax(node.func, self.environment) is output:
                names = [arg.id for arg in node.args if isinstance(arg, ast.Name)]
                self.exports.setdefault(self.regions[-1], []).extend(names)
        self.generic_visit(node)
