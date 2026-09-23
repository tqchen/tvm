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
"""Rewrite owned Python syntax into ordinary native-builder programs.

The prescan supplies immutable syntax facts. This visitor owns only translation
inputs, generated-name allocation and a small lexical rewrite context. Native
frames own symbols, declarations, parameters, region results and final IR.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, NoReturn, TypeVar

from . import protocol
from .call_args_policy import handle_call_args_policy, parse_annotation
from .prescan import PrescanContext, resolve_syntax

_Node = TypeVar("_Node", bound=ast.AST)


class IRBuilderTranspiler(ast.NodeTransformer):
    """A single statement/expression visitor over the entry-owned AST.

    ``environment``, ``prescan``, ``filename`` and ``span`` are fixed inputs for
    one translation. ``fresh`` allocates names in the entry-owned name map;
    ``bindings`` receives only injected host namespaces and helper functions.
    ``dialect_prefix``, ``current_scope``, ``host_expression`` and annotation
    substitutions are saved/restored at their lexical visitor boundaries. They
    contain AST names, never native frames, values or construction ownership.
    """

    def __init__(
        self,
        filename: str,
        environment: Mapping[str, object],
        builder_name: str,
        infrastructure_name: str,
        span: Callable[[ast.AST], ast.expr],
        fresh: Callable[[str], str],
        *,
        prescan: PrescanContext | None = None,
        track_span: bool = True,
        definition_scopes_name: str | None = None,
        current_scope: ast.AST | None = None,
        bindings: dict[str, Any] | None = None,
        preserve_return: bool = False,
    ) -> None:
        # Read-only translation inputs; namespace meanings never change in flow.
        self.filename: str = filename
        self.environment: Mapping[str, object] = environment
        self.prescan: PrescanContext | None = prescan
        self.infrastructure_name: str = infrastructure_name
        self.span: Callable[[ast.AST], ast.expr] = span
        self.track_span: bool = track_span
        self.definition_scopes_name: str | None = definition_scopes_name
        # Allocation/injection last for this source unit, including nested code.
        self.fresh: Callable[[str], str] = fresh
        self.bindings: dict[str, Any] = bindings if bindings is not None else {}
        # Lexical syntax context, restored on nested function/host/annotation exit.
        self.dialect_prefix: str = builder_name
        self.current_scope: ast.AST | None = current_scope
        self.host_expression: bool = False
        self.preserve_return: bool = preserve_return
        self.annotation_aliases: dict[str, str] = {}
        # Annotation-only aliases in the current body; reset at function exit.
        self.body_annotation_aliases: dict[str, str] = {}
        self.module_name: str | None = None
        self.module_functions: frozenset[str] = frozenset()

    def _inject(self, value: object, prefix: str = "_host") -> ast.Name:
        name = self.fresh(prefix)
        self.bindings[name] = value
        return ast.Name(name, ast.Load())

    def _error(self, node: ast.AST, message: str) -> NoReturn:
        raise SyntaxError(message, (self.filename, node.lineno, node.col_offset + 1, None))

    def _call(
        self, namespace: str, member: str, args: list[ast.expr], node: ast.AST, **keywords: ast.expr
    ) -> ast.Call:
        """Build a generated operation with its source range and named arguments."""
        if not self.track_span:
            keywords.pop("span", None)
            keywords.pop("name_span", None)
        return ast.copy_location(
            ast.Call(
                ast.Attribute(ast.Name(namespace, ast.Load()), member, ast.Load()),
                args,
                [ast.keyword(key, value) for key, value in keywords.items()],
            ),
            node,
        )

    def _operation(
        self, member: str, args: list[ast.expr], node: ast.AST, **keywords: ast.expr
    ) -> ast.Call:
        return self._call(self.dialect_prefix, member, args, node, span=self.span(node), **keywords)

    def _at(self, value: ast.expr, node: ast.AST) -> ast.expr:
        if not self.track_span:
            return value
        return self._call(self.infrastructure_name, "at_", [self.span(node), value], node)

    @staticmethod
    def _lambda(names: list[str], value: ast.expr) -> ast.Lambda:
        return ast.Lambda(
            ast.arguments(
                posonlyargs=[],
                args=[ast.arg(name) for name in names],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            value,
        )

    @staticmethod
    def _definition(name: str, body: list[ast.stmt], node: ast.AST) -> ast.FunctionDef:
        definition = ast.copy_location(
            ast.FunctionDef(
                name,
                ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body or [ast.Pass()],
                [],
                None,
            ),
            node,
        )
        if "type_params" in ast.FunctionDef._fields:
            definition.type_params = []
        return definition

    def transform_statements(self, body: list[ast.stmt]) -> list[ast.stmt]:
        """Visit source statements once, flattening statement-list rewrites."""
        result = []
        for statement in body:
            rewritten = self.visit(statement)
            if rewritten is not None:
                result.extend(rewritten if isinstance(rewritten, list) else [rewritten])
        return result

    @contextmanager
    def _host(self) -> Iterator[None]:
        # Only constexpr operands and module host syntax keep Python operators.
        # The same visitor still instruments their source calls and restores mode.
        old = self.host_expression
        self.host_expression = True
        try:
            yield
        finally:
            self.host_expression = old

    def _resolve(self, node: ast.AST | None) -> object:
        # Fixed namespace meanings coexist with Python lexical value bindings.
        # A local ``range`` or callable hides the ambient binding for the whole
        # source function, including reads before its assignment.
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        scope = self.current_scope
        if isinstance(root, ast.Name) and self.prescan is not None:
            while isinstance(scope, ast.FunctionDef):
                if any(item.name == root.id for item in self.prescan.bindings.get(scope, ())):
                    return None
                scope = next(
                    (
                        parent
                        for parent, items in self.prescan.bindings.items()
                        if any(item.node is scope and item.kind == "function" for item in items)
                    ),
                    None,
                )
        return resolve_syntax(node, self.environment)

    def _constexpr_operand(self, node: ast.expr) -> ast.expr | None:
        # Source: I.constexpr(expr), X.constexpr(expr)
        # Builder: expr under the visitor's host-expression context.
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return None
        if node.func.attr != "constexpr" or not isinstance(node.func.value, ast.Name):
            return None
        if self.prescan is not None and node.func.value.id not in self.prescan.namespaces:
            return None
        if len(node.args) != 1 or node.keywords or isinstance(node.args[0], ast.Starred):
            self._error(node, "constexpr expects exactly one controlling value")
        return node.args[0]

    def visit_Name(self, node: ast.Name) -> ast.expr:
        # Source: "n" in a marked expression argument
        # Builder: X.resolve_type_var_("n") without a Python binding.
        if hasattr(node, "_tvm_quoted_symbol"):
            keywords = {}
            if node._tvm_quoted_symbol is not None:
                keywords["dtype"] = ast.Constant(node._tvm_quoted_symbol)
            return self._at(
                self._call(
                    self.dialect_prefix,
                    "resolve_type_var_",
                    [ast.Constant(node.id)],
                    node,
                    **keywords,
                ),
                node,
            )
        # Source: real annotation n; Builder: hygienic definition-context alias.
        if isinstance(node.ctx, ast.Load) and node.id in self.annotation_aliases:
            return self._call(
                self.infrastructure_name,
                "require_defined",
                [
                    self._call(
                        self.infrastructure_name,
                        "annotation_value_",
                        [
                            ast.Constant(node.id),
                            ast.Name(self.annotation_aliases[node.id], ast.Load()),
                        ],
                        node,
                    ),
                    ast.Constant(node.id),
                ],
                node,
            )
        return (
            self._at(node, node)
            if isinstance(node.ctx, ast.Load) and not self.host_expression
            else node
        )

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        # A normalized protocol callee is already builder syntax, with no source
        # child expressions to transform or locations to invent.
        if getattr(node, "_tvm_intrinsic", False):
            return node
        # Source: Module.f; Builder: Module.f, using the native module's map.
        # Retaining the owner keeps a local f from shadowing this GlobalVar.
        result = self.generic_visit(node)
        return (
            self._at(result, node)
            if isinstance(node.ctx, ast.Load) and not self.host_expression
            else result
        )

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        # Source: lambda n: n + outer; Builder: preserve the lambda's locals
        # while real annotation-only outer reads use their definition aliases.
        node.args.defaults = [self.visit(value) for value in node.args.defaults]
        node.args.kw_defaults = [
            self.visit(value) if value is not None else None for value in node.args.kw_defaults
        ]
        old = self.annotation_aliases
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments += [arg for arg in (node.args.vararg, node.args.kwarg) if arg]
        local_names = {argument.arg for argument in arguments}
        self.annotation_aliases = {
            name: alias for name, alias in old.items() if name not in local_names
        }
        node.body = self.visit(node.body)
        self.annotation_aliases = old
        return node

    def visit_ListComp(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp:
        # Source: [f(n) for n in values]; Builder: preserve Python comprehension
        # scope while translating its call/operand expressions exactly once.
        old = self.annotation_aliases
        self.annotation_aliases = dict(old)
        for generator in node.generators:
            generator.iter = self.visit(generator.iter)
            for target in ast.walk(generator.target):
                if isinstance(target, ast.Name):
                    self.annotation_aliases.pop(target.id, None)
            generator.ifs = [self.visit(value) for value in generator.ifs]
        if isinstance(node, ast.DictComp):
            node.key, node.value = self.visit(node.key), self.visit(node.value)
        else:
            node.elt = self.visit(node.elt)
        self.annotation_aliases = old
        return node

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_Constant(self, node: ast.Constant) -> ast.expr:
        # Source: literal; Builder: I.at_(literal_loc, literal).
        return node if self.host_expression else self._at(node, node)

    def visit_List(self, node: ast.List | ast.Tuple | ast.Set | ast.Dict) -> ast.expr:
        # Source: [a,b] / (a,b) / {a,b}; Builder: preserve the Python container
        # and source-location every original child through this same visitor.
        result = self.generic_visit(node)
        return result if self.host_expression else self._at(result, node)

    visit_Tuple = visit_List
    visit_Set = visit_List
    visit_Dict = visit_List

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.JoinedStr:
        # Source: f"value={expr}"; Builder: keep literal fragments, visit expr.
        # Python requires these fragments to remain Constant/FormattedValue.
        for child in node.values:
            if isinstance(child, ast.FormattedValue):
                child.value = self.visit(child.value)
                if child.format_spec is not None:
                    child.format_spec = self.visit_JoinedStr(child.format_spec)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        # Source: buffer[index]; Builder: I.at_(load_loc, buffer[index]).
        result = self.generic_visit(node)
        if not self.host_expression and isinstance(node.ctx, ast.Load):
            return self._at(result, node)
        return result

    def _module_owner(self, node: ast.expr) -> bool:
        """Recognize fixed source module aliases from existing binding records."""
        if not isinstance(node, ast.Name):
            return False
        if node.id == self.module_name:
            return True
        records = [
            item
            for item in self.prescan.bindings.get(self.current_scope, ())
            if item.name == node.id
        ]
        return bool(records) and all(item.kind == "module_alias" for item in records)

    def visit_Call(self, node: ast.Call, *, callee: ast.expr | None = None) -> ast.expr:
        # Source: X.Tensor(("n",), vdevice="cuda:0")
        # Builder: X.Tensor((X.resolve_type_var_("n"),),
        #                   vdevice=I.resolve_global_info("cuda:0"))
        marker = self._constexpr_operand(node)
        if marker is not None:
            with self._host():
                return self.visit(marker)
        generated = getattr(node, "_tvm_intrinsic", False)
        if not self.host_expression and not generated:
            node = handle_call_args_policy(
                node, self._resolve, self.infrastructure_name, self.filename
            )
        global_call = (
            isinstance(node.func, ast.Name) and node.func.id in self.module_functions
        ) or (
            isinstance(node.func, ast.Attribute)
            and self._module_owner(node.func.value)
            and node.func.attr in self.module_functions
        )
        # Normalized callees are already builder syntax. Mark only that
        # attribute so generic_visit still handles every original argument once.
        if callee is not None:
            node.func = callee
        if callee is not None or generated:
            node.func._tvm_intrinsic = True
        node = self.generic_visit(node)
        # Source: declared_global(x, y)
        # Builder: X.call_global_var_(declared_global, [x, y])
        if global_call and not self.host_expression:
            if node.keywords:
                self._error(node, "Global function calls require positional arguments")
            node = self._call(
                self.dialect_prefix,
                "call_global_var_",
                [node.func, ast.List(node.args, ast.Load())],
                node,
            )
        # Source: f(a); Builder: I.with_at_group_(loc, lambda: f(a)).
        # The thunk evaluates callee/arguments once inside the source call span.
        if self.track_span and not generated:
            return self._call(
                self.infrastructure_name,
                "with_at_group_",
                [self.span(node), self._lambda([], node)],
                node,
            )
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        # Source: not x; Builder: X.not_(x). Host constexpr keeps Python not.
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not) and not self.host_expression:
            return self._at(self._call(self.dialect_prefix, "not_", [node.operand], node), node)
        return self._at(node, node) if not self.host_expression else node

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        # Source: a + b; Builder: I.at_(loc, a + b), using native overloads.
        return (
            self._at(self.generic_visit(node), node)
            if not self.host_expression
            else self.generic_visit(node)
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        # Source: I.constexpr(enabled) and expr
        # Builder: enabled and expr. Other operands use X.and_/X.or_.
        if self.host_expression:
            return self.generic_visit(node)

        def lower(position: int) -> ast.expr:
            value = node.values[position]
            marker = self._constexpr_operand(value)
            if marker is not None:
                with self._host():
                    left = self.visit(marker)
                if position + 1 == len(node.values):
                    return left
                return ast.copy_location(ast.BoolOp(node.op, [left, lower(position + 1)]), node)
            left = self.visit(value)
            method = "and_" if isinstance(node.op, ast.And) else "or_"
            for index in range(position + 1, len(node.values)):
                if self._constexpr_operand(node.values[index]) is not None:
                    return self._call(self.dialect_prefix, method, [left, lower(index)], node)
                left = self._call(
                    self.dialect_prefix, method, [left, self.visit(node.values[index])], node
                )
            return left

        return self._at(lower(0), node)

    def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
        # Source: yes if I.constexpr(test) else no
        # Builder: yes if test else no. Unmarked tests use X.if_then_else_.
        if self.host_expression:
            return self.generic_visit(node)
        marker = self._constexpr_operand(node.test)
        if marker is not None:
            with self._host():
                test = self.visit(marker)
            return ast.copy_location(
                ast.IfExp(test, self.visit(node.body), self.visit(node.orelse)), node
            )
        return self._at(
            self._call(
                self.dialect_prefix,
                "if_then_else_",
                [self.visit(node.test), self.visit(node.body), self.visit(node.orelse)],
                node,
            ),
            node,
        )

    def _comparison(
        self, left: ast.expr, operation: ast.cmpop, right: ast.expr, node: ast.AST
    ) -> ast.expr:
        operations = {
            ast.Lt: "lt",
            ast.LtE: "le",
            ast.Gt: "gt",
            ast.GtE: "ge",
            ast.Eq: "eq",
            ast.NotEq: "ne",
        }
        if type(operation) in operations:
            return self._call(self.dialect_prefix, operations[type(operation)], [left, right], node)
        return ast.copy_location(ast.Compare(left, [operation], [right]), node)

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        # Source: a < b < c
        # Builder: (lambda a,b,c: X.and_(X.lt(a,b), X.lt(b,c),
        #                               chain=(a,b,c)))(a,b,c)
        # Written operands evaluate once and native builders bind shared IR uses.
        if self.host_expression:
            return self.generic_visit(node)
        if len(node.ops) == 1:
            result = self._comparison(
                self.visit(node.left), node.ops[0], self.visit(node.comparators[0]), node
            )
        else:
            operands = [node.left, *node.comparators]
            names = [self.fresh("_operand") for _ in operands]
            comparisons = [
                self._comparison(ast.Name(lhs, ast.Load()), op, ast.Name(rhs, ast.Load()), node)
                for lhs, op, rhs in zip(names, node.ops, names[1:])
            ]
            value = self._call(
                self.dialect_prefix,
                "and_",
                comparisons,
                node,
                chain=ast.Tuple([ast.Name(name, ast.Load()) for name in names], ast.Load()),
            )
            result = ast.copy_location(
                ast.Call(self._lambda(names, value), [self.visit(value) for value in operands], []),
                node,
            )
        return self._at(result, node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> NoReturn:
        # Source assignment expressions have no builder declaration contract.
        self._error(node, "Unsupported expression: NamedExpr")

    def visit_Await(self, node: ast.Await | ast.Yield | ast.YieldFrom) -> NoReturn:
        self._error(node, f"Unsupported expression: {type(node).__name__}")

    visit_Yield = visit_Await
    visit_YieldFrom = visit_Await

    def _index(self, node: ast.expr) -> ast.expr:
        # Source: a[start:stop:step]; Builder: X.setitem(a, slice(...), value).
        if isinstance(node, ast.Slice):
            return ast.copy_location(
                ast.Call(
                    self._inject(slice),
                    [
                        self.visit(value) if value else ast.Constant(None)
                        for value in (node.lower, node.upper, node.step)
                    ],
                    [],
                ),
                node,
            )
        if isinstance(node, ast.Tuple):
            return ast.copy_location(
                ast.Tuple([self._index(value) for value in node.elts], ast.Load()), node
            )
        return self.visit(node)

    def _bind(
        self,
        target: ast.expr,
        value: ast.expr,
        statement: ast.stmt,
        *,
        ty: ast.expr | None = None,
        frame_value: bool = False,
    ) -> list[ast.stmt]:
        # Declaration syntax has precedence; no previous/existence tracking.
        if isinstance(target, ast.Name):
            site = self.prescan.sites.get(target) if self.prescan else None
            kind = site.kind if site is not None else "ordinary"
            binding_declaration = kind == "binding_declaration" and (
                self._resolve(site.declaration_root) is not None
            )
            mutable = self.prescan.mutable_names.get(self.current_scope, ()) if self.prescan else ()
            keywords = {"name": ast.Constant(target.id), "name_span": self.span(target)}
            if ty is not None:
                keywords["ty"] = ty
            if frame_value:
                keywords["frame_value"] = ast.Constant(True)
            if kind == "symbol" and not frame_value:
                # Source: n = X.int64(); Builder: n = X.resolve_type_var_("n", dtype="int64").
                value = self._operation(
                    "resolve_type_var_",
                    [ast.Constant(target.id)],
                    target,
                    **({"dtype": ast.Constant(site.dtype)} if site.dtype else {}),
                )
            elif kind == "mutable" and not frame_value:
                # Source: x = X.local_scalar(...); Builder: x = X.decl_mutable_var_(...).
                value = self._operation("decl_mutable_var_", [value], statement, **keywords)
            elif target.id in mutable and not binding_declaration and not frame_value:
                # Source: x = value; Builder: X.set_mutable_var_(x, value).
                return [
                    ast.copy_location(
                        ast.Expr(
                            self._operation(
                                "set_mutable_var_",
                                [ast.Name(target.id, ast.Load()), value],
                                statement,
                            )
                        ),
                        statement,
                    )
                ]
            else:
                # Source: y = value; Builder: y = X.bind_(value, name="y").
                value = self._operation("bind_", [value], statement, **keywords)
            return [ast.copy_location(ast.Assign([target], value), statement)]
        if isinstance(target, ast.Attribute):
            # Source: a.field = value; Builder: X.setattr(a, "field", value).
            value = self._operation(
                "setattr", [self.visit(target.value), ast.Constant(target.attr), value], statement
            )
            return [ast.copy_location(ast.Expr(value), statement)]
        if isinstance(target, ast.Subscript):
            # Source: a[index] = value; Builder: X.setitem(a, index, value).
            value = self._operation(
                "setitem", [self.visit(target.value), self._index(target.slice), value], statement
            )
            return [ast.copy_location(ast.Expr(value), statement)]
        if isinstance(target, ast.Tuple | ast.List):
            # Source: a, (b,c) = rhs; Builder: unpack once at each reached level.
            names = [self.fresh("_unpack") for _ in target.elts]
            pattern: list[ast.expr] = [
                ast.Starred(ast.Name(name, ast.Store()), ast.Store())
                if isinstance(item, ast.Starred)
                else ast.Name(name, ast.Store())
                for name, item in zip(names, target.elts)
            ]
            result: list[ast.stmt] = [
                ast.copy_location(
                    ast.Assign(
                        [ast.Tuple(pattern, ast.Store())],
                        self._call(self.dialect_prefix, "unpack", [value], target),
                    ),
                    target,
                )
            ]
            for item, name in zip(target.elts, names):
                result.extend(
                    self._bind(
                        item.value if isinstance(item, ast.Starred) else item,
                        ast.Name(name, ast.Load()),
                        statement,
                        frame_value=frame_value,
                    )
                )
            return result
        self._error(target, f"Unsupported assignment target: {type(target).__name__}")

    def visit_Assign(self, node: ast.Assign) -> ast.Assign | list[ast.stmt]:
        target: ast.expr
        # Source: a = b = rhs; Builder: tmp = rhs; a = X.bind_(tmp); b = X.bind_(tmp).
        if self.host_expression:
            return self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0]
            site = self.prescan.sites.get(target) if self.prescan else None
            value = ast.Constant(None) if site and site.kind == "symbol" else self.visit(node.value)
            return self._bind(target, value, node)
        temporary = self.fresh("_value")
        result: list[ast.stmt] = [
            ast.copy_location(
                ast.Assign([ast.Name(temporary, ast.Store())], self.visit(node.value)), node
            )
        ]
        for target in node.targets:
            result.extend(self._bind(target, ast.Name(temporary, ast.Load()), node))
        return result

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign | list[ast.stmt]:
        # Source: x: X.int32 = v; Builder: x = X.decl_mutable_var_(v, ty=X.int32).
        if self.host_expression:
            return self.generic_visit(node)
        if not isinstance(node.target, ast.Name):
            self._error(node.target, "An annotated binding requires a name")
        value = (
            self.visit(node.value)
            if node.value
            else ast.Attribute(
                ast.Name(self.infrastructure_name, ast.Load()), "MISSING", ast.Load()
            )
        )
        old_aliases = self.annotation_aliases
        self.annotation_aliases = self.body_annotation_aliases
        annotation = self.visit(parse_annotation(node.annotation, self.filename))
        self.annotation_aliases = old_aliases
        return self._bind(node.target, value, node, ty=annotation)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AugAssign | list[ast.stmt]:
        key: ast.expr
        load: ast.expr
        value: ast.expr
        # Source: a[index] += value; Builder: evaluate base/index/load/RHS once,
        # then X.setitem(base, index, old + value). Name stores use D27 dispatch.
        if self.host_expression:
            return self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            value = ast.copy_location(
                ast.BinOp(ast.Name(node.target.id, ast.Load()), node.op, self.visit(node.value)),
                node,
            )
            return self._bind(node.target, self._at(value, node), node)
        if not isinstance(node.target, ast.Subscript | ast.Attribute):
            self._error(node.target, "An augmented assignment requires a name, attribute, or index")
        base = self.fresh("_base")
        statements: list[ast.stmt] = [
            ast.copy_location(
                ast.Assign([ast.Name(base, ast.Store())], self.visit(node.target.value)), node
            )
        ]
        if isinstance(node.target, ast.Attribute):
            key = ast.Constant(node.target.attr)
            load = ast.Attribute(ast.Name(base, ast.Load()), node.target.attr, ast.Load())
            operation = "setattr"
        else:
            index = self.fresh("_index")
            statements.append(
                ast.copy_location(
                    ast.Assign([ast.Name(index, ast.Store())], self._index(node.target.slice)), node
                )
            )
            key = ast.Name(index, ast.Load())
            load = ast.Subscript(ast.Name(base, ast.Load()), key, ast.Load())
            operation = "setitem"
        old = self.fresh("_old")
        statements.append(
            ast.copy_location(
                ast.Assign([ast.Name(old, ast.Store())], self._at(load, node.target)), node
            )
        )
        value = self._at(
            ast.BinOp(ast.Name(old, ast.Load()), node.op, self.visit(node.value)), node
        )
        statements.append(
            ast.copy_location(
                ast.Expr(
                    self._operation(operation, [ast.Name(base, ast.Load()), key, value], node)
                ),
                node,
            )
        )
        return statements

    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        # Source: f(); Builder: X.emit_(I.with_at_group_(loc, lambda: f())).
        if self.host_expression:
            return self.generic_visit(node)
        return ast.copy_location(
            ast.Expr(self._call(self.dialect_prefix, "emit_", [self.visit(node.value)], node)), node
        )

    def visit_Return(self, node: ast.Return) -> ast.Return | ast.Expr:
        # Source: return x; Builder: X.return_(x). Macros retain Python return.
        value = self.visit(node.value) if node.value else None
        if self.preserve_return or self.host_expression:
            return ast.copy_location(ast.Return(value), node)
        return ast.copy_location(
            ast.Expr(self._operation("return_", [] if value is None else [value], node)), node
        )

    def visit_Break(self, node: ast.Break) -> ast.Break | ast.Expr:
        # Source: break; Builder: X.break_().
        return (
            node
            if self.host_expression
            else ast.copy_location(ast.Expr(self._operation("break_", [], node)), node)
        )

    def visit_Continue(self, node: ast.Continue) -> ast.Continue | ast.Expr:
        # Source: continue; Builder: X.continue_().
        return (
            node
            if self.host_expression
            else ast.copy_location(ast.Expr(self._operation("continue_", [], node)), node)
        )

    def visit_Assert(self, node: ast.Assert) -> ast.Assert | ast.Expr:
        # Source: assert cond, msg; Builder: X.assert_(cond, msg).
        if self.host_expression:
            return self.generic_visit(node)
        message = self.visit(node.msg) if node.msg else ast.Constant("")
        return ast.copy_location(
            ast.Expr(self._operation("assert_", [self.visit(node.test), message], node)), node
        )

    def _branch(self, body: list[ast.stmt], node: ast.AST, prefix: str) -> list[ast.stmt]:
        # Branch helper scoping keeps source names; mutable stores bind no locals.
        name = self.fresh(prefix)
        return [
            self._definition(name, self.transform_statements(body), node),
            ast.copy_location(ast.Expr(ast.Call(ast.Name(name, ast.Load()), [], [])), node),
        ]

    def visit_If(self, node: ast.If) -> ast.If | list[ast.stmt]:
        # Source: if I.constexpr(flag): ...
        # Builder: if flag: ... in the enclosing Python scope.
        marker = self._constexpr_operand(node.test)
        if self.host_expression:
            return self.generic_visit(node)
        if marker is not None:
            with self._host():
                condition = self.visit(marker)
            return ast.copy_location(
                ast.If(
                    condition,
                    self.transform_statements(node.body) or [ast.Pass()],
                    self.transform_statements(node.orelse),
                ),
                node,
            )
        # Source: if cond: y = a; else: y = b
        # Builder:
        #   with X.If(cond) as frame:
        #       with X.Then():
        #           def then(): y = X.bind_(a, name="y")
        #           then()
        #       with X.Else():
        #           def otherwise(): y = X.bind_(b, name="y")
        #           otherwise()
        #   y = frame.var
        frame = self.fresh("_conditional")
        then = (
            self.transform_statements(node.body)
            if self.preserve_return
            else self._branch(node.body, node, "_then")
        )
        branches: list[ast.stmt] = [
            ast.copy_location(
                ast.With([ast.withitem(self._operation("Then", [], node))], then or [ast.Pass()]),
                node,
            )
        ]
        if node.orelse:
            otherwise = (
                self.transform_statements(node.orelse)
                if self.preserve_return
                else self._branch(node.orelse, node, "_else")
            )
            branches.append(
                ast.copy_location(
                    ast.With(
                        [ast.withitem(self._operation("Else", [], node))], otherwise or [ast.Pass()]
                    ),
                    node,
                )
            )
        result: list[ast.stmt] = [
            ast.copy_location(
                ast.With(
                    [
                        ast.withitem(
                            self._operation("If", [self.visit(node.test)], node),
                            ast.Name(frame, ast.Store()),
                        )
                    ],
                    branches,
                ),
                node,
            )
        ]
        output = self.prescan.conditional_outputs.get(node) if self.prescan else None
        if output is not None:
            result.append(
                ast.copy_location(
                    ast.Assign(
                        [ast.Name(output, ast.Store())],
                        ast.Attribute(ast.Name(frame, ast.Load()), "var", ast.Load()),
                    ),
                    node,
                )
            )
        return result

    def visit_For(self, node: ast.For) -> ast.For | ast.With:
        # Source:
        #   for i, *tail in X.grid(m,n,k): body
        # Builder:
        #   with X.for_(X.grid(m,n,k), names=("i","*tail")) as (i,*tail): body
        if self.host_expression:
            return self.generic_visit(node)
        if node.orelse:
            self._error(node, "A construction loop does not support an else clause")
        if isinstance(node.iter, ast.Call) and self._resolve(node.iter.func) is range:
            # Normalize only the known builtin; a lexical range binding follows
            # the ordinary source-call path. Arguments keep that one call scope.
            iterable = self.visit_Call(
                node.iter,
                callee=ast.Attribute(
                    ast.Name(self.dialect_prefix, ast.Load()), "range_", ast.Load()
                ),
            )
        else:
            iterable = self.visit(node.iter)
        if isinstance(node.target, ast.Name):
            names: ast.expr = ast.Constant(node.target.id)
        elif isinstance(node.target, ast.Tuple | ast.List):
            names = ast.Tuple(
                [
                    ast.Constant("*" + item.value.id if isinstance(item, ast.Starred) else item.id)
                    for item in node.target.elts
                ],
                ast.Load(),
            )
        else:
            self._error(node.target, "Loop targets must be names or a flat tuple of names")
        context = self._operation("for_", [iterable], node, names=names)
        body = self.transform_statements(node.body)
        return ast.copy_location(
            ast.With([ast.withitem(context, node.target)], body or [ast.Pass()]), node
        )

    def visit_While(self, node: ast.While) -> ast.While | ast.With:
        # Source: while cond: body; Builder: with X.While(cond): body.
        if self.host_expression:
            return self.generic_visit(node)
        if node.orelse:
            self._error(node, "A construction loop does not support an else clause")
        context = self._operation("While", [self.visit(node.test)], node)
        return ast.copy_location(
            ast.With([ast.withitem(context)], self.transform_statements(node.body) or [ast.Pass()]),
            node,
        )

    def visit_With(self, node: ast.With) -> ast.With | list[ast.stmt]:
        # Source: with X.block() as v: body
        # Builder: with X.block() as entered: v = X.bind_(entered, frame_value=True); body
        # Ordinary regions use Python locals; native dataflow outputs retain identity.
        if self.host_expression:
            return self.generic_visit(node)
        item, rest = node.items[0], node.items[1:]
        body: list[ast.stmt] = (
            [ast.copy_location(ast.With(rest, node.body), node)] if rest else node.body
        )
        if item.optional_vars is None:
            target, initial = None, []
        else:
            entered = self.fresh("_entered")
            target = ast.Name(entered, ast.Store())
            initial = self._bind(
                item.optional_vars, ast.Name(entered, ast.Load()), node, frame_value=True
            )
        outputs = self.prescan.with_outputs.get(node, ()) if self.prescan else ()
        context = self.visit(item.context_expr)
        translated_body = initial + self.transform_statements(body) or [ast.Pass()]
        if not outputs:
            return ast.copy_location(
                ast.With([ast.withitem(context, target)], translated_body), node
            )
        # Source: with X.dataflow(): ...; X.output(y)
        # Builder: with frame: ...; y = frame.output_vars[0]
        # The native frame owns conversion to ordinary output variables.
        frame = self.fresh("_dataflow")
        statements: list[ast.stmt] = [
            ast.copy_location(ast.Assign([ast.Name(frame, ast.Store())], context), node),
            ast.copy_location(
                ast.With([ast.withitem(ast.Name(frame, ast.Load()), target)], translated_body), node
            ),
        ]
        for index, name in enumerate(outputs):
            value = ast.Subscript(
                ast.Attribute(ast.Name(frame, ast.Load()), "output_vars", ast.Load()),
                ast.Constant(index),
                ast.Load(),
            )
            statements.append(
                ast.copy_location(ast.Assign([ast.Name(name, ast.Store())], value), node)
            )
        return statements

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef | list[ast.stmt]:
        # Source: nested @X.function def f(...): body
        # Builder:
        #   declare fn
        #   with fn:
        #       def build(): body
        #       build()
        if self.host_expression:
            return node
        kind, _ = self.function_metadata(node, allow_python=True)
        if kind.python:
            return node
        declaration, _, body = self.function_program(node, local=True)
        return [*declaration, body]

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Pass:
        # Source closure declarations do not mutate the host closure during build.
        return ast.copy_location(ast.Pass(), node)

    def visit_Pass(self, node: ast.Pass) -> ast.Pass:
        # Source: pass; Builder: pass.
        return node

    def generic_visit(self, node: _Node) -> _Node:
        if isinstance(node, ast.stmt) and not self.host_expression:
            self._error(node, f"Unsupported statement: {type(node).__name__}")
        return super().generic_visit(node)

    def function_metadata(
        self, node: ast.FunctionDef, *, allow_python: bool = False
    ) -> tuple[protocol.FunctionDecoratorInfo, ast.Dict]:
        """Read registered decorator options without evaluating source values."""
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            kind = protocol.function_info(self._resolve(target))
            if kind is None:
                continue
            values: dict[str, ast.expr] = {
                key: ast.Constant(value) for key, value in (kind.defaults or {}).items()
            }
            expansions = []
            if isinstance(decorator, ast.Call):
                if decorator.args:
                    self._error(decorator, "Function decorators accept keyword options only")
                for keyword in decorator.keywords:
                    if keyword.arg is None:
                        expansions.append(keyword.value)
                    elif keyword.arg != "check_well_formed":
                        values[(kind.option_map or {}).get(keyword.arg, keyword.arg)] = (
                            keyword.value
                        )
            return kind, ast.copy_location(
                ast.Dict(
                    [ast.Constant(key) for key in values] + [None] * len(expansions),
                    list(values.values()) + expansions,
                ),
                node,
            )
        if allow_python:
            return protocol.FunctionDecoratorInfo(None, python=True), ast.Dict([], [])
        self._error(node, f"Function {node.name!r} has no registered construction kind")

    def function_program(
        self, node: ast.FunctionDef, *, local: bool = False, declare: bool = True
    ) -> tuple[list[ast.stmt], str, ast.With]:
        """Declare a native frame and emit a lexical body helper inside its scope.

        Annotation aliases retain actual definition-local Python values; source
        parameters are read from the enclosing frame while its zero-argument
        helper runs. No factory, callback record, copied parameter map or symbol
        owner is generated.
        """
        from . import jit_support

        kind, options = self.function_metadata(node)
        builder = self.fresh("_X")
        self.bindings[builder] = kind.builder
        frame, body_name = self.fresh("_fn"), self.fresh("_build")
        old = (self.dialect_prefix, self.current_scope, self.annotation_aliases)
        self.dialect_prefix, self.current_scope = builder, node
        parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.args.vararg or node.args.kwarg:
            self._error(node, "IR signatures require ordinary named parameters")
        facts = self.prescan.bindings.get(node, ()) if self.prescan else ()
        type_parameters = list(getattr(node, "type_params", ()))
        declared_names = {item.name for item in type_parameters}
        local_names = {item.name for item in facts}
        # Definition-context aliases are needed only for real annotation names.
        # Quoted expression names are created later by argument normalization.
        annotations = [
            parse_annotation(parameter.annotation, self.filename) if parameter.annotation else None
            for parameter in parameters
        ]
        returns = parse_annotation(node.returns, self.filename) if node.returns else None
        annotation_names = {
            item.id
            for annotation in [*annotations, returns]
            if annotation is not None
            for item in ast.walk(annotation)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
        }
        body_annotations = [
            parse_annotation(item.annotation, self.filename)
            for item in facts
            if item.annotation is not None
            and item.kind not in ("parameter", "mutable_parameter", "symbol")
        ]
        annotation_names.update(
            item.id
            for annotation in body_annotations
            for item in ast.walk(annotation)
            if isinstance(item, ast.Name)
            and isinstance(item.ctx, ast.Load)
            and item.id not in local_names
        )
        aliases = {
            name: self.fresh("_annotation") for name in sorted(annotation_names - declared_names)
        }
        captures = self.fresh("_definition")
        captures_expr = ast.Dict(
            [None, None, None],
            [
                ast.Call(self._inject(globals), [], []),
                self._call(
                    self.definition_scopes_name,
                    "get",
                    [ast.Constant(node.name), ast.Dict([], [])],
                    node,
                )
                if self.definition_scopes_name is not None and not local
                else ast.Dict([], []),
                ast.Call(self._inject(locals), [], []),
            ],
        )
        statements: list[ast.stmt] = [
            ast.copy_location(ast.Assign([ast.Name(captures, ast.Store())], captures_expr), node)
        ]
        fallback: ast.expr
        value: ast.expr
        for name, alias in aliases.items():
            fallback = (
                self._inject(getattr(builtins, name))
                if hasattr(builtins, name)
                else ast.Attribute(
                    ast.Name(self.infrastructure_name, ast.Load()), "MISSING", ast.Load()
                )
            )
            value = self._call(captures, "get", [ast.Constant(name), fallback], node)
            statements.append(
                ast.copy_location(ast.Assign([ast.Name(alias, ast.Store())], value), node)
            )
        special, absent = self.fresh("_specialization"), self.fresh("_absent")
        special_expr: ast.expr
        absent_expr: ast.expr
        if local:
            special_expr, absent_expr = ast.Constant(None), ast.Tuple([], ast.Load())
        else:
            special_expr = ast.Call(
                self._inject(jit_support.specialization_bindings), [ast.Constant(node.name)], []
            )
            absent_expr = ast.Call(
                self._inject(jit_support.absent_parameter_names), [ast.Constant(node.name)], []
            )
        statements.extend(
            [
                ast.copy_location(ast.Assign([ast.Name(special, ast.Store())], special_expr), node),
                ast.copy_location(ast.Assign([ast.Name(absent, ast.Store())], absent_expr), node),
            ]
        )
        declaration: list[ast.stmt] = [
            ast.copy_location(
                ast.Expr(self._call(builder, "func_name", [ast.Constant(node.name)], node)), node
            )
        ]
        # Source: def f[n](...); Builder: n = X.resolve_type_var_("n").
        # Explicit dtype declarations are predeclared before quoted shapes, but
        # only explicit type parameters introduce signature Python bindings.
        symbol_aliases = {}
        for parameter in type_parameters:
            if not isinstance(parameter, getattr(ast, "TypeVar", ())):
                self._error(parameter, "Only scalar type parameters are supported")
            bound = getattr(parameter, "bound", None)
            if bound is not None and self._resolve(bound) is not int:
                self._error(parameter, "A symbolic type parameter bound must be int")
            if getattr(parameter, "default_value", None) is not None:
                self._error(parameter, "A symbolic type parameter cannot have a default")
            alias = self.fresh("_symbol")
            symbol_aliases[parameter.name] = alias
            declaration.append(
                ast.copy_location(
                    ast.Assign(
                        [ast.Name(alias, ast.Store())],
                        self._operation(
                            "resolve_type_var_", [ast.Constant(parameter.name)], parameter
                        ),
                    ),
                    parameter,
                )
            )
        for item in facts:
            if item.direct and item.dtype is not None and item.kind in ("symbol", "parameter"):
                symbol = self._operation(
                    "resolve_type_var_",
                    [ast.Constant(item.name)],
                    item.node,
                    dtype=ast.Constant(item.dtype),
                )
                # The native map knows later scalar dtypes, but a later Python
                # parameter name is not in annotation scope until its own arg.
                declaration.append(ast.copy_location(ast.Expr(symbol), item.node))
        self.annotation_aliases = {**aliases, **symbol_aliases}
        constexpr_aliases = {}
        ordered = sorted(
            zip(parameters, annotations),
            key=lambda pair: not (
                isinstance(pair[1], ast.Attribute) and pair[1].attr == "constexpr"
            ),
        )
        for parameter, annotation in ordered:
            if annotation is None:
                self._error(parameter, f"Parameter {parameter.arg!r} requires an annotation")
            name = parameter.arg
            alias = self.fresh("_parameter")
            is_constexpr = isinstance(annotation, ast.Attribute) and annotation.attr == "constexpr"
            if is_constexpr:
                constexpr_aliases[name] = alias
                fallback = ast.Call(
                    self._inject(jit_support.constexpr_binding),
                    [
                        self._call(
                            captures,
                            "get",
                            [
                                ast.Constant(name),
                                ast.Attribute(
                                    ast.Name(self.infrastructure_name, ast.Load()),
                                    "MISSING",
                                    ast.Load(),
                                ),
                            ],
                            parameter,
                        ),
                        ast.Constant(name),
                    ],
                    [],
                )
            else:
                translated = self.visit(annotation)
                checked = ast.Call(
                    self._inject(jit_support.unwrap_annotation),
                    [translated, ast.Name(special, ast.Load())],
                    [],
                )
                fallback = self._operation("arg", [ast.Constant(name), checked], parameter)
            # Source parameters selected by JIT have no runtime ABI slot. Their
            # annotation thunk is absent from the selected Python branch.
            selected = ast.BoolOp(
                ast.And(),
                [
                    ast.Compare(ast.Name(special, ast.Load()), [ast.IsNot()], [ast.Constant(None)]),
                    ast.Compare(ast.Constant(name), [ast.In()], [ast.Name(special, ast.Load())]),
                ],
            )
            absent_test = ast.Compare(
                ast.Constant(name), [ast.In()], [ast.Name(absent, ast.Load())]
            )
            value = ast.IfExp(
                selected,
                ast.Subscript(ast.Name(special, ast.Load()), ast.Constant(name), ast.Load()),
                ast.IfExp(absent_test, ast.Constant(None), fallback),
            )
            declaration.append(
                ast.copy_location(ast.Assign([ast.Name(alias, ast.Store())], value), parameter)
            )
            self.annotation_aliases[name] = alias
        if returns is not None:
            declaration.append(
                ast.copy_location(
                    ast.Expr(
                        self._call(
                            builder,
                            "func_ret_type",
                            [self._lambda([], self.visit(returns))],
                            returns,
                        )
                    ),
                    returns,
                )
            )
        self.annotation_aliases = {}
        # Source: @X.function def f(...): ...
        # Builder: with X.function(decl=True) as fn: signature
        with self._host():
            options = self.visit(options)
        keywords = [ast.keyword(None, options)]
        if declare:
            keywords.append(ast.keyword("decl", ast.Constant(True)))
        if local:
            keywords.append(ast.keyword("local", ast.Constant(True)))
        if self.track_span:
            keywords.append(ast.keyword("span", self.span(node)))
        constructor = ast.copy_location(
            ast.Call(
                ast.Attribute(ast.Name(builder, ast.Load()), "function", ast.Load()), [], keywords
            ),
            node,
        )
        if declare:
            declaration_scope = ast.With(
                [ast.withitem(constructor, ast.Name(frame, ast.Store()))], declaration
            )
            statements.append(ast.copy_location(declaration_scope, node))
            reference = ast.Attribute(ast.Name(frame, ast.Load()), "reference", ast.Load())
            statements.append(
                ast.copy_location(ast.Assign([ast.Name(node.name, ast.Store())], reference), node)
            )
        else:
            # Source: a standalone nonrecursive function.
            # Builder: fn = X.function(); with fn: define/call a helper for
            # both signature and body, entering this ordinary frame just once.
            statements.append(
                ast.copy_location(ast.Assign([ast.Name(frame, ast.Store())], constructor), node)
            )
        # The body re-enters the exact native frame. Runtime parameter storage
        # stays native; the short iterator is consumed once by source parameters.
        iterator = self.fresh("_arguments")
        body: list[ast.stmt] = [
            ast.copy_location(
                ast.Assign(
                    [ast.Name(iterator, ast.Store())],
                    ast.Call(
                        self._inject(iter),
                        [ast.Attribute(ast.Name(frame, ast.Load()), "params", ast.Load())],
                        [],
                    ),
                ),
                node,
            )
        ]
        for parameter in parameters:
            name = parameter.arg
            if name in constexpr_aliases:
                value = ast.Name(constexpr_aliases[name], ast.Load())
            else:
                selected = ast.BoolOp(
                    ast.And(),
                    [
                        ast.Compare(
                            ast.Name(special, ast.Load()), [ast.IsNot()], [ast.Constant(None)]
                        ),
                        ast.Compare(
                            ast.Constant(name), [ast.In()], [ast.Name(special, ast.Load())]
                        ),
                    ],
                )
                absent_test = ast.Compare(
                    ast.Constant(name), [ast.In()], [ast.Name(absent, ast.Load())]
                )
                value = ast.IfExp(
                    selected,
                    ast.Subscript(ast.Name(special, ast.Load()), ast.Constant(name), ast.Load()),
                    ast.IfExp(
                        absent_test,
                        ast.Constant(None),
                        ast.Call(self._inject(next), [ast.Name(iterator, ast.Load())], []),
                    ),
                )
            body.append(
                ast.copy_location(ast.Assign([ast.Name(name, ast.Store())], value), parameter)
            )
        for parameter in type_parameters:
            body.append(
                ast.copy_location(
                    ast.Assign(
                        [ast.Name(parameter.name, ast.Store())],
                        self._operation(
                            "resolve_type_var_", [ast.Constant(parameter.name)], parameter
                        ),
                    ),
                    parameter,
                )
            )
        # Body annotations use definition aliases only for names without a real
        # body binding. Ordinary body references are visited with no substitution.
        body_annotation_aliases = {
            name: alias for name, alias in aliases.items() if name not in local_names
        }
        old_body_annotations = self.body_annotation_aliases
        self.body_annotation_aliases = body_annotation_aliases
        body.extend(self.transform_statements(node.body))
        self.body_annotation_aliases = old_body_annotations
        definition = self._definition(body_name, body if declare else [*declaration, *body], node)
        if not local:
            definition._tvm_source_name = node.name
            definition._tvm_signature_names = (
                declared_names | {node.name} | set(self.module_functions)
            )
            if self.module_name:
                definition._tvm_signature_names.add(self.module_name)
        # Source: def f(...): body
        # Builder:
        #   with fn:
        #       def build(): body
        #       build()
        # The helper owns only Python lexical scope; the enclosing with owns
        # native frame entry/exit, including unwinding a failed body.
        invocation = ast.copy_location(
            ast.Expr(ast.Call(ast.Name(body_name, ast.Load()), [], [])), node
        )
        resumed = ast.copy_location(
            ast.With([ast.withitem(ast.Name(frame, ast.Load()))], [definition, invocation]), node
        )
        self.dialect_prefix, self.current_scope, self.annotation_aliases = old
        return statements, frame, resumed

    def program(self, tree: ast.Module) -> tuple[ast.Module, str]:
        """Emit direct native module construction, declarations, then bodies."""
        from .entry import make_opaque_function

        root, prefix = tree.body[-1], tree.body[:-1]
        if not isinstance(root, ast.ClassDef | ast.FunctionDef):
            self._error(root, "Source must contain one function or module class")
        is_module = isinstance(root, ast.ClassDef)
        members = root.body if is_module else [root]
        functions = [item for item in members if isinstance(item, ast.FunctionDef)]
        if len({item.name for item in functions}) != len(functions):
            self._error(root, "Duplicate function declaration")
        self.module_name = root.name if is_module else None
        self.module_functions = frozenset(item.name for item in functions)
        declare = is_module or root in self.prescan.recursive_functions
        builder, result = self.fresh("_builder"), self.fresh("_result")
        body: list[ast.stmt] = []
        for function in functions if declare else ():
            body.append(
                ast.copy_location(
                    ast.Assign(
                        [ast.Name(function.name, ast.Store())],
                        self._call(
                            self.infrastructure_name,
                            "reserve_function",
                            [ast.Constant(function.name)],
                            function,
                        ),
                    ),
                    function,
                )
            )
        for member in members:
            if isinstance(member, ast.FunctionDef):
                continue
            # Source class statements are host setup inside the native module;
            # global-info registration therefore precedes dependent annotations.
            with self._host():
                body.append(self.visit(member))
            targets = (
                member.targets
                if isinstance(member, ast.Assign)
                else [member.target]
                if isinstance(member, ast.AnnAssign)
                else []
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    value = self._call(
                        self.infrastructure_name,
                        "module_member_",
                        [ast.Constant(target.id), ast.Name(target.id, ast.Load())],
                        target,
                    )
                    body.append(
                        ast.copy_location(
                            ast.Assign([ast.Name(target.id, ast.Store())], value), target
                        )
                    )
        definitions, frames, python_functions = [], [], []
        for function in functions:
            kind, _ = self.function_metadata(function)
            if kind.python:
                # Source: @I.pyfunc def f(...): Python body
                # Builder: opaque = make_opaque_function(...); I.def_function(...)
                source = ast.unparse(function)
                function.decorator_list = []
                body.append(function)
                host = self.fresh("_python")
                body.append(
                    ast.copy_location(
                        ast.Assign(
                            [ast.Name(host, ast.Store())], ast.Name(function.name, ast.Load())
                        ),
                        function,
                    )
                )
                opaque = self.fresh("_opaque")
                value = ast.Call(
                    self._inject(make_opaque_function),
                    [
                        ast.Constant(function.name),
                        ast.Name(host, ast.Load()),
                        ast.Constant(source),
                        self.span(function),
                    ],
                    [],
                )
                body.append(
                    ast.copy_location(ast.Assign([ast.Name(opaque, ast.Store())], value), function)
                )
                body.append(
                    ast.copy_location(
                        ast.Assign(
                            [ast.Name(function.name, ast.Store())],
                            self._call(
                                self.infrastructure_name,
                                "decl_function",
                                [ast.Constant(function.name), ast.Name(opaque, ast.Load())],
                                function,
                            ),
                        ),
                        function,
                    )
                )
                body.append(
                    ast.copy_location(
                        ast.Expr(
                            self._call(
                                self.infrastructure_name,
                                "def_function",
                                [ast.Constant(function.name), ast.Name(opaque, ast.Load())],
                                function,
                            )
                        ),
                        function,
                    )
                )
                python_functions.append((function.name, host))
                continue
            declaration, frame, definition = self.function_program(function, declare=declare)
            body.extend(declaration)
            definitions.append(definition)
            frames.append(frame)
        body.extend(definitions)
        # Source: class Module: functions...
        # Builder:
        #   with IRBuilder() as builder:
        #       with I.ir_module():
        #           declarations
        #           with fn:
        #               def build_f(): body_f
        #               build_f()
        #           with gn:
        #               def build_g(): body_g
        #               build_g()
        #   result = builder.get()
        module = ast.copy_location(
            ast.With(
                [
                    ast.withitem(
                        self._call(self.infrastructure_name, "ir_module", [], root),
                        ast.Name(root.name, ast.Store()) if is_module else None,
                    )
                ],
                body,
            ),
            root,
        )
        construction = ast.copy_location(
            ast.With(
                [
                    ast.withitem(
                        self._call(self.infrastructure_name, "IRBuilder", [], root),
                        ast.Name(builder, ast.Store()),
                    )
                ],
                [module] if declare else body,
            ),
            root,
        )
        output = (
            self._call(builder, "get", [], root)
            if is_module
            else ast.Attribute(ast.Name(frames[0], ast.Load()), "function", ast.Load())
        )
        translated = [item for item in prefix if not isinstance(item, ast.Import | ast.ImportFrom)]
        translated.extend(
            [
                construction,
                ast.copy_location(ast.Assign([ast.Name(result, ast.Store())], output), root),
            ]
        )
        translated.append(
            ast.copy_location(
                ast.Assign(
                    [ast.Attribute(ast.Name(result, ast.Load()), "__name__", ast.Store())],
                    ast.Constant(root.name),
                ),
                root,
            )
        )
        if python_functions:
            translated.append(
                ast.copy_location(
                    ast.Assign(
                        [ast.Attribute(ast.Name(result, ast.Load()), "pyfuncs", ast.Store())],
                        ast.Dict(
                            [ast.Constant(name) for name, _ in python_functions],
                            [ast.Name(alias, ast.Load()) for _, alias in python_functions],
                        ),
                    ),
                    root,
                )
            )
        return ast.fix_missing_locations(ast.Module(translated, [])), result
