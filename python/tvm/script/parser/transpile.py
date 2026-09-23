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

The prescan supplies read-only syntax facts. This visitor owns only translation
inputs, generated-name allocation and a small lexical rewrite context. Native
frames own symbols, declarations, parameters, region results and final IR.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from types import FunctionType
from typing import Any, NoReturn, TypeVar

from tvm.script.ir_builder.ir import parser_protocol as protocol

from .call_args_policy import handle_call_args_policy, parse_annotation, parse_expression_string
from .prescan import Binding, PrescanContext, resolve_constructor, resolve_syntax

_Node = TypeVar("_Node", bound=ast.AST)


class ModuleContext:
    """One parse's shared syntax inputs and hygienic allocation, never native state."""

    def __init__(
        self,
        filename: str,
        environment: Mapping[str, object],
        infrastructure_name: str,
        span: Callable[[ast.AST], ast.expr],
        fresh: Callable[[str], str],
        *,
        prescan: PrescanContext | None,
        track_span: bool,
        definition_scope: Mapping[str, Any],
        definition_scope_name: str | None,
        source_functions: Mapping[str, FunctionType],
        bindings: dict[str, Any],
    ) -> None:
        # Fixed namespace/protocol metadata and source-coordinate inputs for this parse.
        self.filename = filename
        self.environment = environment
        self.infrastructure_name = infrastructure_name
        self.span = span
        self.track_span = track_span
        self.prescan = prescan
        # One temporary entry scope, shared by all members without per-member copies.
        self.definition_scope = definition_scope
        self.definition_scope_name = definition_scope_name
        self.source_functions = source_functions
        # Allocation and injected host values intentionally accumulate across functions.
        self.fresh = fresh
        self.bindings = bindings
        # Root syntax identifies plain module references and dialect function callees.
        self.module_name: str | None = None
        self.module_functions: frozenset[str] = frozenset()
        # Explicit, one-way generated-helper/source handoff to private recomposition.
        self.body_sources: list[tuple[ast.FunctionDef, str, set[str]]] = []


class FunctionContext:
    """Fresh lexical rewrite state for one source function, restored after nesting."""

    def __init__(self, current_scope: ast.AST | None, dialect_prefix: str) -> None:
        # Source scope selects declaration facts; the dialect binding is generated syntax.
        self.current_scope = current_scope
        self.dialect_prefix = dialect_prefix
        # Only definition/signature substitutions, never values or a second body map.
        self.annotation_aliases: dict[str, str] = {}


class IRBuilderTranspiler(ast.NodeTransformer):
    """A single statement/expression visitor over the entry-owned AST.

    ``environment``, ``prescan``, ``filename`` and ``span`` are fixed inputs for
    one translation. ``fresh`` allocates names in the entry-owned name map;
    ``bindings`` receives only injected host namespaces and helper functions.
    ``dialect_prefix``, ``current_scope``, ``bypass_ast_rewrite`` and annotation
    substitutions are saved/restored at their lexical visitor boundaries. They
    contain AST names, never native frames, values or construction ownership.
    """

    def __init__(
        self, module: ModuleContext, function: FunctionContext, *, preserve_return: bool = False
    ) -> None:
        # Contexts own syntax inputs only and never refer back to this rewriter.
        self.module = module
        self.function = function
        # Macro return policy is fixed for one invocation.
        self.preserve_return = preserve_return
        # Bypass syntax-to-builder lowering.
        # Still traverse children and instrument source calls with spans.
        self.bypass_ast_rewrite = False
        # Temporary policy only while this same visitor processes a decoded string.
        self.expression_string: protocol.ExprStrPolicy | None = None
        # Only annotation syntax consults the function's one substitution map;
        # ordinary body globals keep Python lookup even when names coincide.
        self.annotation_expression = False

    def _inject(self, value: object, prefix: str = "_host") -> ast.Name:
        name = self.module.fresh(prefix)
        self.module.bindings[name] = value
        return ast.Name(name, ast.Load())

    def _raise_error(self, node: ast.AST, message: str) -> NoReturn:
        raise SyntaxError(message, (self.module.filename, node.lineno, node.col_offset + 1, None))

    def _call(
        self, namespace: str, member: str, args: list[ast.expr], node: ast.AST, **keywords: ast.expr
    ) -> ast.Call:
        """Build a generated operation with its source range and named arguments."""
        if not self.module.track_span:
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

    def _call_dialect(
        self, member: str, args: list[ast.expr], node: ast.AST, **keywords: ast.expr
    ) -> ast.Call:
        return self._call(
            self.function.dialect_prefix,
            member,
            args,
            node,
            span=self.module.span(node),
            **keywords,
        )

    def _attach_span(self, value: ast.expr, node: ast.AST) -> ast.expr:
        if not self.module.track_span:
            return value
        return self._call(
            self.module.infrastructure_name, "at_", [self.module.span(node), value], node
        )

    @staticmethod
    def _assign(name: str, value: ast.expr, node: ast.AST) -> ast.Assign:
        """Assign an injected or source name while retaining its source range."""
        return ast.copy_location(ast.Assign([ast.Name(name, ast.Store())], value), node)

    @staticmethod
    def _create_lambda(names: list[str], value: ast.expr) -> ast.Lambda:
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
    def _create_definition(name: str, body: list[ast.stmt], node: ast.AST) -> ast.FunctionDef:
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
    def _bypass_rewrite(self) -> Iterator[None]:
        # Only constexpr operands and module host syntax keep Python operators.
        # The same visitor still instruments their source calls and restores mode.
        old = self.bypass_ast_rewrite
        self.bypass_ast_rewrite = True
        try:
            yield
        finally:
            self.bypass_ast_rewrite = old

    @contextmanager
    def _use_aliases(self, mapping: dict[str, str]) -> Iterator[None]:
        """Restore lexical annotation substitutions even when a visitor fails."""
        old = self.function.annotation_aliases
        self.function.annotation_aliases = mapping
        try:
            yield
        finally:
            self.function.annotation_aliases = old

    @contextmanager
    def _rewrite_annotation(self) -> Iterator[None]:
        """Use definition substitutions only while rewriting annotation syntax."""
        previous, self.annotation_expression = self.annotation_expression, True
        try:
            yield
        finally:
            self.annotation_expression = previous

    def _resolve(self, node: ast.AST | None) -> object:
        # Fixed namespace meanings coexist with Python lexical value bindings.
        # A local ``range`` or callable hides the ambient binding for the whole
        # source function, including reads before its assignment.
        scope = self.function.current_scope
        facts: list[Binding] = []
        if self.module.prescan is not None:
            while isinstance(scope, ast.FunctionDef):
                current = self.module.prescan.bindings.get(scope, ())
                hidden = {item.name for item in facts}
                facts.extend(item for item in current if item.name not in hidden)
                scope = next(
                    (
                        parent
                        for parent, items in self.module.prescan.bindings.items()
                        if any(item.node is scope and item.kind == "function" for item in items)
                    ),
                    None,
                )
        return resolve_syntax(node, self.module.environment, facts)

    def _resolve_constructor(self, node: ast.expr) -> object:
        if isinstance(node, ast.Call):
            return self._resolve(node.func)
        facts = (
            self.module.prescan.bindings.get(self.function.current_scope, ())
            if self.module.prescan
            else ()
        )
        return resolve_constructor(node, self.module.environment, facts)

    def _read_constexpr_operand(self, node: ast.expr) -> ast.expr | None:
        # -------------------- Pattern --------------------
        # Python source:
        #     I.constexpr(expr)
        #
        # Builder:
        #     expr
        # -------------------------------------------------
        # The same visitor keeps Python operators and still instruments nested source calls.
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return None
        if node.func.attr != "constexpr" or not isinstance(node.func.value, ast.Name):
            return None
        if (
            self.module.prescan is not None
            and node.func.value.id not in self.module.prescan.namespaces
        ):
            return None
        if len(node.args) != 1 or node.keywords or isinstance(node.args[0], ast.Starred):
            self._raise_error(node, "constexpr expects exactly one controlling value")
        return node.args[0]

    def visit_Name(self, node: ast.Name) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     X.Tensor(("n",))
        #
        # Builder:
        #     X.Tensor((X.resolve_type_var_("n"),))
        # -------------------------------------------------
        # Only the marked string is symbolic; no Python binding is introduced.
        if self.expression_string is not None:
            keywords = {}
            if self.expression_string.dtype is not None:
                keywords["dtype"] = ast.Constant(self.expression_string.dtype)
            return self._attach_span(
                self._call(
                    self.function.dialect_prefix,
                    "resolve_type_var_",
                    [ast.Constant(node.id)],
                    node,
                    **keywords,
                ),
                node,
            )
        # -------------------- Pattern --------------------
        # Python source:
        #     value: X.Tensor((n,))
        #
        # Builder:
        #     value_type = X.Tensor((I.require_defined(I.annotation_value_("n", captured_n), "n"),))
        # -------------------------------------------------
        # Definition substitutions apply only within annotation syntax.
        # A preceding body target is already a Python binding (including symbols).
        # Consult its existing prescan location, without a second body-alias map.
        if self.annotation_expression and self.module.prescan is not None:
            if any(
                item.name == node.id
                and isinstance(item.node, ast.Name)
                and (item.node.lineno, item.node.col_offset) < (node.lineno, node.col_offset)
                for item in self.module.prescan.bindings.get(self.function.current_scope, ())
            ):
                return self._attach_span(node, node)
        if (
            self.annotation_expression
            and isinstance(node.ctx, ast.Load)
            and node.id in self.function.annotation_aliases
        ):
            return self._call(
                self.module.infrastructure_name,
                "require_defined",
                [
                    self._call(
                        self.module.infrastructure_name,
                        "annotation_value_",
                        [
                            ast.Constant(node.id),
                            ast.Name(self.function.annotation_aliases[node.id], ast.Load()),
                        ],
                        node,
                    ),
                    ast.Constant(node.id),
                ],
                node,
            )
        return (
            self._attach_span(node, node)
            if isinstance(node.ctx, ast.Load) and not self.bypass_ast_rewrite
            else node
        )

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        # Within a decoded string, known namespace attributes remain Python
        # lookup; unknown roots still denote symbolic variables.
        if self.expression_string is not None and self._resolve(node.value) is not None:
            with self._use_string_policy(None):
                return self.visit_Attribute(node)
        # -------------------- Pattern --------------------
        # Python source:
        #     Module.f
        #
        # Builder:
        #     I.at_(loc, Module.f)
        # -------------------------------------------------
        # Keeping the module owner prevents a local f from shadowing its GlobalVar.
        result = self.generic_visit(node)
        return (
            self._attach_span(result, node)
            if isinstance(node.ctx, ast.Load) and not self.bypass_ast_rewrite
            else result
        )

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        # -------------------- Pattern --------------------
        # Python source:
        #     lambda n: n + outer
        #
        # Builder:
        #     lambda n: n + captured_outer
        # -------------------------------------------------
        # Inside annotations, lambda-local binders mask definition substitutions.
        node.args.defaults = [self.visit(value) for value in node.args.defaults]
        node.args.kw_defaults = [
            self.visit(value) if value is not None else None for value in node.args.kw_defaults
        ]
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments += [arg for arg in (node.args.vararg, node.args.kwarg) if arg]
        local_names = {argument.arg for argument in arguments}
        with self._use_aliases(
            {
                name: alias
                for name, alias in self.function.annotation_aliases.items()
                if name not in local_names
            }
        ):
            node.body = self.visit(node.body)
        return node

    def visit_ListComp(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp:
        # -------------------- Pattern --------------------
        # Python source:
        #     [f(n) for n in values]
        #
        # Builder:
        #     [I.with_at_group_(loc, lambda: f(n)) for n in values]
        # -------------------------------------------------
        # Comprehension binders mask annotation substitutions in Python evaluation order.
        with self._use_aliases(dict(self.function.annotation_aliases)):
            for generator in node.generators:
                generator.iter = self.visit(generator.iter)
                for target in ast.walk(generator.target):
                    if isinstance(target, ast.Name):
                        self.function.annotation_aliases.pop(target.id, None)
                generator.ifs = [self.visit(value) for value in generator.ifs]
            if isinstance(node, ast.DictComp):
                node.key, node.value = self.visit(node.key), self.visit(node.value)
            else:
                node.elt = self.visit(node.elt)
        return node

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_Constant(self, node: ast.Constant) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     42
        #
        # Builder:
        #     I.at_(loc, 42)
        # -------------------------------------------------
        return node if self.bypass_ast_rewrite else self._attach_span(node, node)

    def visit_List(self, node: ast.List | ast.Tuple | ast.Set | ast.Dict) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     [a, b]
        #
        # Builder:
        #     I.at_(loc, [I.at_(a_loc, a), I.at_(b_loc, b)])
        # -------------------------------------------------
        result = self.generic_visit(node)
        return result if self.bypass_ast_rewrite else self._attach_span(result, node)

    visit_Tuple = visit_List
    visit_Set = visit_List
    visit_Dict = visit_List

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.JoinedStr:
        # -------------------- Pattern --------------------
        # Python source:
        #     f"value={expr}"
        #
        # Builder:
        #     f"value={I.at_(loc, expr)}"
        # -------------------------------------------------
        # Literal fragments remain Constant/FormattedValue nodes required by Python.
        for child in node.values:
            if isinstance(child, ast.FormattedValue):
                child.value = self.visit(child.value)
                if child.format_spec is not None:
                    child.format_spec = self.visit_JoinedStr(child.format_spec)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     buffer[index]
        #
        # Builder:
        #     I.at_(loc, buffer[index])
        # -------------------------------------------------
        # Registered direct indexers retain their own result identity and source policy.
        direct = protocol.is_direct_call(self._resolve_constructor(node))
        result = self.generic_visit(node)
        if not direct and not self.bypass_ast_rewrite and isinstance(node.ctx, ast.Load):
            return self._attach_span(result, node)
        return result

    def _is_module_owner(self, node: ast.expr) -> bool:
        """Recognize fixed source module aliases from existing binding records."""
        if not isinstance(node, ast.Name):
            return False
        if node.id == self.module.module_name:
            return True
        records = [
            item
            for item in self.module.prescan.bindings.get(self.function.current_scope, ())
            if item.name == node.id
        ]
        return bool(records) and all(item.kind == "module_alias" for item in records)

    @contextmanager
    def _use_string_policy(self, policy: protocol.ExprStrPolicy | None) -> Iterator[None]:
        """Restore decoded-string dtype context after normal or failed traversal."""
        previous, self.expression_string = self.expression_string, policy
        try:
            yield
        finally:
            self.expression_string = previous

    def _rewrite_policy_argument(
        self,
        node: ast.expr,
        kind: str | None,
        policy: protocol.ExprStrPolicy,
        *,
        nested: bool = False,
    ) -> ast.expr:
        """Visit an argument once, assembling policy lookups after source children."""
        if kind == "expr_str" and isinstance(node, ast.Tuple | ast.List):
            # -------------------- Pattern --------------------
            # Python source:
            #     X.Tensor(("n", value))
            #
            # Builder:
            #     X.Tensor((X.resolve_type_var_("n"), value))
            # -------------------------------------------------
            node.elts = [
                self._rewrite_policy_argument(item, kind, policy, nested=True) for item in node.elts
            ]
            return node if self.bypass_ast_rewrite else self._attach_span(node, node)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if kind == "expr_str" and (nested or policy.scalar_strings):
                with self._use_string_policy(policy):
                    return self.visit(parse_expression_string(node, self.module.filename))
            if kind == "global_info":
                # -------------------- Pattern --------------------
                # Python source:
                #     X.Tensor(vdevice="cuda:1")
                #
                # Builder:
                #     X.Tensor(vdevice=I.resolve_global_info_("cuda:1"))
                # -------------------------------------------------
                value = self.visit(node)
                return self._call(
                    self.module.infrastructure_name, "resolve_global_info_", [value], node
                )
        return self.visit(node)

    def _visit_direct_operand(self, node: ast.expr) -> ast.expr:
        """Preserve an existing payload's span without bypassing child operations."""
        if isinstance(node, ast.Name):
            return (
                node
                if not self.annotation_expression and self.expression_string is None
                else self.visit(node)
            )
        if isinstance(node, ast.Attribute):
            node.value = self._visit_direct_operand(node.value)
            return node
        if isinstance(node, ast.Tuple | ast.List):
            node.elts = [self._visit_direct_operand(value) for value in node.elts]
            return node
        if isinstance(node, ast.Starred):
            node.value = self._visit_direct_operand(node.value)
            return node
        return self.visit(node)

    def visit_Call(self, node: ast.Call, *, callee: ast.expr | None = None) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     X.Tensor(("n",), vdevice="cuda:0")
        #
        # Builder:
        #     X.Tensor((X.resolve_type_var_("n"),), vdevice=I.resolve_global_info_("cuda:0"))
        # -------------------------------------------------
        marker = self._read_constexpr_operand(node)
        if marker is not None:
            with self._bypass_rewrite():
                return self.visit(marker)
        selected = None if self.bypass_ast_rewrite else handle_call_args_policy(node, self._resolve)
        constructor = self._resolve(node.func)
        direct = protocol.is_direct_call(constructor)
        global_call = (
            isinstance(node.func, ast.Name) and node.func.id in self.module.module_functions
        ) or (
            isinstance(node.func, ast.Attribute)
            and self._is_module_owner(node.func.value)
            and node.func.attr in self.module.module_functions
        )
        # Visit source callee/arguments first. A normalized range callee is
        # assembled afterward, but its original arguments keep normal rewriting.
        if callee is None:
            if self.expression_string is not None and constructor is not None:
                with self._use_string_policy(None):
                    node.func = self.visit(node.func)
            else:
                node.func = self.visit(node.func)
        if selected is None:
            visit_operand = self._visit_direct_operand if direct else self.visit
            node.args = [visit_operand(value) for value in node.args]
            for keyword in node.keywords:
                keyword.value = visit_operand(keyword.value)
        else:
            policy, parameters = selected
            known_position = True
            for index, value in enumerate(node.args):
                known_position = known_position and not isinstance(value, ast.Starred)
                name = parameters[index] if known_position and index < len(parameters) else None
                node.args[index] = self._rewrite_policy_argument(
                    value, policy.fields.get(name), policy.expression
                )
            for keyword in node.keywords:
                keyword.value = self._rewrite_policy_argument(
                    keyword.value, policy.fields.get(keyword.arg), policy.expression
                )
        if callee is not None:
            node.func = callee
        # -------------------- Pattern --------------------
        # Python source:
        #     declared_global(x, y)
        #
        # Builder:
        #     X.call_global_var_(declared_global, [x, y])
        # -------------------------------------------------
        if global_call and not self.bypass_ast_rewrite:
            if node.keywords:
                self._raise_error(node, "Global function calls require positional arguments")
            node = self._call(
                self.function.dialect_prefix,
                "call_global_var_",
                [node.func, ast.List(node.args, ast.Load())],
                node,
            )
        # -------------------- Pattern --------------------
        # Python source:
        #     f(a)
        #
        # Builder:
        #     I.with_at_group_(loc, lambda: f(a))
        # -------------------------------------------------
        # Registered direct calls omit this outer wrapper; nested calls keep their own scopes.
        if self.module.track_span and not direct:
            return self._call(
                self.module.infrastructure_name,
                "with_at_group_",
                [self.module.span(node), self._create_lambda([], node)],
                node,
            )
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     not x
        #
        # Builder:
        #     X.not_(x)
        # -------------------------------------------------
        # An explicit constexpr operand retains ordinary Python not.
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not) and not self.bypass_ast_rewrite:
            return self._attach_span(
                self._call(self.function.dialect_prefix, "not_", [node.operand], node), node
            )
        return self._attach_span(node, node) if not self.bypass_ast_rewrite else node

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     a + b
        #
        # Builder:
        #     I.at_(loc, a + b)
        # -------------------------------------------------
        return (
            self._attach_span(self.generic_visit(node), node)
            if not self.bypass_ast_rewrite
            else self.generic_visit(node)
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     I.constexpr(enabled) and expr
        #
        # Builder:
        #     enabled and expr
        # -------------------------------------------------
        # Unmarked IR operands use X.and_ or X.or_.
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)

        def lower(position: int) -> ast.expr:
            value = node.values[position]
            marker = self._read_constexpr_operand(value)
            if marker is not None:
                with self._bypass_rewrite():
                    left = self.visit(marker)
                if position + 1 == len(node.values):
                    return left
                return ast.copy_location(ast.BoolOp(node.op, [left, lower(position + 1)]), node)
            left = self.visit(value)
            method = "and_" if isinstance(node.op, ast.And) else "or_"
            for index in range(position + 1, len(node.values)):
                if self._read_constexpr_operand(node.values[index]) is not None:
                    return self._call(
                        self.function.dialect_prefix, method, [left, lower(index)], node
                    )
                left = self._call(
                    self.function.dialect_prefix,
                    method,
                    [left, self.visit(node.values[index])],
                    node,
                )
            return left

        return self._attach_span(lower(0), node)

    def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     yes if I.constexpr(test) else no
        #
        # Builder:
        #     yes if test else no
        # -------------------------------------------------
        # Unmarked tests use X.if_then_else_.
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        marker = self._read_constexpr_operand(node.test)
        if marker is not None:
            with self._bypass_rewrite():
                test = self.visit(marker)
            return ast.copy_location(
                ast.IfExp(test, self.visit(node.body), self.visit(node.orelse)), node
            )
        return self._attach_span(
            self._call(
                self.function.dialect_prefix,
                "if_then_else_",
                [self.visit(node.test), self.visit(node.body), self.visit(node.orelse)],
                node,
            ),
            node,
        )

    def _create_comparison(
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
            return self._call(
                self.function.dialect_prefix, operations[type(operation)], [left, right], node
            )
        return ast.copy_location(ast.Compare(left, [operation], [right]), node)

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     a < b < c
        #
        # Builder:
        #     (lambda x, y, z: X.and_(X.lt(x, y), X.lt(y, z), chain=(x, y, z)))(a, b, c)
        # -------------------------------------------------
        # Written operands evaluate once; native builders bind shared IR uses.
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        if len(node.ops) == 1:
            result = self._create_comparison(
                self.visit(node.left), node.ops[0], self.visit(node.comparators[0]), node
            )
        else:
            operands = [node.left, *node.comparators]
            names = [self.module.fresh("_operand") for _ in operands]
            comparisons = [
                self._create_comparison(
                    ast.Name(lhs, ast.Load()), op, ast.Name(rhs, ast.Load()), node
                )
                for lhs, op, rhs in zip(names, node.ops, names[1:])
            ]
            value = self._call(
                self.function.dialect_prefix,
                "and_",
                comparisons,
                node,
                chain=ast.Tuple([ast.Name(name, ast.Load()) for name in names], ast.Load()),
            )
            result = ast.copy_location(
                ast.Call(
                    self._create_lambda(names, value), [self.visit(value) for value in operands], []
                ),
                node,
            )
        return self._attach_span(result, node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> NoReturn:
        # -------------------- Pattern --------------------
        # Python source:
        #     (x := value)
        #
        # Builder:
        #     raise SyntaxError("Assignment expressions are unsupported")
        # -------------------------------------------------
        # Source assignment expressions have no builder declaration contract.
        self._raise_error(node, "Unsupported expression: NamedExpr")

    def visit_Await(self, node: ast.Await | ast.Yield | ast.YieldFrom) -> NoReturn:
        # -------------------- Pattern --------------------
        # Python source:
        #     await expression
        #
        # Builder:
        #     raise SyntaxError("Async and generator expressions are unsupported")
        # -------------------------------------------------
        self._raise_error(node, f"Unsupported expression: {type(node).__name__}")

    visit_Yield = visit_Await
    visit_YieldFrom = visit_Await

    def _rewrite_index(self, node: ast.expr) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     a[start:stop:step] = value
        #
        # Builder:
        #     X.setitem(a, slice(start, stop, step), value)
        # -------------------------------------------------
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
                ast.Tuple([self._rewrite_index(value) for value in node.elts], ast.Load()), node
            )
        return self.visit(node)

    def _read_call_kind(self, value: ast.expr | None) -> str | None:
        """Classify the original RHS against completed lexical binding facts."""
        if not isinstance(value, ast.Call | ast.Subscript):
            return None
        constructor = self._resolve_constructor(value)
        if protocol.is_scope_var_query_or_decl(constructor):
            return "scope_var_query_or_decl"
        if protocol.is_direct_call(constructor):
            return "direct_call"
        return "ordinary"

    def _bind(
        self,
        target: ast.expr,
        value: ast.expr,
        statement: ast.stmt,
        *,
        ty: ast.expr | None = None,
        frame_value: bool = False,
        call_kind: str | None = None,
    ) -> list[ast.stmt]:
        # Declaration syntax has precedence; no previous/existence tracking.
        if isinstance(target, ast.Name):
            site = self.module.prescan.sites.get(target) if self.module.prescan else None
            kind = site.kind if site is not None else "ordinary"
            if call_kind is not None and kind in (
                "ordinary",
                "direct_call",
                "scope_var_query_or_decl",
            ):
                kind = call_kind
            binding_declaration = kind == "binding_declaration" and (
                self._resolve(site.declaration_root) is not None
            )
            mutable = (
                self.module.prescan.mutable_names.get(self.function.current_scope, ())
                if self.module.prescan
                else ()
            )
            keywords = {"name": ast.Constant(target.id), "name_span": self.module.span(target)}
            if ty is not None:
                keywords["ty"] = ty
            if frame_value:
                keywords["frame_value"] = ast.Constant(True)
            if kind == "symbol" and not frame_value:
                # -------------------- Pattern --------------------
                # Python source:
                #     n = X.int64()
                #
                # Builder:
                #     n = X.resolve_type_var_("n", dtype="int64")
                # -------------------------------------------------
                value = self._call_dialect(
                    "resolve_type_var_",
                    [ast.Constant(target.id)],
                    target,
                    **({"dtype": ast.Constant(site.dtype)} if site.dtype else {}),
                )
            elif kind == "mutable" and not frame_value:
                # -------------------- Pattern --------------------
                # Python source:
                #     x = X.local_scalar(initial)
                #
                # Builder:
                #     x = X.decl_mutable_var_(X.local_scalar(initial), name="x")
                # -------------------------------------------------
                value = self._call_dialect("decl_mutable_var_", [value], statement, **keywords)
            elif kind == "scope_var_query_or_decl" and not frame_value:
                # -------------------- Pattern --------------------
                # Python source:
                #     i = X.axis.spatial(extent, value)
                #
                # Builder:
                #     i = X.scope_var_query_or_decl_(X.axis.spatial(extent, value), name="i")
                # -------------------------------------------------
                # Only the dialect naming/validation hook runs; it preserves the variable identity.
                value = self._call_dialect(
                    "scope_var_query_or_decl_", [value], statement, **keywords
                )
            elif kind in ("direct_call", "module_alias") and not frame_value:
                # -------------------- Pattern --------------------
                # Python source:
                #     value = I.meta_var(x)
                #     alias = Module
                #
                # Builder:
                #     value = I.meta_var(x)
                #     alias = Module
                # -------------------------------------------------
                pass
            elif target.id in mutable and not binding_declaration and not frame_value:
                # -------------------- Pattern --------------------
                # Python source:
                #     x = value
                #
                # Builder:
                #     X.set_mutable_var_(x, value)
                # -------------------------------------------------
                # The prescan identifies x as mutable.
                return [
                    ast.copy_location(
                        ast.Expr(
                            self._call_dialect(
                                "set_mutable_var_",
                                [ast.Name(target.id, ast.Load()), value],
                                statement,
                            )
                        ),
                        statement,
                    )
                ]
            else:
                # -------------------- Pattern --------------------
                # Python source:
                #     y = value
                #
                # Builder:
                #     y = X.bind_(value, name="y")
                # -------------------------------------------------
                value = self._call_dialect("bind_", [value], statement, **keywords)
            return [ast.copy_location(ast.Assign([target], value), statement)]
        if isinstance(target, ast.Attribute):
            # -------------------- Pattern --------------------
            # Python source:
            #     a.field = value
            #
            # Builder:
            #     X.setattr(a, "field", value)
            # -------------------------------------------------
            value = self._call_dialect(
                "setattr", [self.visit(target.value), ast.Constant(target.attr), value], statement
            )
            return [ast.copy_location(ast.Expr(value), statement)]
        if isinstance(target, ast.Subscript):
            # -------------------- Pattern --------------------
            # Python source:
            #     a[index] = value
            #
            # Builder:
            #     X.setitem(a, index, value)
            # -------------------------------------------------
            value = self._call_dialect(
                "setitem",
                [self.visit(target.value), self._rewrite_index(target.slice), value],
                statement,
            )
            return [ast.copy_location(ast.Expr(value), statement)]
        if isinstance(target, ast.Tuple | ast.List):
            # -------------------- Pattern --------------------
            # Python source:
            #     a, (b, c) = rhs
            #
            # Builder:
            #     first, second = X.unpack(rhs)
            #     a = X.bind_(first, name="a")
            #     left, right = X.unpack(second)
            #     b = X.bind_(left, name="b")
            #     c = X.bind_(right, name="c")
            # -------------------------------------------------
            names = [self.module.fresh("_unpack") for _ in target.elts]
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
                        self._call(self.function.dialect_prefix, "unpack", [value], target),
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
                        call_kind=call_kind,
                    )
                )
            return result
        self._raise_error(target, f"Unsupported assignment target: {type(target).__name__}")

    def visit_Assign(self, node: ast.Assign) -> ast.Assign | list[ast.stmt]:
        target: ast.expr
        # -------------------- Pattern --------------------
        # Python source:
        #     a = b = rhs
        #
        # Builder:
        #     temporary = rhs
        #     a = X.bind_(temporary, name="a")
        #     b = X.bind_(temporary, name="b")
        # -------------------------------------------------
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        call_kind = self._read_call_kind(node.value)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0]
            site = self.module.prescan.sites.get(target) if self.module.prescan else None
            value = ast.Constant(None) if site and site.kind == "symbol" else self.visit(node.value)
            return self._bind(target, value, node, call_kind=call_kind)
        temporary = self.module.fresh("_value")
        result: list[ast.stmt] = [self._assign(temporary, self.visit(node.value), node)]
        for target in node.targets:
            result.extend(
                self._bind(target, ast.Name(temporary, ast.Load()), node, call_kind=call_kind)
            )
        return result

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign | list[ast.stmt]:
        # -------------------- Pattern --------------------
        # Python source:
        #     x: X.int32 = value
        #
        # Builder:
        #     x = X.decl_mutable_var_(value, ty=X.int32, name="x")
        # -------------------------------------------------
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        if not isinstance(node.target, ast.Name):
            self._raise_error(node.target, "An annotated binding requires a name")
        call_kind = self._read_call_kind(node.value)
        value = (
            self.visit(node.value)
            if node.value
            else ast.Attribute(
                ast.Name(self.module.infrastructure_name, ast.Load()), "MISSING", ast.Load()
            )
        )
        with self._rewrite_annotation():
            annotation = self.visit(parse_annotation(node.annotation, self.module.filename))
        return self._bind(node.target, value, node, ty=annotation, call_kind=call_kind)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AugAssign | list[ast.stmt]:
        key: ast.expr
        load: ast.expr
        value: ast.expr
        # -------------------- Pattern --------------------
        # Python source:
        #     a[index] += value
        #
        # Builder:
        #     base = a
        #     key = index
        #     old = base[key]
        #     X.setitem(base, key, old + value)
        # -------------------------------------------------
        # Base, index, load and RHS each evaluate once; name targets use declaration dispatch.
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            value = ast.copy_location(
                ast.BinOp(ast.Name(node.target.id, ast.Load()), node.op, self.visit(node.value)),
                node,
            )
            return self._bind(node.target, self._attach_span(value, node), node)
        if not isinstance(node.target, ast.Subscript | ast.Attribute):
            self._raise_error(
                node.target, "An augmented assignment requires a name, attribute, or index"
            )
        base = self.module.fresh("_base")
        statements: list[ast.stmt] = [self._assign(base, self.visit(node.target.value), node)]
        if isinstance(node.target, ast.Attribute):
            key = ast.Constant(node.target.attr)
            load = ast.Attribute(ast.Name(base, ast.Load()), node.target.attr, ast.Load())
            operation = "setattr"
        else:
            index = self.module.fresh("_index")
            statements.append(self._assign(index, self._rewrite_index(node.target.slice), node))
            key = ast.Name(index, ast.Load())
            load = ast.Subscript(ast.Name(base, ast.Load()), key, ast.Load())
            operation = "setitem"
        old = self.module.fresh("_old")
        statements.append(self._assign(old, self._attach_span(load, node.target), node))
        value = self._attach_span(
            ast.BinOp(ast.Name(old, ast.Load()), node.op, self.visit(node.value)), node
        )
        statements.append(
            ast.copy_location(
                ast.Expr(
                    self._call_dialect(operation, [ast.Name(base, ast.Load()), key, value], node)
                ),
                node,
            )
        )
        return statements

    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     f()
        #
        # Builder:
        #     X.emit_(I.with_at_group_(loc, lambda: f()))
        # -------------------------------------------------
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        if protocol.is_direct_call(self._resolve_constructor(node.value)):
            # -------------------- Pattern --------------------
            # Python source:
            #     I.meta_var(value)
            #
            # Builder:
            #     I.meta_var(value)
            # -------------------------------------------------
            return ast.copy_location(ast.Expr(self.visit(node.value)), node)
        return ast.copy_location(
            ast.Expr(
                self._call(self.function.dialect_prefix, "emit_", [self.visit(node.value)], node)
            ),
            node,
        )

    def visit_Return(self, node: ast.Return) -> ast.Return | ast.Expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     return x
        #
        # Builder:
        #     X.return_(x)
        # -------------------------------------------------
        # Macro and bypass modes retain ordinary Python return.
        value = self.visit(node.value) if node.value else None
        if self.preserve_return or self.bypass_ast_rewrite:
            return ast.copy_location(ast.Return(value), node)
        return ast.copy_location(
            ast.Expr(self._call_dialect("return_", [] if value is None else [value], node)), node
        )

    def visit_Break(self, node: ast.Break) -> ast.Break | ast.Expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     break
        #
        # Builder:
        #     X.break_()
        # -------------------------------------------------
        return (
            node
            if self.bypass_ast_rewrite
            else ast.copy_location(ast.Expr(self._call_dialect("break_", [], node)), node)
        )

    def visit_Continue(self, node: ast.Continue) -> ast.Continue | ast.Expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     continue
        #
        # Builder:
        #     X.continue_()
        # -------------------------------------------------
        return (
            node
            if self.bypass_ast_rewrite
            else ast.copy_location(ast.Expr(self._call_dialect("continue_", [], node)), node)
        )

    def visit_Assert(self, node: ast.Assert) -> ast.Assert | ast.Expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     assert condition, message
        #
        # Builder:
        #     X.assert_(condition, message)
        # -------------------------------------------------
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        message = self.visit(node.msg) if node.msg else ast.Constant("")
        return ast.copy_location(
            ast.Expr(self._call_dialect("assert_", [self.visit(node.test), message], node)), node
        )

    def _rewrite_branch(self, body: list[ast.stmt], node: ast.AST, prefix: str) -> list[ast.stmt]:
        # Branch helper scoping keeps source names; mutable stores bind no locals.
        name = self.module.fresh(prefix)
        return [
            self._create_definition(name, self.transform_statements(body), node),
            ast.copy_location(ast.Expr(ast.Call(ast.Name(name, ast.Load()), [], [])), node),
        ]

    def visit_If(self, node: ast.If) -> ast.If | list[ast.stmt]:
        # -------------------- Pattern --------------------
        # Python source:
        #     if I.constexpr(flag):
        #         body()
        #
        # Builder:
        #     if flag:
        #         X.emit_(body())
        # -------------------------------------------------
        marker = self._read_constexpr_operand(node.test)
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        if marker is not None:
            with self._bypass_rewrite():
                condition = self.visit(marker)
            return ast.copy_location(
                ast.If(
                    condition,
                    self.transform_statements(node.body) or [ast.Pass()],
                    self.transform_statements(node.orelse),
                ),
                node,
            )
        # -------------------- Pattern --------------------
        # Python source:
        #     if condition:
        #         y = a
        #     else:
        #         y = b
        #
        # Builder:
        #     with X.if_(condition) as frame:
        #         with X.Then():
        #             def then():
        #                 y = X.bind_(a, name="y")
        #             then()
        #         with X.Else():
        #             def otherwise():
        #                 y = X.bind_(b, name="y")
        #             otherwise()
        #     y = frame.var
        # -------------------------------------------------
        frame = self.module.fresh("_conditional")
        then = (
            self.transform_statements(node.body)
            if self.preserve_return
            else self._rewrite_branch(node.body, node, "_then")
        )
        branches: list[ast.stmt] = [
            ast.copy_location(
                ast.With(
                    [ast.withitem(self._call_dialect("Then", [], node))], then or [ast.Pass()]
                ),
                node,
            )
        ]
        if node.orelse:
            otherwise = (
                self.transform_statements(node.orelse)
                if self.preserve_return
                else self._rewrite_branch(node.orelse, node, "_else")
            )
            branches.append(
                ast.copy_location(
                    ast.With(
                        [ast.withitem(self._call_dialect("Else", [], node))],
                        otherwise or [ast.Pass()],
                    ),
                    node,
                )
            )
        result: list[ast.stmt] = [
            ast.copy_location(
                ast.With(
                    [
                        ast.withitem(
                            self._call_dialect("if_", [self.visit(node.test)], node),
                            ast.Name(frame, ast.Store()),
                        )
                    ],
                    branches,
                ),
                node,
            )
        ]
        output = self.module.prescan.conditional_outputs.get(node) if self.module.prescan else None
        if output is not None:
            result.append(
                self._assign(
                    output, ast.Attribute(ast.Name(frame, ast.Load()), "var", ast.Load()), node
                )
            )
        return result

    def visit_For(self, node: ast.For) -> ast.For | ast.With:
        # -------------------- Pattern --------------------
        # Python source:
        #     for i, *tail in X.grid(m, n, k):
        #         body(i, tail)
        #
        # Builder:
        #     with X.for_(X.grid(m, n, k), names=("i", "*tail")) as (i, *tail):
        #         X.emit_(body(i, tail))
        # -------------------------------------------------
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        if node.orelse:
            self._raise_error(node, "A construction loop does not support an else clause")
        if isinstance(node.iter, ast.Call) and self._resolve(node.iter.func) is range:
            # Normalize only the known builtin; a lexical range binding follows
            # the ordinary source-call path. Arguments keep that one call scope.
            iterable = self.visit_Call(
                node.iter,
                callee=ast.Attribute(
                    ast.Name(self.function.dialect_prefix, ast.Load()), "range_", ast.Load()
                ),
            )
        else:
            iterable = self.visit(node.iter)
        if isinstance(node.target, ast.Name):
            names: ast.expr = ast.Tuple([ast.Constant(node.target.id)], ast.Load())
            node.target = ast.copy_location(ast.Tuple([node.target], ast.Store()), node.target)
        elif isinstance(node.target, ast.Tuple | ast.List):
            names = ast.Tuple(
                [
                    ast.Constant("*" + item.value.id if isinstance(item, ast.Starred) else item.id)
                    for item in node.target.elts
                ],
                ast.Load(),
            )
        else:
            self._raise_error(node.target, "Loop targets must be names or a flat tuple of names")
        context = self._call_dialect("for_", [iterable], node, names=names)
        body = self.transform_statements(node.body)
        return ast.copy_location(
            ast.With([ast.withitem(context, node.target)], body or [ast.Pass()]), node
        )

    def visit_While(self, node: ast.While) -> ast.While | ast.With:
        # -------------------- Pattern --------------------
        # Python source:
        #     while condition:
        #         body()
        #
        # Builder:
        #     with X.While(condition):
        #         X.emit_(body())
        # -------------------------------------------------
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        if node.orelse:
            self._raise_error(node, "A construction loop does not support an else clause")
        context = self._call_dialect("While", [self.visit(node.test)], node)
        return ast.copy_location(
            ast.With([ast.withitem(context)], self.transform_statements(node.body) or [ast.Pass()]),
            node,
        )

    def visit_With(self, node: ast.With) -> ast.With | list[ast.stmt]:
        # -------------------- Pattern --------------------
        # Python source:
        #     with X.block() as value:
        #         body(value)
        #
        # Builder:
        #     with X.block() as entered:
        #         value = X.bind_(entered, name="value", frame_value=True)
        #         X.emit_(body(value))
        # -------------------------------------------------
        # Ordinary regions use Python locals; native dataflow outputs retain identity.
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        item, rest = node.items[0], node.items[1:]
        body: list[ast.stmt] = (
            [ast.copy_location(ast.With(rest, node.body), node)] if rest else node.body
        )
        if item.optional_vars is None:
            target, initial = None, []
        else:
            entered = self.module.fresh("_entered")
            target = ast.Name(entered, ast.Store())
            initial = self._bind(
                item.optional_vars, ast.Name(entered, ast.Load()), node, frame_value=True
            )
        outputs = self.module.prescan.with_outputs.get(node, ()) if self.module.prescan else ()
        context = self.visit(item.context_expr)
        translated_body = initial + self.transform_statements(body) or [ast.Pass()]
        if not outputs:
            return ast.copy_location(
                ast.With([ast.withitem(context, target)], translated_body), node
            )
        # -------------------- Pattern --------------------
        # Python source:
        #     with X.dataflow():
        #         y = expression
        #         X.output(y)
        #
        # Builder:
        #     with X.dataflow() as frame:
        #         y = X.bind_(expression, name="y")
        #         X.output(y)
        #     y = frame.output_vars[0]
        # -------------------------------------------------
        # The native frame converts exports to ordinary output variables.
        frame = self.module.fresh("_dataflow")
        statements: list[ast.stmt] = [
            self._assign(frame, context, node),
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
            statements.append(self._assign(name, value, node))
        return statements

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef | list[ast.stmt]:
        # -------------------- Pattern --------------------
        # Python source:
        #     @X.function
        #     def nested():
        #         body()
        #
        # Builder:
        #     with X.function(decl=True, local=True) as frame:
        #         X.func_name("nested")
        #     nested = frame.reference
        #     with frame:
        #         def build():
        #             X.emit_(body())
        #         build()
        # -------------------------------------------------
        if self.bypass_ast_rewrite:
            return node
        kind, _ = self.read_function_metadata(node, allow_python=True)
        if kind.python:
            return node
        declaration, _, body = self.create_function_builder_fragments(node, local_function=True)
        return [*declaration, body]

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Pass:
        # -------------------- Pattern --------------------
        # Python source:
        #     nonlocal value
        #
        # Builder:
        #     pass
        # -------------------------------------------------
        # Source closure declarations do not mutate the host closure during build.
        return ast.copy_location(ast.Pass(), node)

    def visit_Pass(self, node: ast.Pass) -> ast.Pass:
        # -------------------- Pattern --------------------
        # Python source:
        #     pass
        #
        # Builder:
        #     pass
        # -------------------------------------------------
        return node

    def generic_visit(self, node: _Node) -> _Node:
        if isinstance(node, ast.stmt) and not self.bypass_ast_rewrite:
            self._raise_error(node, f"Unsupported statement: {type(node).__name__}")
        return super().generic_visit(node)

    def read_function_metadata(
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
                    self._raise_error(decorator, "Function decorators accept keyword options only")
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
        self._raise_error(node, f"Function {node.name!r} has no registered construction kind")

    def _read_function_annotations(
        self, node: ast.FunctionDef, parameters: list[ast.arg], facts: list[Binding]
    ) -> tuple[list[ast.expr | None], ast.expr | None, dict[str, str]]:
        """Find definition-scope names needed by signatures and body annotations."""
        declared_names = {item.name for item in getattr(node, "type_params", ())}
        # Quoted expression names are created later by argument normalization.
        annotations = [
            parse_annotation(parameter.annotation, self.module.filename)
            if parameter.annotation
            else None
            for parameter in parameters
        ]
        returns = parse_annotation(node.returns, self.module.filename) if node.returns else None
        annotation_names = {
            item.id
            for annotation in [*annotations, returns]
            if annotation is not None
            for item in ast.walk(annotation)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
        }
        body_annotations = [
            parse_annotation(item.annotation, self.module.filename)
            for item in facts
            if item.annotation is not None
            and item.kind not in ("parameter", "mutable_parameter", "symbol")
        ]
        annotation_names.update(
            item.id
            for annotation in body_annotations
            for item in ast.walk(annotation)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
        )
        aliases = {
            name: self.module.fresh("_annotation")
            for name in sorted(annotation_names - declared_names)
        }
        return annotations, returns, aliases

    def _create_definition_bindings(
        self, node: ast.FunctionDef, aliases: dict[str, str], *, captures: str, local_function: bool
    ) -> list[ast.stmt]:
        """Capture lexical annotation values without entering a construction frame."""
        # Inject builtin objects under fresh names: a source binding named globals,
        # locals, iter or next must not replace these generated operations.
        # Each function retains only annotation/constexpr names. Refer directly to
        # the single root scope; never copy globals or enclosing locals wholesale.
        names = set(aliases) | {
            parameter.arg
            for parameter in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            if self._is_constexpr_annotation(parameter.annotation)
        }
        values: list[ast.expr] = []
        for name in sorted(names):
            value: ast.expr = (
                self._inject(getattr(builtins, name))
                if name in aliases and hasattr(builtins, name)
                else ast.Attribute(
                    ast.Name(self.module.infrastructure_name, ast.Load()), "MISSING", ast.Load()
                )
            )
            scopes: list[ast.expr] = [ast.Call(self._inject(globals), [], [])]
            if self.module.definition_scope_name is not None and not local_function:
                scopes.append(ast.Name(self.module.definition_scope_name, ast.Load()))
            scopes.append(ast.Call(self._inject(locals), [], []))
            for scope in scopes:
                value = ast.Call(
                    ast.Attribute(scope, "get", ast.Load()), [ast.Constant(name), value], []
                )
            values.append(value)
        captures_expr = ast.Dict([ast.Constant(name) for name in sorted(names)], values)
        statements: list[ast.stmt] = [self._assign(captures, captures_expr, node)]
        for name, alias in aliases.items():
            fallback = (
                self._inject(getattr(builtins, name))
                if hasattr(builtins, name)
                else ast.Attribute(
                    ast.Name(self.module.infrastructure_name, ast.Load()), "MISSING", ast.Load()
                )
            )
            value = self._call(captures, "get", [ast.Constant(name), fallback], node)
            statements.append(self._assign(alias, value, node))
        return statements

    def _create_specialization_bindings(
        self, node: ast.FunctionDef, *, special: str, local_function: bool
    ) -> list[ast.stmt]:
        """Read root JIT inputs; nested functions keep ordinary runtime parameters."""
        from . import jit_support

        special_expr: ast.expr
        if local_function:
            special_expr = ast.Constant(None)
        else:
            special_expr = ast.Call(
                self._inject(jit_support.read_specialization_bindings),
                [ast.Constant(node.name)],
                [],
            )
        return [self._assign(special, special_expr, node)]

    def _create_symbol_declarations(
        self, node: ast.FunctionDef, facts: list[Binding]
    ) -> tuple[list[ast.stmt], dict[str, str]]:
        """Predeclare symbol types and bind explicit signature type parameters."""
        declaration: list[ast.stmt] = []
        symbol_aliases: dict[str, str] = {}
        # -------------------- Pattern --------------------
        # Python source:
        #     def f[n]():
        #         body(n)
        #
        # Builder:
        #     n = X.resolve_type_var_("n")
        # -------------------------------------------------
        # Explicit symbol dtypes precede quoted shapes; only explicit type
        # parameters bind signature names.
        for parameter in getattr(node, "type_params", ()):
            if not isinstance(parameter, getattr(ast, "TypeVar", ())):
                self._raise_error(parameter, "Only scalar type parameters are supported")
            bound = getattr(parameter, "bound", None)
            if bound is not None and self._resolve(bound) is not int:
                self._raise_error(parameter, "A symbolic type parameter bound must be int")
            if getattr(parameter, "default_value", None) is not None:
                self._raise_error(parameter, "A symbolic type parameter cannot have a default")
            alias = self.module.fresh("_symbol")
            symbol_aliases[parameter.name] = alias
            declaration.append(
                self._assign(
                    alias,
                    self._call_dialect(
                        "resolve_type_var_", [ast.Constant(parameter.name)], parameter
                    ),
                    parameter,
                )
            )
        for item in facts:
            if item.direct and item.dtype is not None and item.kind in ("symbol", "parameter"):
                symbol = self._call_dialect(
                    "resolve_type_var_",
                    [ast.Constant(item.name)],
                    item.node,
                    dtype=ast.Constant(item.dtype),
                )
                # A later Python parameter name does not enter annotation scope
                # until its own arg, even though the native map knows its dtype.
                declaration.append(ast.copy_location(ast.Expr(symbol), item.node))
        return declaration, symbol_aliases

    @staticmethod
    def _select_specialized_value(
        name: str, fallback: ast.expr, node: ast.AST, *, special: str
    ) -> ast.IfExp:
        """Select a JIT value or explicit absence without evaluating the fallback."""
        selected = ast.BoolOp(
            ast.And(),
            [
                ast.Compare(ast.Name(special, ast.Load()), [ast.IsNot()], [ast.Constant(None)]),
                ast.Compare(ast.Constant(name), [ast.In()], [ast.Name(special, ast.Load())]),
            ],
        )
        return ast.copy_location(
            ast.IfExp(
                selected,
                ast.Subscript(ast.Name(special, ast.Load()), ast.Constant(name), ast.Load()),
                fallback,
            ),
            node,
        )

    @staticmethod
    def _is_constexpr_annotation(annotation: ast.expr | None) -> bool:
        """Recognize the existing source-level constexpr annotation pattern."""
        return isinstance(annotation, ast.Attribute) and annotation.attr == "constexpr"

    def _rewrite_parameters(
        self,
        parameters: list[ast.arg],
        annotations: list[ast.expr | None],
        *,
        captures: str,
        special: str,
    ) -> tuple[list[ast.stmt], dict[str, str]]:
        """Declare signature parameters, making constexpr values available first."""
        from . import jit_support

        declaration: list[ast.stmt] = []
        constexpr_aliases: dict[str, str] = {}
        constexpr_params: list[tuple[ast.arg, ast.expr | None, bool]] = []
        other_params: list[tuple[ast.arg, ast.expr | None, bool]] = []
        # Runtime annotations may depend on a later constexpr parameter. Preserve
        # source order within each group and reuse this one syntactic classification.
        for parameter, annotation in zip(parameters, annotations):
            is_constexpr = self._is_constexpr_annotation(annotation)
            group = constexpr_params if is_constexpr else other_params
            group.append((parameter, annotation, is_constexpr))
        for parameter, annotation, is_constexpr in [*constexpr_params, *other_params]:
            if annotation is None:
                self._raise_error(parameter, f"Parameter {parameter.arg!r} requires an annotation")
            name = parameter.arg
            alias = self.module.fresh("_parameter")
            if is_constexpr:
                constexpr_aliases[name] = alias
                fallback = ast.Call(
                    self._inject(jit_support.require_constexpr_binding),
                    [
                        self._call(
                            captures,
                            "get",
                            [
                                ast.Constant(name),
                                ast.Attribute(
                                    ast.Name(self.module.infrastructure_name, ast.Load()),
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
                fallback = self._call_dialect("arg", [ast.Constant(name), checked], parameter)
            # Specialized parameters have no runtime ABI slot. The annotation
            # thunk is absent from the selected generated Python branch.
            value = self._select_specialized_value(name, fallback, parameter, special=special)
            declaration.append(self._assign(alias, value, parameter))
            self.function.annotation_aliases[name] = alias
        return declaration, constexpr_aliases

    def _create_function_frame(
        self,
        node: ast.FunctionDef,
        options: ast.Dict,
        declaration: list[ast.stmt],
        *,
        frame: str,
        local_function: bool,
        split_declare: bool,
    ) -> list[ast.stmt]:
        """Create the native frame and optionally execute its declaration pass."""
        # -------------------- Pattern --------------------
        # Python source:
        #     @X.function
        #     def f(x: X.int32):
        #         body(x)
        #
        # Builder:
        #     with X.function(decl=True) as frame:
        #         X.func_name("f")
        #         parameter = X.arg("x", X.int32)
        #     f = frame.reference
        # -------------------------------------------------
        keywords = [ast.keyword(None, options)]
        if split_declare:
            keywords.append(ast.keyword("decl", ast.Constant(True)))
        if local_function:
            keywords.append(ast.keyword("local", ast.Constant(True)))
        if self.module.track_span:
            keywords.append(ast.keyword("span", self.module.span(node)))
        constructor = ast.copy_location(
            ast.Call(
                ast.Attribute(
                    ast.Name(self.function.dialect_prefix, ast.Load()), "function", ast.Load()
                ),
                [],
                keywords,
            ),
            node,
        )
        if split_declare:
            declaration_scope = ast.With(
                [ast.withitem(constructor, ast.Name(frame, ast.Store()))], declaration
            )
            reference = ast.Attribute(ast.Name(frame, ast.Load()), "reference", ast.Load())
            return [
                ast.copy_location(declaration_scope, node),
                self._assign(node.name, reference, node),
            ]
        # Ordinary standalone functions enter once for their signature and body.
        return [self._assign(frame, constructor, node)]

    def _create_body_parameters(
        self,
        node: ast.FunctionDef,
        parameters: list[ast.arg],
        constexpr_aliases: dict[str, str],
        *,
        frame: str,
        special: str,
    ) -> list[ast.stmt]:
        """Read existing frame parameters into the body's Python lexical scope."""
        iterator = self.module.fresh("_arguments")
        body: list[ast.stmt] = [
            self._assign(
                iterator,
                ast.Call(
                    self._inject(iter),
                    [ast.Attribute(ast.Name(frame, ast.Load()), "params", ast.Load())],
                    [],
                ),
                node,
            )
        ]
        for parameter in parameters:
            name = parameter.arg
            value: ast.expr
            if name in constexpr_aliases:
                value = ast.Name(constexpr_aliases[name], ast.Load())
            else:
                value = self._select_specialized_value(
                    name,
                    ast.Call(self._inject(next), [ast.Name(iterator, ast.Load())], []),
                    parameter,
                    special=special,
                )
            body.append(self._assign(name, value, parameter))
        for parameter in getattr(node, "type_params", ()):
            body.append(
                self._assign(
                    parameter.name,
                    self._call_dialect(
                        "resolve_type_var_", [ast.Constant(parameter.name)], parameter
                    ),
                    parameter,
                )
            )
        return body

    def create_function_builder_fragments(
        self, node: ast.FunctionDef, *, local_function: bool = False, split_declare: bool = True
    ) -> tuple[list[ast.stmt], str, ast.With]:
        """Declare a native frame and emit a lexical body helper inside its scope.

        Definition aliases retain outer annotation values. Signature aliases add
        declared symbols and each preceding parameter; constexpr aliases retain
        compile-time values for the body. One alias map serves annotation syntax;
        ordinary body reads retain Python globals/closures. None owns native IR.
        """
        kind, options = self.read_function_metadata(node)
        builder = self.module.fresh("_X")
        self.module.bindings[builder] = kind.builder
        frame, body_name = self.module.fresh("_fn"), self.module.fresh("_build")
        old = self.function
        self.function = FunctionContext(node, builder)
        try:
            with self._use_aliases({}):
                parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                if node.args.vararg or node.args.kwarg:
                    self._raise_error(node, "IR signatures require ordinary named parameters")
                facts = self.module.prescan.bindings.get(node, []) if self.module.prescan else []
                annotations, returns, definition_aliases = self._read_function_annotations(
                    node, parameters, facts
                )
                self.function.annotation_aliases = definition_aliases
                captures = self.module.fresh("_definition")
                statements = self._create_definition_bindings(
                    node, definition_aliases, captures=captures, local_function=local_function
                )
                special = self.module.fresh("_specialization")
                statements.extend(
                    self._create_specialization_bindings(
                        node, special=special, local_function=local_function
                    )
                )
                declaration: list[ast.stmt] = [
                    ast.copy_location(
                        ast.Expr(self._call(builder, "func_name", [ast.Constant(node.name)], node)),
                        node,
                    )
                ]
                symbols, symbol_aliases = self._create_symbol_declarations(node, facts)
                declaration.extend(symbols)
                with (
                    self._use_aliases({**definition_aliases, **symbol_aliases}),
                    self._rewrite_annotation(),
                ):
                    arguments, constexpr_aliases = self._rewrite_parameters(
                        parameters, annotations, captures=captures, special=special
                    )
                    declaration.extend(arguments)
                    if returns is not None:
                        declaration.append(
                            ast.copy_location(
                                ast.Expr(
                                    self._call(
                                        builder,
                                        "func_ret_type",
                                        [self._create_lambda([], self.visit(returns))],
                                        returns,
                                    )
                                ),
                                returns,
                            )
                        )
                    definition_aliases.update(self.function.annotation_aliases)
                with self._bypass_rewrite():
                    options = self.visit(options)
                statements.extend(
                    self._create_function_frame(
                        node,
                        options,
                        declaration,
                        frame=frame,
                        local_function=local_function,
                        split_declare=split_declare,
                    )
                )
                body = self._create_body_parameters(
                    node, parameters, constexpr_aliases, frame=frame, special=special
                )
                body.extend(self.transform_statements(node.body))
                definition = self._create_definition(
                    body_name, body if split_declare else [*declaration, *body], node
                )
                if not local_function:
                    protected = (
                        {item.name for item in getattr(node, "type_params", ())}
                        | {node.name}
                        | set(self.module.module_functions)
                    )
                    if self.module.module_name:
                        protected.add(self.module.module_name)
                    self.module.body_sources.append((definition, node.name, protected))
                # -------------------- Pattern --------------------
                # Python source:
                #     def f():
                #         body()
                #
                # Builder:
                #     with frame:
                #         def build():
                #             X.emit_(body())
                #         build()
                # -------------------------------------------------
                # The helper owns lexical scope; the surrounding with owns frame
                # entry/exit and error unwinding.
                invocation = ast.copy_location(
                    ast.Expr(ast.Call(ast.Name(body_name, ast.Load()), [], [])), node
                )
                resumed = ast.copy_location(
                    ast.With([ast.withitem(ast.Name(frame, ast.Load()))], [definition, invocation]),
                    node,
                )
                return statements, frame, resumed
        finally:
            self.function = old

    def rewrite_module(
        self, tree: ast.Module, *, check_well_formed: bool = True
    ) -> tuple[ast.Module, str]:
        """Emit direct native module construction, declarations, then bodies."""
        root, prefix = tree.body[-1], tree.body[:-1]
        if not isinstance(root, ast.ClassDef | ast.FunctionDef):
            self._raise_error(root, "Source must contain one function or module class")
        is_module = isinstance(root, ast.ClassDef)
        members = root.body if is_module else [root]
        functions = [item for item in members if isinstance(item, ast.FunctionDef)]
        if len({item.name for item in functions}) != len(functions):
            self._raise_error(root, "Duplicate function declaration")
        self.module.module_name = root.name if is_module else None
        self.module.module_functions = frozenset(
            item.name for item in functions if not self.read_function_metadata(item)[0].python
        )
        split_declare = is_module or root in self.module.prescan.recursive_functions
        builder, result = self.module.fresh("_builder"), self.module.fresh("_result")
        body: list[ast.stmt] = []
        for function in functions if split_declare else ():
            if self.read_function_metadata(function)[0].python:
                continue
            body.append(
                self._assign(
                    function.name,
                    self._call(
                        self.module.infrastructure_name,
                        "reserve_function",
                        [ast.Constant(function.name)],
                        function,
                    ),
                    function,
                )
            )
        for member in members:
            if isinstance(member, ast.FunctionDef):
                continue
            # Source class statements are host setup inside the native module;
            # global-info registration therefore precedes dependent annotations.
            with self._bypass_rewrite():
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
                        self.module.infrastructure_name,
                        "module_member_",
                        [ast.Constant(target.id), ast.Name(target.id, ast.Load())],
                        target,
                    )
                    body.append(self._assign(target.id, value, target))
        definitions, frames, python_functions = [], [], []
        for function in functions:
            kind, _ = self.read_function_metadata(function)
            if kind.python:
                # -------------------- Pattern --------------------
                # Python source:
                #     @I.pyfunc
                #     def f(value):
                #         return value
                #
                # Builder:
                #     f = original_callable
                #     result.__pyfuncs__["f"] = f
                # -------------------------------------------------
                original = self.module.source_functions.get(function.name)
                if original is not None:
                    body.append(self._assign(function.name, self._inject(original), function))
                else:
                    function.decorator_list = []
                    body.append(function)
                python_functions.append((function.name, function.name))
                continue
            declaration, frame, definition = self.create_function_builder_fragments(
                function, split_declare=split_declare
            )
            body.extend(declaration)
            definitions.append(definition)
            frames.append(frame)
        body.extend(definitions)
        # -------------------- Pattern --------------------
        # Python source:
        #     class Module:
        #         @X.function
        #         def f():
        #             first()
        #         @X.function
        #         def g():
        #             second()
        #
        # Builder:
        #     with IRBuilder() as builder:
        #         with I.ir_module():
        #             with X.function(decl=True) as f_frame:
        #                 X.func_name("f")
        #             with X.function(decl=True) as g_frame:
        #                 X.func_name("g")
        #             with f_frame:
        #                 def build_f():
        #                     X.emit_(first())
        #                 build_f()
        #             with g_frame:
        #                 def build_g():
        #                     X.emit_(second())
        #                 build_g()
        #     result = builder.get()
        # -------------------------------------------------
        module = ast.copy_location(
            ast.With(
                [
                    ast.withitem(
                        self._call(self.module.infrastructure_name, "ir_module", [], root),
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
                        self._call(self.module.infrastructure_name, "IRBuilder", [], root),
                        ast.Name(builder, ast.Store()),
                    )
                ],
                [module] if split_declare else body,
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
                self._assign(result, output, root),
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
        if is_module:
            translated.append(
                ast.copy_location(
                    ast.Assign(
                        [ast.Attribute(ast.Name(result, ast.Load()), "__pyfuncs__", ast.Store())],
                        ast.Dict(
                            [ast.Constant(name) for name, _ in python_functions],
                            [ast.Name(alias, ast.Load()) for _, alias in python_functions],
                        ),
                    ),
                    root,
                )
            )
        if check_well_formed:
            # -------------------- Pattern --------------------
            # Python source:
            #     @X.function
            #     def f():
            #         body()
            #
            # Builder:
            #     result = builder.get()
            #     X.check_well_formed_(result)
            # -------------------------------------------------
            # Module syntax selects I.check_well_formed_ instead, after all bodies complete.
            namespace = (
                self.module.infrastructure_name
                if is_module
                else self._inject(self.read_function_metadata(root)[0].builder, "_X").id
            )
            translated.append(
                ast.copy_location(
                    ast.Expr(
                        self._call(
                            namespace, "check_well_formed_", [ast.Name(result, ast.Load())], root
                        )
                    ),
                    root,
                )
            )
        return ast.fix_missing_locations(ast.Module(translated, [])), result
