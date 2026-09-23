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
"""Syntax-only rewriting of registered constructor argument policies.

Existing AST nodes retain all four Python location fields. Parsed strings use a
UTF-8 byte-offset map through the original literal spelling, including escapes
and physical newlines. If a literal cannot be recovered (for example a synthetic
AST or adjacent concatenated literals), ranges conservatively fall back to the
literal's complete original range. Rewriting never evaluates source expressions.
"""

from __future__ import annotations

import ast
import inspect
import linecache
import re
from collections.abc import Callable

from . import protocol


class _LiteralParser:
    # filename is immutable for one expression rewrite and owns no source/IR cache.
    def __init__(self, filename: str) -> None:
        self.filename: str = filename

    def _string_expression(self, node: ast.Constant) -> ast.expr:
        try:
            expression = ast.parse(node.value, mode="eval").body
        except SyntaxError as error:
            raise SyntaxError(f"Invalid annotation expression: {error.msg}") from error
        # Example: in R.Tensor(("n + 1",), "float32"), the generated resolve
        # call for n retains the byte range of n inside the quoted literal, not
        # the whole constructor. A triple-quoted physical newline advances lineno;
        # an escaped \n advances decoded input but maps back to the escape's real
        # source bytes. Generated call wrappers copy these four mapped fields.
        source = "".join(linecache.getlines(self.filename))
        literal = ast.get_source_segment(source, node) if source else None
        positions = self._literal_positions(literal, node) if literal else None
        lines = node.value.splitlines(keepends=True)
        for inner in ast.walk(expression):
            if not hasattr(inner, "lineno"):
                continue
            for line_field, column_field in (
                ("lineno", "col_offset"),
                ("end_lineno", "end_col_offset"),
            ):
                line, column = getattr(inner, line_field), getattr(inner, column_field)
                offset = len("".join(lines[: line - 1]).encode("utf-8")) + column
                if positions is not None and offset in positions:
                    line, column = positions[offset]
                else:
                    # Without a reliable spelling map (synthetic or concatenated
                    # literals), preserve the literal's entire real source range.
                    # Never invent physical lines from decoded escape sequences.
                    line = getattr(node, line_field)
                    column = getattr(node, column_field)
                setattr(inner, line_field, line)
                setattr(inner, column_field, column)
        return expression

    @staticmethod
    def _literal_positions(literal: str, node: ast.Constant) -> dict[int, tuple[int, int]] | None:
        """Map decoded expression byte offsets back through the literal's escapes."""
        match = re.match("(?i:([rub]*))([\"'])", literal)
        if match is None:
            return None
        prefix, quote = match.groups()
        width = 3 if literal[len(prefix) :].startswith(quote * 3) else 1
        start, stop = len(prefix) + width, len(literal) - width
        delimiter = quote * width
        positions, decoded, offset = {}, "", 0
        index = start

        def location(raw_index: int) -> tuple[int, int]:
            before = literal[:raw_index]
            line = node.lineno + before.count("\n")
            column = len(before.rsplit("\n", 1)[-1].encode("utf-8"))
            return line, column + (node.col_offset if line == node.lineno else 0)

        while index < stop:
            end = index + 1
            if literal[index] == "\\" and "r" not in prefix.lower():
                escape = re.match(
                    r"\\(?:N\{[^}]*\}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|x[0-9a-fA-F]{2}|[0-7]{1,3}|\r?\n|.)",
                    literal[index:stop],
                )
                if escape:
                    end = index + len(escape.group())
            piece = literal[index:end]
            try:
                value = ast.literal_eval(prefix + delimiter + piece + delimiter)
            except (SyntaxError, ValueError):
                return None
            positions[offset] = location(index)
            for char in value:
                offset += len(char.encode("utf-8"))
                positions[offset] = location(end)
            decoded += value
            index = end
        return positions if decoded == node.value else None


def parse_annotation(node: ast.expr, filename: str) -> ast.expr:
    """Decode a quoted whole annotation without interpreting its Python names."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _LiteralParser(filename)._string_expression(node)
    return node


def handle_call_args_policy(
    node: ast.Call, resolve: Callable[[ast.expr], object], infrastructure: str, filename: str
) -> ast.Call:
    """Normalize this call's marked literal arguments before normal visitation.

    This local preorder step neither visits ordinary arguments nor copies the
    owned tree. The main visitor handles the normalized children exactly once.
    String-derived names carry syntax provenance until ``visit_Name`` emits the
    dialect's symbol lookup; they never introduce Python bindings.
    """
    constructor = resolve(node.func)
    policy = protocol.get_args_policy(constructor)
    if policy is None:
        return node
    literals = _LiteralParser(filename)

    class Symbols(ast.NodeTransformer):
        # Only the newly parsed argument string is visited here. This is literal
        # normalization, not a second pass over source expressions.
        def visit_Attribute(self, current: ast.Attribute) -> ast.Attribute:
            if resolve(current.value) is not None:
                return current
            return self.generic_visit(current)

        def visit_Call(self, current: ast.Call) -> ast.Call:
            if resolve(current.func) is None:
                current.func = self.visit(current.func)
            current.args = [self.visit(value) for value in current.args]
            for keyword in current.keywords:
                keyword.value = self.visit(keyword.value)
            return current

        def visit_Name(self, current: ast.Name) -> ast.Name:
            current._tvm_quoted_symbol = policy.expression.dtype
            return current

    def expression_field(current: ast.expr, nested: bool = False) -> ast.expr:
        if isinstance(current, ast.Tuple | ast.List):
            current.elts = [expression_field(value, True) for value in current.elts]
        elif isinstance(current, ast.Constant) and isinstance(current.value, str):
            if nested or policy.expression.scalar_strings:
                return Symbols().visit(literals._string_expression(current))
        return current

    def argument(current: ast.expr, name: str | None) -> ast.expr:
        kind = policy.fields.get(name)
        if kind == "expr_str":
            return expression_field(current)
        if (
            kind == "global_info"
            and isinstance(current, ast.Constant)
            and isinstance(current.value, str)
        ):
            # Source: X.Tensor(shape, vdevice="cuda:1")
            # Builder: X.Tensor(shape, vdevice=I.resolve_global_info("cuda:1"))
            result = ast.copy_location(
                ast.Call(
                    ast.Attribute(
                        ast.Name(infrastructure, ast.Load()), "resolve_global_info", ast.Load()
                    ),
                    [current],
                    [],
                ),
                current,
            )
            result._tvm_intrinsic = True
            return result
        return current

    parameters = [
        p.name
        for p in inspect.signature(constructor).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    known_position = True
    for index, value in enumerate(node.args):
        known_position = known_position and not isinstance(value, ast.Starred)
        name = parameters[index] if known_position and index < len(parameters) else None
        node.args[index] = argument(value, name)
    for keyword in node.keywords:
        keyword.value = argument(keyword.value, keyword.arg)
    return node
