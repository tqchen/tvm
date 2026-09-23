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
"""A recording mini-language for executing production-generated builder programs.

Values record which protocol hook the generated program called and its operands.
Their addition records only operand order; no type inference, folding or runtime
evaluation is implemented. Tests of expression semantics and span composition use
real common IR/Prim nodes. Frames record entry, parameters, outputs and identity;
they deliberately implement no TVM type system or parser.
Each test creates a fresh language so recorded effects cannot leak across cases.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace

from tvm.script.ir_builder import IRBuilder
from tvm.script.parser import entry
from tvm.script.parser import protocol_registry as registry


@dataclass(eq=False)
class Value:
    """An opaque protocol-call record, with operands and instrumentation locations."""

    op: str
    args: tuple = ()
    name: str = ""
    span: object = None

    def __add__(self, other):
        return Value("add", (self, other))

    def __radd__(self, other):
        return Value("add", (other, self))

    def __bool__(self):
        raise TypeError("an IR expression cannot select Python control flow")


@dataclass(eq=False)
class Function:
    name: str = ""
    params: list = field(default_factory=list)
    body: list = field(default_factory=list)
    ret_type: object = None


class Module(dict):
    """Named dummy module; normal dict lookup and function object identity."""


class Frame:
    """Native-style frame: declarations and body reuse the same params/result."""

    def __init__(self, language, kind, *, decl=False, values=(), span=None, local=False, **options):
        self.language, self.kind, self.decl = language, kind, decl
        self.values = values
        self.function = Function() if kind == "function" else None
        self.params = []
        self.type_var_map = {}
        self.reference = Value("global", (self,))
        self.result = None
        self.branches = []
        self.names = None
        self.span = None

    def __enter__(self):
        self.language.stack.append(self)
        self.language.events.append(("enter", self.kind, self.decl, self))
        if self.kind == "for":
            names = self.names
            names = (names,) if isinstance(names, str) else names
            variables = [Value("loop", (bound,), name) for name, bound in zip(names, self.values)]
            return variables
        return self

    def __exit__(self, error_type, error, traceback):
        assert self.language.stack.pop() is self
        self.language.events.append(("exit", self.kind, self.decl, self))
        if self.kind == "function":
            if self.decl:
                self.decl = False
            else:
                self.language.functions[self.function.name] = self.function
                self.language.result = self.function
        elif self.kind in ("then", "else"):
            self.language.stack[-1].branches.append(self.result)
        elif self.kind == "if":
            self.var = Value("if", (*self.values, *self.branches))
        elif self.kind == "module":
            self.language.result = Module(self.language.functions)
        return False

    def resolve_type_var(self, name, dtype=None, *, value=None, **kwargs):
        if name not in self.type_var_map:
            self.type_var_map[name] = (
                value if value is not None else Value("symbol", (dtype,), name)
            )
        return self.type_var_map[name]

    def __getitem__(self, name):
        return self.language.functions[name]

    def __getattr__(self, name):
        if self.kind == "module" and name in self.language.references:
            return self.language.references[name]
        raise AttributeError(name)


class RecordingSpanEntry:
    """Use the recording hooks for opaque dummy values, with a real fixed span."""

    def __init__(self, language, span):
        self.language, self.span = language, span

    @property
    def location(self):
        span = self.span
        return (span.source_name, span.line, span.end_line, span.column, span.end_column)

    def __call__(self, value):
        return self.language.I.at_(self.location, value)

    def ctx(self, thunk):
        return self.language.I.with_at_group_(self.location, thunk)


class Language:
    """One recording builder namespace X and shared infrastructure namespace I."""

    def __init__(self):
        self.events, self.stack, self.functions = [], [], Module()
        self.references = {}
        self.missing = object()
        self.result = None
        self.source_stack = []
        self.global_infos = {}
        self.I = SimpleNamespace(
            IRBuilder=self.context,
            ir_module=lambda: Frame(self, "module"),
            at_=self.at,
            with_at_group_=self.with_at_group,
            resolve_global_info_=self.resolve_global_info,
            reserve_function=self.reserve_function,
            module_member_=lambda name, value: value,
            require_defined=self.require_defined,
            annotation_value_=lambda name, value: value,
            MISSING=self.missing,
            check_well_formed_=lambda result: None,
            constexpr=registry.constexpr,
        )
        self.X = SimpleNamespace(
            supports_mutable_declarations=True,
            function=lambda **kwargs: Frame(self, "function", **kwargs),
            func_name=self.func_name,
            arg=self.arg,
            func_ret_type=self.func_ret_type,
            func_ret_value=self.func_ret_value,
            return_=lambda value, **kwargs: self.func_ret_value(value),
            resolve_type_var_=self.resolve_type_var,
            bind_=self.bind,
            check_well_formed_=lambda result: None,
            scope_var_query_or_decl_=lambda value, **kwargs: value,
            emit_=self.emit,
            decl_mutable_var_=self.decl_mutable,
            set_mutable_var_=self.set_mutable,
            call_global_var_=lambda function, args: Value("call", (function, *args)),
            range_=lambda *bounds, **kwargs: Frame(self, "for", values=(bounds,)),
            grid=lambda *bounds: Frame(self, "for", values=bounds),
            for_=self.for_frame,
            If=lambda condition, **kwargs: Frame(self, "if", values=(condition,)),
            if_=lambda condition, **kwargs: Frame(self, "if", values=(condition,)),
            Then=lambda **kwargs: Frame(self, "then"),
            Else=lambda **kwargs: Frame(self, "else"),
            While=lambda condition, **kwargs: Frame(self, "while", values=(condition,)),
            if_then_else_=lambda *args: Value("select", args),
            and_=lambda *args, **kwargs: Value("and", args),
            or_=lambda *args: Value("or", args),
            not_=lambda value: Value("not", (value,)),
            constexpr=registry.constexpr,
            value=lambda *args: Value("value", args),
            record=self.record,
        )
        for name in ("eq", "ne", "lt", "le", "gt", "ge"):

            def operation(*args, name=name):
                return Value(name, args)

            setattr(self.X, name, operation)
            setattr(self.X, name + "_", operation)
        self.X.script = entry.make_decorator(self.X)

        @registry.args_policy({"shape": "expr_str", "device": "global_info"}, scalar_strings=False)
        def tensor(shape=None, dtype="float32", device=None, placement="S[0]"):
            return Value("tensor", (shape, dtype, device, placement))

        def symbol(expr=None):
            return Value("symbol", (expr,))

        def cell(value=None):
            return Value("cell", (value,))

        self.X.tensor = tensor
        self.X.symbol = registry.register_type_var_decl(symbol, dtype="int64")
        self.X.cell = registry.register_mutable_var_decl(cell)

    @contextmanager
    def context(self):
        with IRBuilder():
            yield self

    def get(self):
        return self.result

    def frame(self):
        return next(frame for frame in reversed(self.stack) if frame.kind == "function")

    def reserve_function(self, name):
        return self.references.setdefault(name, Value("global", (name,)))

    def require_defined(self, value, name):
        if value is self.missing:
            raise NameError(name)
        return value

    def func_name(self, name):
        self.frame().function.name = name
        self.frame().reference = self.references.setdefault(name, Value("global", (name,)))
        self.events.append(("name", name))

    def arg(self, name, annotation, *, span=None, **kwargs):
        value = Value("arg", (annotation,), name)
        if span is not None:
            value = span(value)
        frame = self.frame()
        frame.params.append(value)
        frame.function.params.append(value)
        self.events.append(("arg", name, value))
        return value

    def func_ret_type(self, annotation):
        self.frame().function.ret_type = annotation() if callable(annotation) else annotation

    def func_ret_value(self, value):
        self.events.append(("return", value))
        self.frame().function.body.append(("return", value))

    def resolve_type_var(self, name, dtype=None, **kwargs):
        value = self.frame().resolve_type_var(name, dtype, **kwargs)
        self.events.append(("symbol", name, value))
        return value

    def resolve_global_info(self, name):
        self.events.append(("global_info", name))
        return self.global_infos[name]

    def bind(self, value, *, name=None, **kwargs):
        self.events.append(("bind", name, value))
        if self.stack[-1].kind in ("then", "else"):
            self.stack[-1].result = value
        return value

    def emit(self, value, *, span=None):
        if span is not None:
            value = span(value)
        self.events.append(("emit", value))
        self.frame().function.body.append(("emit", value))

    def record(self, value):
        self.events.append(("record", value))
        return value

    def decl_mutable(self, value=None, *, name=None, **kwargs):
        value = Value("cell", (value,), name)
        self.events.append(("declare", name, value))
        return value

    def set_mutable(self, variable, value, **kwargs):
        self.events.append(("set", variable, value))
        return variable

    def for_frame(self, frame, *, names=None, span=None, **kwargs):
        frame.names = names
        if span is not None:
            span(frame)
        self.events.append(("loop_names", names))
        return frame

    def at(self, location, value):
        if isinstance(value, Value | Frame):
            value.span = tuple([*self.source_stack, location])
        return value

    def with_at_group(self, location, thunk):
        self.source_stack.append(location)
        try:
            return self.at(location, thunk())
        finally:
            self.source_stack.pop()

    def parse(self, source, **captures):
        return entry.parse(source, extra_vars={"X": self.X, **captures}, filename="dummy.py")
