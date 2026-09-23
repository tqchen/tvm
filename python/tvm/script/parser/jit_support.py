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
"""Scoped transport of already selected JIT inputs into builder execution.

JIT owns validation, defaults and caching. These contexts carry only the root
function's specialization and explicit parameter absences, restoring the prior
values after nested parsing or an exception. Native builder frames own no JIT
state, and ordinary parsing does not import the TIRx JIT entry point.
"""

from contextlib import contextmanager
from contextvars import ContextVar

# Per-execution inputs, never a second owner of native function or parameter state.
_SPECIALIZATION = ContextVar("tvm_parser_specialization", default=None)
_ABSENT_PARAMETERS = ContextVar("tvm_parser_absent_parameters", default=None)


@contextmanager
def specialization_context(name, bindings):
    """Pass selected JIT bindings to one root builder execution.

    ``None`` denotes ordinary parsing; an empty mapping is a specialization
    with no compile-time values. Copy inputs so nested execution cannot alter
    the caller's selection.
    """
    token = _SPECIALIZATION.set(None if bindings is None else (name, dict(bindings)))
    try:
        yield
    finally:
        _SPECIALIZATION.reset(token)


@contextmanager
def absent_parameters(name, parameters):
    """Transport explicitly absent root parameters while preserving None.

    The legacy entry input maps each selected parameter name to None. No
    absence is inferred from captures, and a non-None map value is invalid.
    """
    parameters = dict(parameters or {})
    if any(value is not None for value in parameters.values()):
        raise TypeError("absent_params values must be None")
    token = _ABSENT_PARAMETERS.set((name, parameters))
    try:
        yield
    finally:
        _ABSENT_PARAMETERS.reset(token)


def specialization_bindings(name):
    """Read the current root's bindings, or None outside specialization."""
    context = _SPECIALIZATION.get()
    return context[1] if context is not None and context[0] == name else None


def absent_parameter_names(name):
    """Read explicit root absences without treating missing bindings as None."""
    context = _ABSENT_PARAMETERS.get()
    return frozenset(context[1]) if context is not None and context[0] == name else frozenset()


def unwrap_annotation(annotation, specialization):
    """Read an optional runtime annotation only within a JIT specialization.

    Generated argument construction calls this after checking selected values
    and absences, so omitted parameters never evaluate their annotation.
    """
    unwrap = getattr(annotation, "__tvm_optional_annotation__", None)
    if unwrap is not None:
        if specialization is None:
            raise TypeError("T.Optional is only supported by @T.jit")
        return unwrap()
    return annotation


def constexpr_binding(value, name):
    """Require an explicit captured value for a constexpr parameter."""
    from tvm.script.ir_builder.base import MISSING

    if value is MISSING:
        raise TypeError(f"constexpr parameter {name!r} requires a specialization binding")
    return value
