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
"""IRBuilder for TIR"""

from tvm_ffi import register_object as _register_object

from tvm.script.ir_builder.base import IRBuilder, IRBuilderFrame, _resolve_type_var
from tvm.tirx import Buffer, Var

from . import _ffi_api


@_register_object("script.ir_builder.tirx.TIRFrame")
class TIRFrame(IRBuilderFrame): ...


@_register_object("script.ir_builder.tirx.PrimFuncFrame")
class PrimFuncFrame(TIRFrame):
    """Native function frame retaining signature, symbols and finalized results."""

    @property
    def params(self):
        """The native declared parameters, shared with the resumed body."""
        return self.args

    @property
    def reference(self):
        """The stable declared module or local reference."""
        return self.global_var

    def resolve_type_var(self, name, dtype=None, *, value=None, span=None):
        """Resolve a primitive symbol in this function's native map."""
        return _resolve_type_var(self, _ffi_api.ResolveTypeVar, name, dtype, value=value, span=span)


@_register_object("script.ir_builder.tirx.SSBlockFrame")
class SBlockFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.SBlockInitFrame")
class BlockInitFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.ForFrame")
class ForFrame(TIRFrame):
    def __enter__(self) -> Var | list[Var]:  # type: ignore[override]
        super().__enter__()
        variables = self.vars
        names = self.names
        if names is None:
            return variables if len(variables) > 1 else variables[0]
        if isinstance(names, str):
            targets = (
                [names] if len(variables) == 1 else [f"{names}_{i}" for i in range(len(variables))]
            )
        else:
            targets = list(names)
            for index, name in enumerate(targets):
                if name.startswith("*"):
                    count = len(variables) - len(targets) + 1
                    if count < 0:
                        raise ValueError("Loop target count differs from iteration dimensions")
                    targets[index : index + 1] = [f"{name[1:]}_{i}" for i in range(count)]
                    break
            if len(targets) != len(variables):
                raise ValueError("Loop target count differs from iteration dimensions")
        for name, variable in zip(targets, variables):
            IRBuilder.name(name, variable)
        return variables[0] if isinstance(names, str) and len(variables) == 1 else variables


@_register_object("script.ir_builder.tirx.AssertFrame")
class AssertFrame(TIRFrame): ...


class LetFrame(TIRFrame):
    def __enter__(self) -> Var:
        super().__enter__()
        return self.var


class AllocateFrame(TIRFrame):
    def __enter__(self) -> Buffer:
        super().__enter__()
        return self.buffer_var


@_register_object("script.ir_builder.tirx.AttrFrame")
class AttrFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.WhileFrame")
class WhileFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.IfFrame")
class IfFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.ThenFrame")
class ThenFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.ElseFrame")
class ElseFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.DeclBufferFrame")
class DeclBufferFrame(TIRFrame):
    def __enter__(self) -> Buffer:
        super().__enter__()
        return self.buffer


@_register_object("script.ir_builder.tirx.LaunchThreadFrame")
class LaunchThreadFrame(TIRFrame):
    def __enter__(self) -> Var:
        super().__enter__()
        return self.iter_var.var


@_register_object("script.ir_builder.tirx.AllocBufferFrame")
class AllocBufferFrame(TIRFrame):
    def __enter__(self) -> Buffer:
        super().__enter__()
        return self.buffer


@_register_object("script.ir_builder.tirx.HintFrame")
class HintFrame(TIRFrame): ...
