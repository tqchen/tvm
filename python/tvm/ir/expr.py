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
"""Common expressions data structures in the IR."""

from numbers import Number

import tvm_ffi

import tvm

from ..runtime import DataTypeCode, Object, Scriptable, const
from . import _ffi_api
from .base import Node, Span


@tvm_ffi.register_object("ir.Expr")
class Expr(Node):
    """Base class of all the expressions."""

    span: Span | None
    ty: "tvm.ir.Type | None"


class _PrimExprMeta(type(Expr)):
    def __instancecheck__(cls, instance: object) -> bool:
        if cls is not PrimExpr:
            return super().__instancecheck__(instance)
        return super().__instancecheck__(instance) or (
            isinstance(instance, Expr)
            and isinstance(getattr(instance, "ty", None), tvm.ir.PrimType)
        )


@tvm_ffi.register_object("ir.PrimExpr")
class PrimExpr(Expr, metaclass=_PrimExprMeta):
    """Base class of all primitive expressions.

    PrimExpr is used in the low-level code
    optimizations and integer analysis.
    """


def _is_prim_expr(value: object) -> bool:
    return isinstance(value, PrimExpr) or (
        isinstance(value, Expr) and isinstance(value.ty, tvm.ir.PrimType)
    )


@tvm_ffi.register_object("ir.GlobalVar")
class GlobalVar(Expr):
    """A global variable in the IR.

    GlobalVar is used to refer to the global functions
    stored in the IRModule.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
    """

    name_hint: str

    def __init__(self, name_hint: str):
        self.__init_handle_by_constructor__(_ffi_api.GlobalVar, name_hint)

    def __call__(self, *args: Expr) -> Expr:
        """Call the global variable.

        Parameters
        ----------
        args: List[Expr]
            The arguments to the call.

        Returns
        -------
        call: Expr
            A call taking the variable as a function.
        """
        # pylint: disable=import-outside-toplevel

        if args and all(isinstance(x, Number) or _is_prim_expr(x) for x in args):
            return tvm.tirx.call_tir(self, *args)

        if all(isinstance(x, Expr) for x in args):
            from tvm import relax

            return relax.Call(self, args)

        arg_types = [type(x) for x in args]
        raise RuntimeError(f"Do not know how to handle GlobalVar.__call__ for types {arg_types}")


@tvm_ffi.register_object("ir.Call")
class Call(Expr, Scriptable):
    """Core function call node."""

    __hash__ = Expr.__hash__

    op: Expr
    args: list[Expr]
    attrs: "tvm.ir.Attrs | None"
    ty_args: list["tvm.ir.Type"]
    span: Span | None

    def __init__(
        self,
        ret_ty: "tvm.ir.Type | str | None",
        op: Expr | str,
        args: list[Expr] | tuple[Expr, ...],
        attrs: "tvm.ir.Attrs | dict | None" = None,
        ty_args: list["tvm.ir.Type"] | tuple["tvm.ir.Type", ...] | None = None,
        span: Span | None = None,
    ) -> None:
        # pylint: disable=import-outside-toplevel
        from .attrs import make_node
        from .op import Op
        from .type import PrimType, Type

        if isinstance(op, str):
            op = Op.get(op)
        if attrs is not None and isinstance(attrs, dict):
            attrs = make_node("ir.DictAttrs", **attrs)
        if ret_ty is None:
            ret_ty = PrimType("void")
        elif not isinstance(ret_ty, Type):
            ret_ty = PrimType(ret_ty)
        if isinstance(ret_ty, PrimType):
            from tvm import tirx

            args = [arg if isinstance(arg, Expr) else tirx.convert(arg) for arg in args]
        if ty_args is None:
            ty_args = []
        self.__init_handle_by_constructor__(_ffi_api.Call, ret_ty, op, args, attrs, ty_args, span)

    def expr_ty(self) -> "tvm.ir.PrimType":
        """Return this call's primitive result type."""
        from .type import PrimType

        if isinstance(self.ty, PrimType):
            return self.ty
        raise TypeError(f"Expected primitive-valued Call, but result type is {self.ty}")

    @staticmethod
    def _dtype_matches(value, code: int) -> bool:
        if code == DataTypeCode.INT and isinstance(value, int):
            return True
        if code == DataTypeCode.FLOAT and isinstance(value, float):
            return True
        if hasattr(value, "expr_ty"):
            try:
                return value.expr_ty().matches_code(code)
            except TypeError:
                return False
        ty = getattr(value, "ty", None)
        if isinstance(ty, tvm.ir.PrimType):
            return ty.matches_code(code)
        return False

    def _tirx_generic(self):
        self.expr_ty()
        from tvm.tirx import generic as _generic

        return _generic

    def _tirx_ffi_api(self):
        self.expr_ty()
        from tvm.tirx import _ffi_api as _tirx_ffi_api

        return _tirx_ffi_api

    def _relax_op(self, name: str):
        if isinstance(self.ty, tvm.ir.PrimType):
            raise TypeError(f"Expected non-primitive-valued Call, but result type is {self.ty}")
        from tvm.relax import op as _relax_op

        return getattr(_relax_op, name)

    def _relax_binary_op(self, other: Expr, name: str):
        if isinstance(other, Expr):
            return self._relax_op(name)(self, other)
        if isinstance(other, Number):
            raise TypeError(f"Please convert {other} with `const` first")
        raise TypeError(f"type {type(other)} not supported")

    @staticmethod
    def _relax_rhs_op(other: Expr):
        if isinstance(other, Number):
            raise TypeError(f"Please convert {other} with `const` first")
        raise TypeError(f"type {type(other)} not supported")

    def __add__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_binary_op(other, "add")
        return self._tirx_generic().add(self, other)

    def __radd__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self.__add__(other)
        return self._tirx_generic().add(other, self)

    def __sub__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_binary_op(other, "subtract")
        return self._tirx_generic().subtract(self, other)

    def __rsub__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_rhs_op(other)
        return self._tirx_generic().subtract(other, self)

    def __mul__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_binary_op(other, "multiply")
        return self._tirx_generic().multiply(self, other)

    def __rmul__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self.__mul__(other)
        return self._tirx_generic().multiply(other, self)

    def __div__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_binary_op(other, "divide")
        if self._dtype_matches(self, DataTypeCode.INT) and self._dtype_matches(
            other, DataTypeCode.INT
        ):
            raise tvm.tirx.expr.div_ambiguity_error()
        return self._tirx_generic().divide(self, other)

    def __rdiv__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_rhs_op(other)
        if self._dtype_matches(self, DataTypeCode.INT) and self._dtype_matches(
            other, DataTypeCode.INT
        ):
            raise tvm.tirx.expr.div_ambiguity_error()
        return self._tirx_generic().divide(other, self)

    def __truediv__(self, other: PrimExpr) -> PrimExpr:
        return self.__div__(other)

    def __rtruediv__(self, other: PrimExpr) -> PrimExpr:
        return self.__rdiv__(other)

    def __floordiv__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_binary_op(other, "floor_divide")
        return self._tirx_generic().floordiv(self, other)

    def __rfloordiv__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_rhs_op(other)
        return self._tirx_generic().floordiv(other, self, None)

    def __mod__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_binary_op(other, "mod")
        return self._tirx_ffi_api()._OpFloorMod(self, other, None)  # type: ignore

    def __rmod__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_rhs_op(other)
        return self._tirx_ffi_api()._OpFloorMod(other, self, None)  # type: ignore

    def __neg__(self) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_op("negative")(self)
        return self.__mul__(const(-1, self.expr_ty().dtype))

    def __lshift__(self, other: PrimExpr) -> PrimExpr:
        return self._tirx_ffi_api().left_shift(self, other, None)  # type: ignore

    def __rlshift__(self, other: PrimExpr) -> PrimExpr:
        return self._tirx_ffi_api().left_shift(other, self, None)  # type: ignore

    def __rshift__(self, other: PrimExpr) -> PrimExpr:
        return self._tirx_ffi_api().right_shift(self, other, None)  # type: ignore

    def __rrshift__(self, other: PrimExpr) -> PrimExpr:
        return self._tirx_ffi_api().right_shift(other, self, None)  # type: ignore

    def __and__(self, other: PrimExpr) -> PrimExpr:
        return self._tirx_ffi_api().bitwise_and(self, other, None)  # type: ignore

    def __rand__(self, other: PrimExpr) -> PrimExpr:
        return self._tirx_ffi_api().bitwise_and(other, self, None)  # type: ignore

    def __or__(self, other: PrimExpr) -> PrimExpr:
        return self._tirx_ffi_api().bitwise_or(self, other, None)  # type: ignore

    def __ror__(self, other: PrimExpr) -> PrimExpr:
        return self._tirx_ffi_api().bitwise_or(other, self, None)  # type: ignore

    def __xor__(self, other: PrimExpr) -> PrimExpr:
        return self._tirx_ffi_api().bitwise_xor(self, other, None)  # type: ignore

    def __rxor__(self, other: PrimExpr) -> PrimExpr:
        return self._tirx_ffi_api().bitwise_xor(other, self, None)  # type: ignore

    def __invert__(self) -> PrimExpr:
        if self._dtype_matches(self, DataTypeCode.FLOAT):
            raise RuntimeError("Cannot use ~ operator on float type Expr.")
        return self._tirx_ffi_api().bitwise_not(self, None)  # type: ignore

    def __lt__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_binary_op(other, "less")
        return self._tirx_ffi_api()._OpLT(self, other, None)  # type: ignore

    def __le__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_binary_op(other, "less_equal")
        return self._tirx_ffi_api()._OpLE(self, other, None)  # type: ignore

    def __eq__(self, other: PrimExpr) -> PrimExpr:
        if isinstance(self.ty, tvm.ir.PrimType):
            from tvm.tirx.expr import EqualOp

            return EqualOp(self, other)
        return Object.__eq__(self, other)

    def __ne__(self, other: PrimExpr) -> PrimExpr:
        if isinstance(self.ty, tvm.ir.PrimType):
            from tvm.tirx.expr import NotEqualOp

            return NotEqualOp(self, other)
        return Object.__ne__(self, other)

    def __gt__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_binary_op(other, "greater")
        return self._tirx_ffi_api()._OpGT(self, other, None)  # type: ignore

    def __ge__(self, other: PrimExpr) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_binary_op(other, "greater_equal")
        return self._tirx_ffi_api()._OpGE(self, other, None)  # type: ignore

    def __nonzero__(self):
        raise ValueError(
            "Cannot use and / or / not operator to Expr, hint: "
            + "use tvm.tirx.all / tvm.tirx.any instead"
        )

    def __bool__(self) -> bool:
        return self.__nonzero__()

    def equal(self, other: PrimExpr, span: Span | None = None) -> bool:
        return self._tirx_ffi_api()._OpEQ(self, other, span)  # type: ignore

    def astype(self, dtype: "str | tvm.ir.PrimType", span: Span | None = None) -> PrimExpr:
        if not isinstance(self.ty, tvm.ir.PrimType):
            return self._relax_op("astype")(self, dtype)
        return self._tirx_generic().cast(self, dtype, span)


@tvm_ffi.register_object("ir.Range")
class Range(Node, Scriptable):
    """Represent a range in TVM.

    You do not need to create a Range explicitly.
    Python lists and tuples will be converted automatically to a Range in API functions.

    Parameters
    ----------
    begin : PrimExpr
        The begin value of the range when end is None.
        Otherwise it is the length of the range.

    end : Optional[PrimExpr]
        The end value of the range.

    span : Optional[Span]
        The location of this node in the source code.

    Note
    ----
    The constructor creates the range `[begin, end)`
    if the end argument is not None. Otherwise, it creates `[0, begin)`.
    """

    min: PrimExpr
    extent: PrimExpr
    span: Span | None

    def __init__(
        self, begin: PrimExpr, end: PrimExpr | None = None, span: Span | None = None
    ) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Range, begin, end, span)

    @staticmethod
    def from_min_extent(min_value: PrimExpr, extent: PrimExpr, span: Span | None = None) -> "Range":
        """Construct a Range by min and extent.

        This constructs a range in [min_value, min_value + extent)

        Parameters
        ----------
        min_value : PrimExpr
            The minimum value of the range.

        extent : PrimExpr
            The extent of the range.

        span : Optional[Span]
            The location of this node in the source code.

        Returns
        -------
        rng : Range
            The constructed range.
        """
        return _ffi_api.Range_from_min_extent(min_value, extent, span)

    def __eq__(self, other: Object) -> bool:
        return tvm_ffi.structural_equal(self, other)

    def __ne__(self, other: Object) -> bool:
        return not self.__eq__(other)
