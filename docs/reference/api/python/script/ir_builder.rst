..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

tvm.script.ir_builder
---------------------

Imperative source locations
***************************

Use :func:`tvm.script.ir_builder.line_trace` inside an active builder to attach
executed Python source lines to imperative construction. Both statements in the
with-body and user helpers participate; helper returns restore the caller line.
Native emission and deferred construction frames consume the same location
context as explicit ``at_`` annotations and ``with_loc`` scopes.

.. code-block:: python

    from tvm.script import ir_builder
    from tvm.tirx.script import ir_builder as T

    with ir_builder.IRBuilder() as builder:
        with ir_builder.line_trace():
            T.evaluate(1)
    statement = builder.get()

Locations cover executed lines, without operand or expression column precision.
Tracing skips synthetic filenames, TVM/tvm-ffi implementation files, and the
Python standard library. Other packages are considered user helpers. The context
is synchronous and thread-local: do not yield or await across it. Existing trace
hooks are suspended during construction and restored on exit, so debugger and
coverage interoperability is not provided. See ``line_trace`` below for the
complete filtering and nesting contract.

tvm.script.ir_builder
*********************
.. automodule:: tvm.script.ir_builder
   :members:
   :imported-members:
   :exclude-members: Call, DataTypeImm, FuncType, GenericConst, MissingType, Op, PrimType, Range, StringImm, StringType, TensorRegion, TensorRegionType, Tuple, Type, Var

tvm.relax.script.ir_builder.distributed
***************************************
.. automodule:: tvm.relax.script.ir_builder.distributed
   :members:
