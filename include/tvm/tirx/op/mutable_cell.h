/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*! \file tvm/tirx/op/mutable_cell.h
 *  \brief Initialized mutable local cell operations.
 */
#ifndef TVM_TIRX_OP_MUTABLE_CELL_H_
#define TVM_TIRX_OP_MUTABLE_CELL_H_

#include <tvm/ir/op.h>

namespace tvm::tirx {

/*! \brief Allocate a fresh cell of Call.ty, initialized by args[0].
 *  Only valid as the direct value of a Bind to a MutableCellType Var.
 */
TVM_DLL const Op& mutable_cell_alloc_op();

/*! \brief Read the element value of the cell Var in args[0]. */
TVM_DLL const Op& mutable_cell_load_op();

/*! \brief Assign args[1] to the cell Var in args[0]; returns void.
 *  Only valid as the direct value of an Evaluate statement.
 */
TVM_DLL const Op& mutable_cell_store_op();

}  // namespace tvm::tirx
#endif  // TVM_TIRX_OP_MUTABLE_CELL_H_
