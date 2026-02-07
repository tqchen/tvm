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
"""S-TIR specific transformations."""
# pylint: disable=invalid-name, unsupported-binary-operation

from . import _ffi_api

def CanonicalizeLoop():
    """Canonicalize the loop to start from zero and use trivial step

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.CanonicalizeLoop()  # type: ignore
