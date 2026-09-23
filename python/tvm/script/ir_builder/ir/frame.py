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
"""Package tvm.script.ir_builder.ir.frame"""

import re

from tvm_ffi import register_object as _register_object

from ..base import IRBuilderFrame


@_register_object("script.ir_builder.IRModuleFrame")
class IRModuleFrame(IRBuilderFrame):
    def resolve_global_info(self, content):
        """Resolve a map index or target-kind device ordinal in this module."""
        if not isinstance(content, str):
            return content
        match = re.fullmatch(r"([^\[\]]+)\[(\d+)\]", content)
        if match:
            name, index = match.groups()
            return self.global_infos[name][int(index)]
        selector = re.fullmatch(r"([^:\[\]]+)(?::(\d+)(?::([^:]+))?)?", content)
        if selector is None:
            raise ValueError(f"Invalid global-info reference: {content!r}")
        target, index, _scope = selector.groups()
        ordinal = int(index) if index is not None else 0
        devices = self.global_infos.get("vdevice", ())
        if target == "vdevice":
            return devices[ordinal]
        devices = [device for device in devices if device.target.kind.name == target]
        if ordinal >= len(devices):
            raise ValueError(f"Global-info device reference was not found: {content!r}")
        return devices[ordinal]
