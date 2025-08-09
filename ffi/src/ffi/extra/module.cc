
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
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include "module_internal.h"

namespace tvm {
namespace ffi {

Optional<Function> ModuleObj::GetFunction(const String& name, bool query_imports) {
  if (auto opt_func = this->GetFunction(name)) {
    return opt_func;
  }
  if (query_imports) {
    for (const Any& import : imports_) {
      if (auto opt_func = import.cast<Module>()->GetFunction(name, query_imports)) {
        return *opt_func;
      }
    }
  }
  return std::nullopt;
}

bool ModuleObj::ImplementsFunction(const String& name, bool query_imports) {
  if (this->ImplementsFunction(name)) {
    return true;
  }
  if (query_imports) {
    for (const Any& import : imports_) {
      if (import.cast<Module>()->ImplementsFunction(name, query_imports)) {
        return true;
      }
    }
  }
  return false;
}

Module Module::LoadFromFile(const String& file_name) {
  String format = [&file_name]() -> String {
    const char* data = file_name.data();
    for (size_t i = file_name.size(); i > 0; i--) {
      if (data[i - 1] == '.') {
        return String(data + i, file_name.size() - i);
      }
    }
    TVM_FFI_THROW(RuntimeError) << "Failed to get file format from " << file_name;
    TVM_FFI_UNREACHABLE();
  }();

  if (format == "dll" || format == "dylib" || format == "dso") {
    format = "so";
  }
  String loader_name = "ffi.Module.load_from_file." + format;
  const auto floader = tvm::ffi::Function::GetGlobal(loader_name);
  if (!floader.has_value()) {
    TVM_FFI_THROW(RuntimeError) << "Loader for `." << format << "` files is not registered,"
                                << " resolved to (" << loader_name << ") in the global registry."
                                << "Ensure that you have loaded the correct runtime code, and"
                                << "that you are on the correct hardware architecture.";
  }
  return (*floader)(file_name).cast<Module>();
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ffi.load_module",
                        [](const String& file_name) { return Module::LoadFromFile(file_name); });
});
}  // namespace ffi
}  // namespace tvm

int TVMFFIEnvLookupFromImports(TVMFFIObjectHandle library_ctx, const char* func_name,
                               TVMFFIObjectHandle* out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  *out = tvm::ffi::ModuleObj::InternalUnsafe::GetFunctionFromImports(
      reinterpret_cast<tvm::ffi::ModuleObj*>(library_ctx), func_name);
  TVM_FFI_SAFE_CALL_END();
}
