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

# Be compatible with older version of CMake
if(USE_WEBGPU_DAWN)
  set(DAWN_ENABLE_PIC        ON CACHE BOOL "Position-Independent-Code")
  set(DAWN_ENABLE_DESKTOP_GL OFF CACHE BOOL "OpenGL backend")
  set(DAWN_ENABLE_OPENGLES   OFF CACHE BOOL "OpenGL ES backend")
  set(DAWN_USE_X11 OFF CACHE BOOL "Dawn x11")
  set(DAWN_BUILD_SAMPLES    OFF CACHE BOOL "Dawn examples")
  set(TINT_BUILD_SAMPLES     OFF CACHE BOOL "Tint examples")
  set(TINT_BUILD_GLSL_WRITER OFF CACHE BOOL "OpenGL SL writer")
  if (WIN32)
    set(TINT_BUILD_MSL_WRITER  OFF CACHE BOOL "Metal SL writer")
    set(TINT_BUILD_SPV_WRITER  OFF CACHE BOOL "SPIR-V writer")
  elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(TINT_BUILD_HLSL_WRITER OFF CACHE BOOL "DirectX SL writer")
    set(TINT_BUILD_SPV_WRITER  OFF CACHE BOOL "SPIR-V writer")
  elseif(UNIX)
    set(TINT_BUILD_MSL_WRITER  OFF CACHE BOOL "Metal SL writer")
    set(TINT_BUILD_HLSL_WRITER OFF CACHE BOOL "DirectX SL writer")
  endif()
  set(DAWN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dawn)
  set(DAWN_PATH ${USE_WEBGPU_DAWN})
  add_subdirectory(${DAWN_PATH} dawn EXCLUDE_FROM_ALL)
  message(STATUS "Build with WebGPU native Dawn support")
  tvm_file_glob(GLOB WEBGPU_NATIVE_SRCS src/runtime/webgpu_native/*.cc)
  list(APPEND RUNTIME_SRCS ${WEBGPU_NATIVE_SRCS})
  list(APPEND TVM_RUNTIME_PRIVATE_INCLUDES ${DAWN_BINARY_DIR}/gen/include)
  list(APPEND TVM_RUNTIME_PRIVATE_INCLUDES ${DAWN_PATH}/include)
  message(STATUS "Include dir" ${TVM_RUNTIME_PRIVATE_INCLUDES})
  list(APPEND TVM_RUNTIME_LINKER_LIBS
    dawncpp
    dawn_proc
    dawn_common
    dawn_platform
    dawn_native
    dawn_utils
    dawn_internal_config)
endif(USE_WEBGPU_DAWN)
