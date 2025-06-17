
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
#include <gtest/gtest.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/memory.h>

#include "./testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TEST(Bench, InvokeN) {
  Function fecho = Function::FromPacked([](const AnyView* args, int32_t num_args, Any* rv) {
    *rv = args[0];
  });
  fecho(1);

  int n = 1000000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n; ++i) {
   fecho(i).cast<int>();
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "InvokeNStd: " << duration.count() / n << " seconds" << std::endl;
}

TEST(Bench, InvokeNStd) {
  std::function<int(int)> fecho = [](AnyView a) -> int {
    static int sum = 0;
    sum += a.cast<int>();
    return sum;
  };
  fecho(1);

  int n = 10000000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n; ++i) {
   fecho(i);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "InvokeNStd: " << duration.count() / n << " seconds" << std::endl;
}

TEST(Bench, ObjectRef) {
  String obj = "a";
  Any any = obj;
  int n = 10000000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n; ++i) {
   any.cast<ObjectRef>();
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "ObjectRef.cast: " << duration.count() / n << " seconds" << std::endl;
}

TEST(Bench, StringConcat) {
  String name = "a";
  int n = 10000000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n; ++i) {
    name + "a";
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "StringConcat: " << duration.count() / n << " seconds" << std::endl;
}


TEST(Bench, StdStringConcat) {
  std::string name = "a";
  int n = 10000000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n; ++i) {
    name + "a";
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "StringConcat: " << duration.count() / n << " seconds" << std::endl;
}
}