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

/*!
 * \file benchmark_primtype_cache.cc
 * \brief Focused side benchmark for PrimType cache strategy.
 *
 * Build/run:
 * \code
 * g++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra \
 *   tests/scripts/benchmark_primtype_cache.cc \
 *   -o /tmp/benchmark_primtype_cache
 * /tmp/benchmark_primtype_cache
 * \endcode
 *
 * This intentionally uses a minimal PrimType-like object layout instead of
 * linking TVM.  It isolates the cost shape relevant to the refactor:
 * allocation, compact-key hash lookup, direct singleton lookup, and the
 * downstream IntImm constructor shape.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct DataTypeLike {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

constexpr DataTypeLike kInt32{0, 32, 1};

uint32_t CompactKey(DataTypeLike dtype) {
  return (static_cast<uint32_t>(dtype.code) << 24) | (static_cast<uint32_t>(dtype.bits) << 16) |
         static_cast<uint32_t>(dtype.lanes);
}

struct PrimTypeLikeNode {
  DataTypeLike dtype;
  std::atomic<uint32_t> ref_count{1};
};

struct OldIntImmNode {
  DataTypeLike dtype;
  int64_t value;
};

struct NewIntImmNode {
  PrimTypeLikeNode* ty;
  int64_t value;
};

template <typename T>
void DoNotOptimize(const T& value) {
#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" : : "g"(value) : "memory");
#else
  (void)value;
#endif
}

PrimTypeLikeNode* FreshPrimTypeNode(DataTypeLike dtype) { return new PrimTypeLikeNode{dtype, 1}; }

const std::unordered_map<uint32_t, std::unique_ptr<PrimTypeLikeNode>>& PrimTypeHashCache() {
  static const std::unordered_map<uint32_t, std::unique_ptr<PrimTypeLikeNode>> cache = [] {
    std::unordered_map<uint32_t, std::unique_ptr<PrimTypeLikeNode>> result;
    for (DataTypeLike dtype :
         {DataTypeLike{0, 8, 1}, DataTypeLike{0, 16, 1}, kInt32, DataTypeLike{0, 64, 1},
          DataTypeLike{1, 8, 1}, DataTypeLike{1, 16, 1}, DataTypeLike{1, 32, 1},
          DataTypeLike{2, 32, 1}, DataTypeLike{2, 64, 1}}) {
      auto node = std::make_unique<PrimTypeLikeNode>();
      node->dtype = dtype;
      node->ref_count.store(1, std::memory_order_relaxed);
      result.emplace(CompactKey(dtype), std::move(node));
    }
    return result;
  }();
  return cache;
}

PrimTypeLikeNode* HashCachedPrimTypeNode(DataTypeLike dtype) {
  const auto& cache = PrimTypeHashCache();
  auto it = cache.find(CompactKey(dtype));
  return it == cache.end() ? nullptr : it->second.get();
}

PrimTypeLikeNode* StaticInt32PrimTypeNode() {
  static PrimTypeLikeNode value{kInt32, 1};
  return &value;
}

void IncRef(PrimTypeLikeNode* node) { node->ref_count.fetch_add(1, std::memory_order_relaxed); }

void DecRef(PrimTypeLikeNode* node) { node->ref_count.fetch_sub(1, std::memory_order_relaxed); }

struct BenchResult {
  std::string name;
  uint64_t iterations;
  double ns_per_op;
};

template <typename F>
BenchResult Measure(std::string name, uint64_t iterations, F&& fn) {
  for (uint64_t i = 0; i < std::min<uint64_t>(iterations, 1000000); ++i) {
    fn(i);
  }

  double best_ns_per_op = 1e100;
  for (int repeat = 0; repeat < 7; ++repeat) {
    const auto start = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < iterations; ++i) {
      fn(i);
    }
    const auto end = std::chrono::steady_clock::now();
    const double ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    best_ns_per_op = std::min(best_ns_per_op, ns / static_cast<double>(iterations));
  }
  return {std::move(name), iterations, best_ns_per_op};
}

OldIntImmNode* MakeOldIntImm(DataTypeLike dtype, int64_t value) {
  return new OldIntImmNode{dtype, value};
}

NewIntImmNode* MakeNewIntImmWithFreshPrimType(DataTypeLike dtype, int64_t value) {
  return new NewIntImmNode{FreshPrimTypeNode(dtype), value};
}

NewIntImmNode* MakeNewIntImmWithHashCachedPrimType(DataTypeLike dtype, int64_t value) {
  PrimTypeLikeNode* ty = HashCachedPrimTypeNode(dtype);
  IncRef(ty);
  return new NewIntImmNode{ty, value};
}

NewIntImmNode* MakeNewIntImmWithStaticPrimType(int64_t value) {
  PrimTypeLikeNode* ty = StaticInt32PrimTypeNode();
  IncRef(ty);
  return new NewIntImmNode{ty, value};
}

void DeleteFreshNewIntImm(NewIntImmNode* node) {
  delete node->ty;
  delete node;
}

void DeleteCachedNewIntImm(NewIntImmNode* node) {
  DecRef(node->ty);
  delete node;
}

}  // namespace

int main() {
  std::vector<BenchResult> results;

  results.push_back(Measure("PrimType fresh node from DataType", 5000000, [](uint64_t i) {
    auto* node = FreshPrimTypeNode(kInt32);
    DoNotOptimize(node);
    delete node;
    DoNotOptimize(i);
  }));
  results.push_back(Measure("PrimType hash cache lookup by compact key", 50000000, [](uint64_t i) {
    auto* node = HashCachedPrimTypeNode(kInt32);
    DoNotOptimize(node);
    DoNotOptimize(i);
  }));
  results.push_back(Measure("PrimType hash lookup plus ref materialize", 20000000, [](uint64_t i) {
    auto* node = HashCachedPrimTypeNode(kInt32);
    IncRef(node);
    DoNotOptimize(node);
    DecRef(node);
    DoNotOptimize(i);
  }));
  results.push_back(Measure("PrimType direct static singleton lookup", 50000000, [](uint64_t i) {
    auto* node = StaticInt32PrimTypeNode();
    DoNotOptimize(node);
    DoNotOptimize(i);
  }));
  results.push_back(
      Measure("PrimType static singleton plus ref materialize", 20000000, [](uint64_t i) {
        auto* node = StaticInt32PrimTypeNode();
        IncRef(node);
        DoNotOptimize(node);
        DecRef(node);
        DoNotOptimize(i);
      }));
  results.push_back(Measure("IntImm old dtype-field shape", 5000000, [](uint64_t i) {
    auto* node = MakeOldIntImm(kInt32, static_cast<int64_t>(i));
    DoNotOptimize(node);
    delete node;
  }));
  results.push_back(Measure("IntImm current PrimType fresh shape", 3000000, [](uint64_t i) {
    auto* node = MakeNewIntImmWithFreshPrimType(kInt32, static_cast<int64_t>(i));
    DoNotOptimize(node);
    DeleteFreshNewIntImm(node);
  }));
  results.push_back(Measure("IntImm current PrimType hash cache shape", 5000000, [](uint64_t i) {
    auto* node = MakeNewIntImmWithHashCachedPrimType(kInt32, static_cast<int64_t>(i));
    DoNotOptimize(node);
    DeleteCachedNewIntImm(node);
  }));
  results.push_back(
      Measure("IntImm current PrimType static singleton shape", 5000000, [](uint64_t i) {
        auto* node = MakeNewIntImmWithStaticPrimType(static_cast<int64_t>(i));
        DoNotOptimize(node);
        DeleteCachedNewIntImm(node);
      }));

  std::cout << "| benchmark | iterations | ns/op |\n";
  std::cout << "|---|---:|---:|\n";
  for (const BenchResult& result : results) {
    std::cout << "| " << result.name << " | " << result.iterations << " | " << std::fixed
              << std::setprecision(2) << result.ns_per_op << " |\n";
  }
  return 0;
}
