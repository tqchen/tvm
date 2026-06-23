#include <tvm/arith/analyzer.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/var.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

template <typename T>
TVM_FFI_INLINE void DoNotOptimize(const T& value) {
#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" : : "g"(value) : "memory");
#else
  (void)value;
#endif
}

struct BenchResult {
  std::string task;
  uint64_t iterations;
  double ns_per_op;
};

template <typename F>
BenchResult MeasureImmediate(std::string task, uint64_t iterations, F&& task_func) {
  std::cerr << "running," << task << '\n';
  const uint64_t warmup = std::min<uint64_t>(iterations, 5000);
  for (uint64_t i = 0; i < warmup; ++i) {
    task_func(i);
  }

  double best_ns_per_op = 1e100;
  for (int repeat = 0; repeat < 5; ++repeat) {
    const auto start = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < iterations; ++i) {
      task_func(i);
    }
    const auto end = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(end - start).count();
    best_ns_per_op = std::min(best_ns_per_op, ns / static_cast<double>(iterations));
  }
  return {std::move(task), iterations, best_ns_per_op};
}

template <typename T, typename F>
BenchResult MeasureRetained(std::string task, uint64_t iterations, F&& task_func) {
  std::cerr << "running," << task << '\n';
  {
    std::vector<T> warmup_sink;
    warmup_sink.reserve(static_cast<size_t>(std::min<uint64_t>(iterations, 5000)));
    for (uint64_t i = 0; i < std::min<uint64_t>(iterations, 5000); ++i) {
      warmup_sink.push_back(task_func(i));
    }
    if (!warmup_sink.empty()) {
      DoNotOptimize(warmup_sink.back());
    }
  }

  double best_ns_per_op = 1e100;
  for (int repeat = 0; repeat < 5; ++repeat) {
    std::vector<T> sink;
    sink.reserve(static_cast<size_t>(iterations));
    const auto start = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < iterations; ++i) {
      sink.push_back(task_func(i));
    }
    const auto end = std::chrono::steady_clock::now();
    if (!sink.empty()) {
      DoNotOptimize(sink.back());
    }
    const double ns = std::chrono::duration<double, std::nano>(end - start).count();
    best_ns_per_op = std::min(best_ns_per_op, ns / static_cast<double>(iterations));
  }
  return {std::move(task), iterations, best_ns_per_op};
}

template <typename F>
BenchResult MeasureParallelImmediate(std::string task, uint64_t total_iterations, int num_threads,
                                     F&& task_func) {
  std::cerr << "running," << task << '\n';
  const uint64_t per_thread = std::max<uint64_t>(1, total_iterations / num_threads);
  total_iterations = per_thread * static_cast<uint64_t>(num_threads);

  double best_ns_per_op = 1e100;
  for (int repeat = 0; repeat < 3; ++repeat) {
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(num_threads));
    for (int tid = 0; tid < num_threads; ++tid) {
      workers.emplace_back([&, tid]() {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
        }
        for (uint64_t i = 0; i < per_thread; ++i) {
          task_func(static_cast<uint64_t>(tid) * per_thread + i);
        }
      });
    }
    while (ready.load(std::memory_order_acquire) != num_threads) {
    }
    const auto begin = std::chrono::steady_clock::now();
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) {
      worker.join();
    }
    const auto end = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(end - begin).count();
    best_ns_per_op = std::min(best_ns_per_op, ns / static_cast<double>(total_iterations));
  }
  return {std::move(task), total_iterations, best_ns_per_op};
}

template <typename T, typename F>
BenchResult MeasureParallelRetained(std::string task, uint64_t total_iterations, int num_threads,
                                    F&& task_func) {
  std::cerr << "running," << task << '\n';
  const uint64_t per_thread = std::max<uint64_t>(1, total_iterations / num_threads);
  total_iterations = per_thread * static_cast<uint64_t>(num_threads);

  double best_ns_per_op = 1e100;
  for (int repeat = 0; repeat < 3; ++repeat) {
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(num_threads));
    for (int tid = 0; tid < num_threads; ++tid) {
      workers.emplace_back([&, tid]() {
        std::vector<T> sink;
        sink.reserve(static_cast<size_t>(per_thread));
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
        }
        for (uint64_t i = 0; i < per_thread; ++i) {
          sink.push_back(task_func(static_cast<uint64_t>(tid) * per_thread + i));
        }
        if (!sink.empty()) {
          DoNotOptimize(sink.back());
        }
      });
    }
    while (ready.load(std::memory_order_acquire) != num_threads) {
    }
    const auto begin = std::chrono::steady_clock::now();
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) {
      worker.join();
    }
    const auto end = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(end - begin).count();
    best_ns_per_op = std::min(best_ns_per_op, ns / static_cast<double>(total_iterations));
  }
  return {std::move(task), total_iterations, best_ns_per_op};
}

void Emit(const std::vector<BenchResult>& results) {
  std::cout << "task,iterations,ns_per_op\n";
  for (const BenchResult& result : results) {
    std::cout << result.task << ',' << result.iterations << ',' << std::fixed
              << std::setprecision(2) << result.ns_per_op << '\n';
  }
}

#if defined(TVM_BENCH_OLD)
using BenchType = tvm::DataType;

BenchType IntType(int bits, int lanes = 1) { return tvm::DataType::Int(bits, lanes); }
BenchType FloatType(int bits, int lanes = 1) { return tvm::DataType::Float(bits, lanes); }
BenchType BoolType(int lanes = 1) { return tvm::DataType::Bool(lanes); }

TVM_FFI_INLINE bool MatchesI32Element(const tvm::PrimExpr& expr) {
  return expr.dtype().element_of() == tvm::DataType::Int(32);
}
#else
using BenchType = tvm::PrimType;

BenchType IntType(int bits, int lanes = 1) { return tvm::PrimType::Int(bits, lanes); }
BenchType FloatType(int bits, int lanes = 1) { return tvm::PrimType::Float(bits, lanes); }
BenchType BoolType(int lanes = 1) { return tvm::PrimType::Bool(lanes); }

TVM_FFI_INLINE bool MatchesI32Element(const tvm::PrimExpr& expr) {
  return expr.ty().MatchesElementType(kDLInt, 32);
}
#endif

TVM_FFI_INLINE tvm::PrimExpr MakeIntImm(BenchType ty, int64_t value) {
  return tvm::IntImm(ty, value);
}

TVM_FFI_INLINE tvm::tirx::Var MakeVar(std::string name, BenchType ty) {
  return tvm::tirx::Var(std::move(name), ty);
}

}  // namespace

int main(int argc, char** argv) {
  using tvm::PrimExpr;
  using tvm::tirx::Broadcast;

  int64_t scale = 1;
  if (argc >= 2) {
    scale = std::max<int64_t>(1, std::stoll(argv[1]));
  }

  const uint64_t type_iters = static_cast<uint64_t>(scale) * 5000000;
  const uint64_t retained_iters = static_cast<uint64_t>(scale) * 200000;
  const uint64_t construct_iters = static_cast<uint64_t>(scale) * 20000;
  const uint64_t simplifier_iters = static_cast<uint64_t>(scale) * 20000;
  const int mt_threads = 4;

  const BenchType i32_ty = IntType(32);
  const BenchType i64_ty = IntType(64);
  const BenchType f32_ty = FloatType(32);
  const BenchType bool_ty = BoolType();
  const BenchType vec_bool_ty = BoolType(4);
  const BenchType vec_i32_ty = IntType(32, 4);

  auto outer_i32 = MakeVar("outer_i32", i32_ty);
  auto inner_i32 = MakeVar("inner_i32", i32_ty);
  auto outer_i64 = MakeVar("outer_i64", i64_ty);
  auto inner_i64 = MakeVar("inner_i64", i64_ty);
  auto vec_flag = MakeVar("vec_flag", vec_bool_ty);
  auto vec_value = MakeVar("vec_value", vec_i32_ty);

  auto make_canonical_expr = [&]() {
    PrimExpr fused_i32 = outer_i32 * 128 + inner_i32;
    PrimExpr split_outer_i32 = tvm::floordiv(fused_i32, 128);
    PrimExpr split_inner_i32 = tvm::floormod(fused_i32, 128);
    PrimExpr fused_i64 = outer_i64 * tvm::IntImm::Int64(256) + inner_i64;
    PrimExpr split_outer_i64 = tvm::floordiv(fused_i64, tvm::IntImm::Int64(256));
    PrimExpr split_inner_i64 = tvm::floormod(fused_i64, tvm::IntImm::Int64(256));
    return (split_outer_i32 * 128 + split_inner_i32 - fused_i32) +
           (split_outer_i64 * tvm::IntImm::Int64(256) + split_inner_i64 - fused_i64);
  };

  auto make_rewrite_expr = [&]() {
    PrimExpr fused_i32 = outer_i32 * 128 + inner_i32;
    PrimExpr split_outer_i32 = tvm::floordiv(fused_i32, 128);
    PrimExpr split_inner_i32 = tvm::floormod(fused_i32, 128);
    PrimExpr fused_i64 = outer_i64 * tvm::IntImm::Int64(256) + inner_i64;
    PrimExpr split_outer_i64 = tvm::floordiv(fused_i64, tvm::IntImm::Int64(256));
    PrimExpr split_inner_i64 = tvm::floormod(fused_i64, tvm::IntImm::Int64(256));
    return ((split_outer_i32 * 128 + split_inner_i32) == fused_i32) &&
           ((split_outer_i64 * tvm::IntImm::Int64(256) + split_inner_i64) == fused_i64);
  };

  auto make_bool_vector_expr = [&]() {
    PrimExpr lane_expr = tvm::floormod(vec_value, 8);
    return vec_flag || (lane_expr == lane_expr);
  };

  PrimExpr canonical_expr = make_canonical_expr();
  PrimExpr rewrite_expr = make_rewrite_expr();
  PrimExpr bool_vector_expr = make_bool_vector_expr();
  PrimExpr vec_i32 = Broadcast(MakeIntImm(i32_ty, 1), 4);

  std::vector<BenchResult> results;

  results.push_back(MeasureImmediate("type-check element uncached", type_iters,
                                     [&](uint64_t i) {
                                       bool value = MatchesI32Element(vec_i32);
                                       DoNotOptimize(value);
                                       DoNotOptimize(i);
                                     }));
  results.push_back(MeasureImmediate("type object int32 immediate", type_iters,
                                     [&](uint64_t i) {
                                       BenchType value_ty = IntType(32);
                                       DoNotOptimize(value_ty);
                                       DoNotOptimize(i);
                                     }));
  results.push_back(MeasureImmediate("type object int64 immediate", type_iters,
                                     [&](uint64_t i) {
                                       BenchType value_ty = IntType(64);
                                       DoNotOptimize(value_ty);
                                       DoNotOptimize(i);
                                     }));
  results.push_back(MeasureImmediate("type object float32 immediate", type_iters,
                                     [&](uint64_t i) {
                                       BenchType value_ty = FloatType(32);
                                       DoNotOptimize(value_ty);
                                       DoNotOptimize(i);
                                     }));
  results.push_back(MeasureImmediate("type object bool immediate", type_iters,
                                     [&](uint64_t i) {
                                       BenchType value_ty = BoolType();
                                       DoNotOptimize(value_ty);
                                       DoNotOptimize(i);
                                     }));
  results.push_back(MeasureImmediate("type object vector int32x4 immediate", type_iters,
                                     [&](uint64_t i) {
                                       BenchType value_ty = IntType(32, 4);
                                       DoNotOptimize(value_ty);
                                       DoNotOptimize(i);
                                     }));
#if !defined(TVM_BENCH_OLD)
  results.push_back(MeasureImmediate("type object raw dtype immediate", type_iters,
                                     [&](uint64_t i) {
                                       tvm::PrimType value_ty(DLDataType{kDLInt, 32, 1});
                                       DoNotOptimize(value_ty);
                                       DoNotOptimize(i);
                                     }));
#endif
  results.push_back(MeasureRetained<BenchType>("type object int32 retained", retained_iters,
                                               [&](uint64_t i) {
                                                 DoNotOptimize(i);
                                                 return IntType(32);
                                               }));
  results.push_back(MeasureParallelImmediate("type object int32 immediate mt4", type_iters,
                                             mt_threads, [&](uint64_t i) {
                                               BenchType value_ty = IntType(32);
                                               DoNotOptimize(value_ty);
                                               DoNotOptimize(i);
                                             }));
  results.push_back(MeasureParallelRetained<BenchType>("type object int32 retained mt4",
                                                       retained_iters, mt_threads,
                                                       [&](uint64_t i) {
                                                         DoNotOptimize(i);
                                                         return IntType(32);
                                                       }));

  results.push_back(MeasureImmediate("construct canonical split-fuse input", construct_iters,
                                     [&](uint64_t i) {
                                       PrimExpr value = make_canonical_expr();
                                       DoNotOptimize(value.get());
                                       DoNotOptimize(i);
                                     }));
  results.push_back(MeasureImmediate("construct rewrite split-fuse input", construct_iters,
                                     [&](uint64_t i) {
                                       PrimExpr value = make_rewrite_expr();
                                       DoNotOptimize(value.get());
                                       DoNotOptimize(i);
                                     }));
  results.push_back(MeasureImmediate("construct rewrite bool-vector input", construct_iters,
                                     [&](uint64_t i) {
                                       PrimExpr value = make_bool_vector_expr();
                                       DoNotOptimize(value.get());
                                       DoNotOptimize(i);
                                     }));
  results.push_back(MeasureImmediate("canonical split-fuse simplify", simplifier_iters,
                                     [&](uint64_t i) {
                                       tvm::arith::Analyzer analyzer;
                                       PrimExpr value = analyzer->canonical_simplify(canonical_expr);
                                       DoNotOptimize(value.get());
                                       DoNotOptimize(i);
                                     }));
  results.push_back(MeasureImmediate("rewrite split-fuse simplify", simplifier_iters,
                                     [&](uint64_t i) {
                                       tvm::arith::Analyzer analyzer;
                                       PrimExpr value = analyzer->rewrite_simplify(rewrite_expr);
                                       DoNotOptimize(value.get());
                                       DoNotOptimize(i);
                                     }));
  results.push_back(MeasureImmediate("rewrite bool-vector simplify", simplifier_iters,
                                     [&](uint64_t i) {
                                       tvm::arith::Analyzer analyzer;
                                       PrimExpr value = analyzer->rewrite_simplify(bool_vector_expr);
                                       DoNotOptimize(value.get());
                                       DoNotOptimize(i);
                                     }));

  Emit(results);
  return 0;
}
