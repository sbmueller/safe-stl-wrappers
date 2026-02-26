// benchmark.cc — Compare std::optional vs safe::optional performance.
//
// Requires Google Benchmark (brew install google-benchmark).
//
// Build with asserts ON (debug — measures assert overhead):
//   c++ -std=c++17 -O2 -isystem $(brew --prefix google-benchmark)/include
//       -L$(brew --prefix google-benchmark)/lib benchmark.cc
//       -lbenchmark -lbenchmark_main -lpthread -o benchmark
//
// Build with asserts OFF (release — expected zero overhead):
//   c++ -std=c++17 -O2 -DNDEBUG -isystem $(brew --prefix google-benchmark)/include
//       -L$(brew --prefix google-benchmark)/lib benchmark.cc
//       -lbenchmark -lbenchmark_main -lpthread -o benchmark
//
// Run:
//   ./benchmark

#include <benchmark/benchmark.h>

#include <cassert>
#include <optional>
#include <string>
#include <utility>

// ---------------------------------------------------------------------------
// Inline reproduction of safe::optional from the poster so the benchmark is
// self-contained.  In a real project this would live in safe/optional.hpp.
// ---------------------------------------------------------------------------
namespace safe {

template <typename T>
class optional {
public:
    constexpr optional() noexcept = default;
    constexpr optional(std::nullopt_t) noexcept : m_base{std::nullopt} {}
    constexpr optional(const T& value) : m_base{value} {}
    constexpr optional(T&& value) : m_base{std::move(value)} {}

    template <typename... Args>
    constexpr explicit optional(std::in_place_t, Args&&... args)
        : m_base{std::in_place, std::forward<Args>(args)...} {}

    // Guarded accessors
    constexpr T& operator*() & noexcept {
        assert(has_value() && "UB prevented: dereferencing empty safe::optional");
        return *m_base;
    }
    constexpr const T& operator*() const& noexcept {
        assert(has_value() && "UB prevented: dereferencing empty safe::optional");
        return *m_base;
    }
    constexpr T&& operator*() && noexcept {
        assert(has_value() && "UB prevented: dereferencing empty safe::optional");
        return std::move(*m_base);
    }
    constexpr const T&& operator*() const&& noexcept {
        assert(has_value() && "UB prevented: dereferencing empty safe::optional");
        return std::move(*m_base);
    }

    constexpr T* operator->() noexcept {
        assert(has_value() && "UB prevented: member access on empty safe::optional");
        return m_base.operator->();
    }
    constexpr const T* operator->() const noexcept {
        assert(has_value() && "UB prevented: member access on empty safe::optional");
        return m_base.operator->();
    }

    // Safe forwards
    constexpr bool has_value() const noexcept { return m_base.has_value(); }
    constexpr explicit operator bool() const noexcept { return m_base.has_value(); }

    constexpr T& value() & { return m_base.value(); }
    constexpr const T& value() const& { return m_base.value(); }
    constexpr T&& value() && { return std::move(m_base).value(); }
    constexpr const T&& value() const&& { return std::move(m_base).value(); }

    template <typename U>
    constexpr T value_or(U&& fallback) const& {
        return m_base.value_or(std::forward<U>(fallback));
    }
    template <typename U>
    constexpr T value_or(U&& fallback) && {
        return std::move(m_base).value_or(std::forward<U>(fallback));
    }

    template <typename... Args>
    T& emplace(Args&&... args) {
        return m_base.emplace(std::forward<Args>(args)...);
    }

    void reset() noexcept { m_base.reset(); }

    // Comparisons — optional-to-optional
    friend constexpr bool operator==(const optional& a, const optional& b) { return a.m_base == b.m_base; }
    friend constexpr bool operator!=(const optional& a, const optional& b) { return a.m_base != b.m_base; }
    friend constexpr bool operator< (const optional& a, const optional& b) { return a.m_base <  b.m_base; }

    // Comparisons — optional-to-nullopt
    friend constexpr bool operator==(const optional& o, std::nullopt_t) noexcept { return !o.has_value(); }
    friend constexpr bool operator!=(const optional& o, std::nullopt_t) noexcept { return  o.has_value(); }

    // Comparisons — optional-to-value
    template <typename U>
    friend constexpr bool operator==(const optional& o, const U& v) { return o.m_base == v; }
    template <typename U>
    friend constexpr bool operator!=(const optional& o, const U& v) { return o.m_base != v; }

private:
    std::optional<T> m_base;
};

}  // namespace safe

// ---------------------------------------------------------------------------
// Each benchmark calls DoNotOptimize(opt) before the access under test.
// This forces the compiler to treat the optional's internal engaged state as
// potentially modified on every iteration, preventing it from proving the
// optional is always engaged and eliminating the precondition branch.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Benchmarks — operator*
// ---------------------------------------------------------------------------
static void BM_StdOptional_Deref(benchmark::State& state) {
    std::optional<int> opt{42};
    for (auto _ : state) {
        benchmark::DoNotOptimize(opt);
        auto v = *opt;
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_StdOptional_Deref);

static void BM_SafeOptional_Deref(benchmark::State& state) {
    safe::optional<int> opt{42};
    for (auto _ : state) {
        benchmark::DoNotOptimize(opt);
        auto v = *opt;
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_SafeOptional_Deref);

// ---------------------------------------------------------------------------
// Benchmarks — operator-> (with a struct member access)
// ---------------------------------------------------------------------------
struct Point {
    int x;
    int y;
};

static void BM_StdOptional_Arrow(benchmark::State& state) {
    std::optional<Point> opt{Point{1, 2}};
    for (auto _ : state) {
        benchmark::DoNotOptimize(opt);
        auto v = opt->x;
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_StdOptional_Arrow);

static void BM_SafeOptional_Arrow(benchmark::State& state) {
    safe::optional<Point> opt{Point{1, 2}};
    for (auto _ : state) {
        benchmark::DoNotOptimize(opt);
        auto v = opt->x;
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_SafeOptional_Arrow);

// ---------------------------------------------------------------------------
// Benchmarks — value()  (checked access, throws on empty)
// ---------------------------------------------------------------------------
static void BM_StdOptional_Value(benchmark::State& state) {
    std::optional<int> opt{42};
    for (auto _ : state) {
        benchmark::DoNotOptimize(opt);
        auto v = opt.value();
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_StdOptional_Value);

static void BM_SafeOptional_Value(benchmark::State& state) {
    safe::optional<int> opt{42};
    for (auto _ : state) {
        benchmark::DoNotOptimize(opt);
        auto v = opt.value();
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_SafeOptional_Value);

// ---------------------------------------------------------------------------
// Benchmarks — value_or()
// ---------------------------------------------------------------------------
static void BM_StdOptional_ValueOr(benchmark::State& state) {
    std::optional<int> opt{42};
    for (auto _ : state) {
        benchmark::DoNotOptimize(opt);
        auto v = opt.value_or(-1);
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_StdOptional_ValueOr);

static void BM_SafeOptional_ValueOr(benchmark::State& state) {
    safe::optional<int> opt{42};
    for (auto _ : state) {
        benchmark::DoNotOptimize(opt);
        auto v = opt.value_or(-1);
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_SafeOptional_ValueOr);

// ---------------------------------------------------------------------------
// Benchmarks — comparison (optional == optional)
// ---------------------------------------------------------------------------
static void BM_StdOptional_CmpEqual(benchmark::State& state) {
    std::optional<int> a{42};
    std::optional<int> b{42};
    for (auto _ : state) {
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        auto v = (a == b);
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_StdOptional_CmpEqual);

static void BM_SafeOptional_CmpEqual(benchmark::State& state) {
    safe::optional<int> a{42};
    safe::optional<int> b{42};
    for (auto _ : state) {
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        auto v = (a == b);
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_SafeOptional_CmpEqual);

// ---------------------------------------------------------------------------
// Benchmarks — comparison (optional == nullopt)
// ---------------------------------------------------------------------------
static void BM_StdOptional_CmpNullopt(benchmark::State& state) {
    std::optional<int> opt{42};
    for (auto _ : state) {
        benchmark::DoNotOptimize(opt);
        auto v = (opt == std::nullopt);
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_StdOptional_CmpNullopt);

static void BM_SafeOptional_CmpNullopt(benchmark::State& state) {
    safe::optional<int> opt{42};
    for (auto _ : state) {
        benchmark::DoNotOptimize(opt);
        auto v = (opt == std::nullopt);
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_SafeOptional_CmpNullopt);

// ---------------------------------------------------------------------------
// Benchmarks — operator* with a heap-owning type (std::string) to show the
// wrapper adds no overhead even when the contained type is non-trivial.
// ---------------------------------------------------------------------------
static void BM_StdOptional_Deref_String(benchmark::State& state) {
    std::optional<std::string> opt{"hello, hardened world!"};
    for (auto _ : state) {
        benchmark::DoNotOptimize(opt);
        auto& v = *opt;
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_StdOptional_Deref_String);

static void BM_SafeOptional_Deref_String(benchmark::State& state) {
    safe::optional<std::string> opt{"hello, hardened world!"};
    for (auto _ : state) {
        benchmark::DoNotOptimize(opt);
        auto& v = *opt;
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_SafeOptional_Deref_String);

// ---------------------------------------------------------------------------
// Benchmarks — emplace + reset cycle to measure mutation overhead.
// ---------------------------------------------------------------------------
static void BM_StdOptional_EmplaceReset(benchmark::State& state) {
    std::optional<int> opt;
    for (auto _ : state) {
        opt.emplace(42);
        benchmark::DoNotOptimize(opt);
        opt.reset();
    }
}
BENCHMARK(BM_StdOptional_EmplaceReset);

static void BM_SafeOptional_EmplaceReset(benchmark::State& state) {
    safe::optional<int> opt;
    for (auto _ : state) {
        opt.emplace(42);
        benchmark::DoNotOptimize(opt);
        opt.reset();
    }
}
BENCHMARK(BM_SafeOptional_EmplaceReset);

BENCHMARK_MAIN();
