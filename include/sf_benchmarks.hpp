#ifndef SF_BENCHMARKS_HPP
#define SF_BENCHMARKS_HPP

#include <complex>
#include <functional>
#include <sctl.hpp>
#include <sys/cdefs.h>
#include <vectorclass.h>

// Attempt to force non-aliased pointers actually seems to slow things down...
//#define RESTRICT __restrict
#define RESTRICT

typedef std::complex<double> cdouble;
typedef sctl::Vec<double, 4> sctl_dx4;
typedef sctl::Vec<double, 8> sctl_dx8;

typedef sctl::Vec<float, 8> sctl_fx8;
typedef sctl::Vec<float, 16> sctl_fx16;

typedef std::function<std::pair<cdouble, cdouble>(cdouble)> fun_cdx1_x2;

template <class VAL_T>
using multi_eval_func = std::function<void(const VAL_T *, VAL_T *, size_t)>;

template <class VAL_T, int VecLen, class F>
std::function<void(const VAL_T *RESTRICT, VAL_T *RESTRICT, size_t)> sctl_apply(const F &f) {
    static const auto fn = [f](const VAL_T *RESTRICT vals, VAL_T *RESTRICT res, size_t N) {
        using Vec = sctl::Vec<VAL_T, VecLen>;
        for (size_t i = 0; i < N; i += VecLen) {
            f(Vec::LoadAligned(vals + i)).StoreAligned(res + i);
        }
    };
    return fn;
}

template <class VEC_T, class VAL_T, class F>
std::function<void(const VAL_T *RESTRICT, VAL_T *RESTRICT, size_t)> vec_func_apply(const F &f) {
    static const auto fn = [f](const VAL_T *RESTRICT vals, VAL_T *RESTRICT res, size_t N) {
        for (size_t i = 0; i < N; i += VEC_T::size()) {
            f(VEC_T().load_a(vals + i)).store_a(res + i);
        }
    };
    return fn;
}

template <class VAL_T, class F>
std::function<void(const VAL_T *RESTRICT, VAL_T *RESTRICT, size_t)> scalar_func_apply(const F &f) {
    static const auto fn = [f](const VAL_T *RESTRICT vals, VAL_T *RESTRICT res, size_t N) {
        for (size_t i = 0; i < N; i += 1) {
            res[i] = f(vals[i]);
        }
    };
    return fn;
}

struct configuration_t {
    int id;
    std::string func;
    std::string ftype;
    double lbound = 0.0;
    double ubound = 1.0;
    double ilbound = 0.0;
    double iubound = 0.0;
};

#undef RESTRICT
#endif
