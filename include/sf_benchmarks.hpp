#ifndef SF_BENCHMARKS_HPP
#define SF_BENCHMARKS_HPP

#include <complex>
#include <functional>
#include <vectorclass.h>
#include <sctl.hpp>

typedef std::complex<double> cdouble;
typedef sctl::Vec<double, 4> sctl_dx4;
typedef sctl::Vec<double, 8> sctl_dx8;

typedef sctl::Vec<float, 8> sctl_fx8;
typedef sctl::Vec<float, 16> sctl_fx16;

typedef std::function<std::pair<cdouble, cdouble>(cdouble)> fun_cdx1_x2;

template <class VAL_T>
using multi_eval_func = std::function<void(const VAL_T *, VAL_T *, size_t)>;

template <class VAL_T, int VecLen, class F>
std::function<void(const VAL_T *, VAL_T *, size_t)> sctl_apply(const F &f) {
    static const auto fn = [f](const VAL_T *vals, VAL_T *res, size_t N) {
        using Vec = sctl::Vec<VAL_T, VecLen>;
        for (size_t i = 0; i < N; i += VecLen) {
            Vec v = Vec::LoadAligned(vals + i);
            f(v).StoreAligned(res + i);
        }
    };
    return fn;
}

template <class VEC_T, class VAL_T, class F>
std::function<void(const VAL_T *, VAL_T *, size_t)> vec_func_apply(const F &f) {
    static const auto fn = [f](const VAL_T *vals, VAL_T *res, size_t N) {
        for (size_t i = 0; i < N; i += VEC_T::size()) {
            VEC_T x;
            x.load_a(vals + i);
            VEC_T(f(x)).store_a(res + i);
        }
    };
    return fn;
}

template <class VAL_T, class F>
std::function<void(const VAL_T *, VAL_T *, size_t)> scalar_func_apply(const F &f) {
    static const auto fn = [f](const VAL_T *vals, VAL_T *res, size_t N) {
        for (size_t i = 0; i < N; i++)
            res[i] = f(vals[i]);
    };
    return fn;
}

class Params {
  public:
    std::pair<double, double> domain{0.0, 1.0};
};


#endif
