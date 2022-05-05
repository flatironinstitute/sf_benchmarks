#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <toml.hpp>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#include <Eigen/Core>
#include <baobzi.hpp>
#include <boost/math/special_functions.hpp>
#include <gsl/gsl_sf.h>
#include <sctl.hpp>
#include <sleef.h>
#include <unsupported/Eigen/SpecialFunctions>
#include <vectorclass.h>
#include <vectormath_exp.h>
#include <vectormath_hyp.h>
#include <vectormath_trig.h>

#include <dlfcn.h>
#include <gnu/libc-version.h>
#include <time.h>

struct timespec get_wtime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

double get_wtime_diff(const struct timespec *ts, const struct timespec *tf) {
    return (tf->tv_sec - ts->tv_sec) + (tf->tv_nsec - ts->tv_nsec) * 1E-9;
}

class Params {
  public:
    std::pair<double, double> domain{0.0, 1.0};
};

typedef std::complex<double> cdouble;
typedef sctl::Vec<double, 4> sctl_dx4;
typedef sctl::Vec<double, 8> sctl_dx8;

typedef std::function<double(double)> fun_dx1;
typedef std::function<cdouble(cdouble)> fun_cdx1;
typedef std::function<std::pair<cdouble, cdouble>(cdouble)> fun_cdx1_x2;

typedef std::function<void(const double *, double *, size_t)> vec_function;
template <class Real>
using scalar_function = std::function<void(const Real *, Real *, size_t)>;

template <class Real, int VecLen, class F>
std::function<void(const double *, double *, size_t)> sctl_apply(const F &f) {
    static const auto fn = [f](const Real *vals, Real *res, size_t N) {
        using Vec = sctl::Vec<Real, VecLen>;
        for (size_t i = 0; i < N; i += VecLen) {
            Vec v = Vec::LoadAligned(vals + i);
            f(v).StoreAligned(res + i);
        }
    };
    return fn;
}

template <class VEC_T, class Real, class F>
std::function<void(const double *, double *, size_t)> vec_func_apply(const F &f) {
    static const auto fn = [f](const Real *vals, Real *res, size_t N) {
        for (size_t i = 0; i < N; i += VEC_T::size()) {
            VEC_T x;
            x.load_a(vals + i);
            VEC_T y = f(x);
            y.store_a(res + i);
        }
    };
    return fn;
}

template <class Real, class F>
std::function<void(const Real *, Real *, size_t)> scalar_func_apply(const F &f) {
    static const auto fn = [f](const Real *vals, Real *res, size_t N) {
        for (size_t i = 0; i < N; i++)
            res[i] = f(vals[i]);
    };
    return fn;
}

extern "C" {
void hank103_(double _Complex *, double _Complex *, double _Complex *, int *);
void fort_bessel_jn_(int *, double *, double *);
void fort_bessel_yn_(int *, double *, double *);
}

template <typename VAL_T>
class BenchResult {
  public:
    Eigen::VectorX<VAL_T> res;
    double eval_time = 0.0;
    std::string label;
    std::size_t n_evals;
    Params params;

    BenchResult(const std::string &label_) : label(label_){};
    BenchResult(const std::string &label_, std::size_t size, std::size_t n_evals_, Params params_)
        : res(size), label(label_), n_evals(n_evals_), params(params_){};

    VAL_T &operator[](int i) { return res[i]; }
    double Mevals() const { return n_evals / eval_time / 1E6; }

    template <typename T>
    friend std::ostream &operator<<(std::ostream &, const BenchResult<T> &);
};

template <typename VAL_T>
std::ostream &operator<<(std::ostream &os, const BenchResult<VAL_T> &br) {
    VAL_T mean = 0.0;
    for (const auto &v : br.res)
        mean += v;
    mean /= br.res.size();

    using std::left;
    using std::setw;
    if (br.res.size()) {
        os.precision(6);
        os << left << setw(25) << br.label + ": " << left << setw(15) << br.Mevals();
        os.precision(15);
        os << left << setw(15) << mean << left << setw(5) << " ";
        os.precision(5);
        os << "[" << br.params.domain.first << ", " << br.params.domain.second << "]" << std::endl;
    }
    return os;
}

template <typename VAL_T>
Eigen::VectorX<VAL_T> transform_domain(const Eigen::VectorX<VAL_T> &vals, double lower, double upper) {
    double delta = upper - lower;
    return vals.array() * delta + lower;
}

template <typename FUN_T, typename VAL_T>
BenchResult<VAL_T>
test_func(const std::string name, const std::string library_prefix, const std::unordered_map<std::string, FUN_T> funs,
          std::unordered_map<std::string, Params> params, const Eigen::VectorX<VAL_T> &vals_in, size_t Nrepeat) {
    const std::string label = library_prefix + "_" + name;
    if (!funs.count(name))
        return BenchResult<VAL_T>(label);

    const Params &par = params[name];
    Eigen::VectorX<VAL_T> vals = transform_domain(vals_in, par.domain.first, par.domain.second);

    size_t res_size = vals.size();
    size_t n_evals = vals.size() * Nrepeat;
    if constexpr (std::is_same_v<FUN_T, fun_cdx1_x2>)
        res_size *= 2;
    BenchResult<VAL_T> res(label, res_size, n_evals, par);
    VAL_T *resptr = res.res.data();

    const FUN_T &f = funs.at(name);

    const struct timespec st = get_wtime();
    for (long k = 0; k < Nrepeat; k++) {
        if constexpr (std::is_same_v<FUN_T, fun_cdx1_x2>) {
            for (std::size_t i = 0; i < vals.size(); ++i) {
                std::tie(resptr[i * 2], resptr[i * 2 + 1]) = f(vals[i]);
            }
        } else if constexpr (std::is_same_v<FUN_T, std::shared_ptr<baobzi::Baobzi>>) {
            (*f)(vals.data(), resptr, vals.size());
        } else {
            f(vals.data(), resptr, vals.size());
        }
    }
    const struct timespec ft = get_wtime();

    res.eval_time = get_wtime_diff(&st, &ft);

    return res;
}

// https://eigen.tuxfamily.org/dox/group__CoeffwiseMathFunctions.html
namespace OPS {
enum OPS {
    COS,
    SIN,
    TAN,
    COSH,
    SINH,
    TANH,
    EXP,
    LOG,
    LOG10,
    POW35,
    POW13,
    ASIN,
    ACOS,
    ATAN,
    ASINH,
    ACOSH,
    ATANH,
    ERF,
    ERFC,
    LGAMMA,
    DIGAMMA,
    NDTRI,
    SQRT,
    RSQRT
};
}

template <>
BenchResult<double> test_func(const std::string name, const std::string library_prefix,
                              const std::unordered_map<std::string, OPS::OPS> funs,
                              std::unordered_map<std::string, Params> params, const Eigen::VectorXd &vals_in,
                              size_t Nrepeat) {
    const std::string label = library_prefix + "_" + name;
    if (!funs.count(name))
        return BenchResult<double>(label);

    const Params &par = params[name];
    Eigen::VectorXd x = transform_domain(vals_in, par.domain.first, par.domain.second);

    BenchResult<double> res(label, x.size(), x.size() * Nrepeat, par);

    Eigen::VectorXd &res_eigen = res.res;

    OPS::OPS OP = funs.at(name);
    const struct timespec st = get_wtime();

    for (long k = 0; k < Nrepeat; k++)
        switch (OP) {
        case OPS::COS:
            res_eigen = x.array().cos();
            break;
        case OPS::SIN:
            res_eigen = x.array().sin();
            break;
        case OPS::TAN:
            res_eigen = x.array().tan();
            break;
        case OPS::COSH:
            res_eigen = x.array().cosh();
            break;
        case OPS::SINH:
            res_eigen = x.array().sinh();
            break;
        case OPS::TANH:
            res_eigen = x.array().tanh();
            break;
        case OPS::EXP:
            res_eigen = x.array().exp();
            break;
        case OPS::LOG:
            res_eigen = x.array().log();
            break;
        case OPS::LOG10:
            res_eigen = x.array().log10();
            break;
        case OPS::POW35:
            res_eigen = x.array().pow(3.5);
            break;
        case OPS::POW13:
            res_eigen = x.array().pow(13);
            break;
        case OPS::ASIN:
            res_eigen = x.array().asin();
            break;
        case OPS::ACOS:
            res_eigen = x.array().acos();
            break;
        case OPS::ATAN:
            res_eigen = x.array().atan();
            break;
        case OPS::ASINH:
            res_eigen = x.array().asinh();
            break;
        case OPS::ACOSH:
            res_eigen = x.array().acosh();
            break;
        case OPS::ATANH:
            res_eigen = x.array().atanh();
            break;
        case OPS::ERF:
            res_eigen = x.array().erf();
            break;
        case OPS::ERFC:
            res_eigen = x.array().erfc();
            break;
        case OPS::LGAMMA:
            res_eigen = x.array().lgamma();
            break;
        case OPS::DIGAMMA:
            res_eigen = x.array().digamma();
            break;
        case OPS::NDTRI:
            res_eigen = x.array().ndtri();
            break;
        case OPS::SQRT:
            res_eigen = x.array().sqrt();
            break;
        case OPS::RSQRT:
            res_eigen = x.array().rsqrt();
            break;
        }

    const struct timespec ft = get_wtime();
    res.eval_time = get_wtime_diff(&st, &ft);

    return res;
}

std::set<std::string> parse_args(int argc, char *argv[]) {
    std::set<std::string> res;
    for (int i = 0; i < argc; ++i)
        res.insert(argv[i]);

    return res;
}

inline cdouble gsl_complex_wrapper(cdouble z, int (*f)(double, double, gsl_sf_result *, gsl_sf_result *)) {
    gsl_sf_result re, im;
    f(z.real(), z.imag(), &re, &im);
    return cdouble{re.val, im.val};
}

std::string exec(const char *cmd) {
    // https://stackoverflow.com/a/478960
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    result.pop_back();
    return result;
}

std::string get_alm_version() {
    std::string offset_str = "0x" + exec("objdump -t ../extern/amd-libm/lib/libalm.so --section=.rodata | grep -m1 "
                                         "ALM_VERSION_STRING | cut -d' ' -f 1");
    size_t offset = strtol(offset_str.c_str(), NULL, 0);
    FILE *obj = fopen("../extern/amd-libm/lib/libalm.so", "r");
    fseek(obj, offset, 0);
    char buf[16];
    fread(buf, sizeof(char), 16, obj);
    fclose(obj);
    return buf;
}

std::string get_sleef_version() {
    return std::to_string(SLEEF_VERSION_MAJOR) + "." + std::to_string(SLEEF_VERSION_MINOR) + "." +
           std::to_string(SLEEF_VERSION_PATCHLEVEL);
}

std::string get_af_version() {
    return std::to_string(VECTORCLASS_H / 10000) + "." + std::to_string((VECTORCLASS_H / 100) % 100) + "." +
           std::to_string(VECTORCLASS_H % 10);
}

std::string get_sctl_version() { return exec("cd ../extern/SCTL; git describe --tags"); }

std::string get_baobzi_version() { return exec("cd ../extern/baobzi; git describe --tags").substr(1); }

std::string get_eigen_version() {
    return std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) + "." +
           std::to_string(EIGEN_MINOR_VERSION);
}

std::string get_cpu_name() { return exec("grep -m1 'model name' /proc/cpuinfo | cut -d' ' --complement -f1-3"); }

double baobzi_fun_wrapper(const double *x, const void *data) {
    auto *myfun = (std::function<double(double)> *)data;
    return (*myfun)(*x);
}

std::shared_ptr<baobzi::Baobzi> create_baobzi_func(void *infun, const std::pair<double, double> &domain) {
    baobzi_input_t input = {.func = baobzi_fun_wrapper,
                            .data = infun,
                            .dim = 1,
                            .order = 8,
                            .tol = 1E-10,
                            .minimum_leaf_fraction = 0.6,
                            .split_multi_eval = 0};
    double hl = 0.5 * (domain.second - domain.first);
    double center = domain.first + hl;

    return std::shared_ptr<baobzi::Baobzi>(new baobzi::Baobzi(&input, &center, &hl));
}

int main(int argc, char *argv[]) {
    std::set<std::string> input_keys = parse_args(argc - 1, argv + 1);

    std::unordered_map<std::string, Params> params = {
        {"sin_pi", {.domain{0.0, 2.0}}},     {"cos_pi", {.domain{0.0, 2.0}}},     {"sin", {.domain{0.0, 2 * M_PI}}},
        {"cos", {.domain{0.0, 2 * M_PI}}},   {"tan", {.domain{0.0, 2 * M_PI}}},   {"asin", {.domain{-1.0, 1.0}}},
        {"acos", {.domain{-1.0, 1.0}}},      {"atan", {.domain{-100.0, 100.0}}},  {"erf", {.domain{-1.0, 1.0}}},
        {"erfc", {.domain{-1.0, 1.0}}},      {"exp", {.domain{-10.0, 10.0}}},     {"log", {.domain{0.0, 10.0}}},
        {"asinh", {.domain{-100.0, 100.0}}}, {"acosh", {.domain{1.0, 1000.0}}},   {"atanh", {.domain{-1.0, 1.0}}},
        {"bessel_Y0", {.domain{0.1, 30.0}}}, {"bessel_Y1", {.domain{0.1, 30.0}}}, {"bessel_Y2", {.domain{0.1, 30.0}}},
    };

    void *handle = dlopen("libalm.so", RTLD_NOW);

    using C_FUN1D = double (*)(double);
    using C_FUN2D = double (*)(double, double);
    C_FUN1D amd_sin = (C_FUN1D)dlsym(handle, "amd_sin");
    C_FUN1D amd_cos = (C_FUN1D)dlsym(handle, "amd_cos");
    C_FUN1D amd_tan = (C_FUN1D)dlsym(handle, "amd_tan");
    C_FUN1D amd_sinh = (C_FUN1D)dlsym(handle, "amd_sinh");
    C_FUN1D amd_cosh = (C_FUN1D)dlsym(handle, "amd_cosh");
    C_FUN1D amd_tanh = (C_FUN1D)dlsym(handle, "amd_tanh");
    C_FUN1D amd_asin = (C_FUN1D)dlsym(handle, "amd_asin");
    C_FUN1D amd_acos = (C_FUN1D)dlsym(handle, "amd_acos");
    C_FUN1D amd_atan = (C_FUN1D)dlsym(handle, "amd_atan");
    C_FUN1D amd_asinh = (C_FUN1D)dlsym(handle, "amd_asinh");
    C_FUN1D amd_acosh = (C_FUN1D)dlsym(handle, "amd_acosh");
    C_FUN1D amd_atanh = (C_FUN1D)dlsym(handle, "amd_atanh");
    C_FUN1D amd_log = (C_FUN1D)dlsym(handle, "amd_log");
    C_FUN1D amd_log2 = (C_FUN1D)dlsym(handle, "amd_log2");
    C_FUN1D amd_log10 = (C_FUN1D)dlsym(handle, "amd_log10");
    C_FUN1D amd_exp = (C_FUN1D)dlsym(handle, "amd_exp");
    C_FUN1D amd_exp2 = (C_FUN1D)dlsym(handle, "amd_exp2");
    C_FUN1D amd_exp10 = (C_FUN1D)dlsym(handle, "amd_exp10");
    C_FUN1D amd_sqrt = (C_FUN1D)dlsym(handle, "amd_sqrt");
    C_FUN2D amd_pow = (C_FUN2D)dlsym(handle, "amd_pow");

    using C_DX4_FUN1D = Vec4d (*)(Vec4d);
    using C_DX4_FUN2D = Vec4d (*)(Vec4d, Vec4d);
    C_DX4_FUN1D amd_vrd4_sin = (C_DX4_FUN1D)dlsym(handle, "amd_vrd4_sin");
    C_DX4_FUN1D amd_vrd4_cos = (C_DX4_FUN1D)dlsym(handle, "amd_vrd4_cos");
    C_DX4_FUN1D amd_vrd4_tan = (C_DX4_FUN1D)dlsym(handle, "amd_vrd4_tan");
    C_DX4_FUN1D amd_vrd4_log = (C_DX4_FUN1D)dlsym(handle, "amd_vrd4_log");
    C_DX4_FUN1D amd_vrd4_log2 = (C_DX4_FUN1D)dlsym(handle, "amd_vrd4_log2");
    C_DX4_FUN1D amd_vrd4_exp = (C_DX4_FUN1D)dlsym(handle, "amd_vrd4_exp");
    C_DX4_FUN1D amd_vrd4_exp2 = (C_DX4_FUN1D)dlsym(handle, "amd_vrd4_exp2");
    C_DX4_FUN2D amd_vrd4_pow = (C_DX4_FUN2D)dlsym(handle, "amd_vrd4_pow");

    std::unordered_map<std::string, scalar_function<double>> fort_funs = {
        {"bessel_Y0", scalar_func_apply<double>([](double x) -> double {
             int n = 0;
             double y;
             fort_bessel_yn_(&n, &x, &y);
             return y;
         })},
        {"bessel_J0", scalar_func_apply<double>([](double x) -> double {
             int n = 0;
             double y;
             fort_bessel_jn_(&n, &x, &y);
             return y;
         })},
    };

    std::unordered_map<std::string, fun_cdx1_x2> hank10x_funs = {
        {"hank103", [](cdouble z) -> std::pair<cdouble, cdouble> {
             cdouble h0, h1;
             int ifexpon = 1;
             hank103_((double _Complex *)&z, (double _Complex *)&h0, (double _Complex *)&h1, &ifexpon);
             return {h0, h1};
         }}};

    std::unordered_map<std::string, scalar_function<double>> gsl_funs = {
        {"sin_pi", scalar_func_apply<double>([](double x) -> double { return gsl_sf_sin_pi(x); })},
        {"cos_pi", scalar_func_apply<double>([](double x) -> double { return gsl_sf_cos_pi(x); })},
        {"sin", scalar_func_apply<double>([](double x) -> double { return gsl_sf_sin(x); })},
        {"cos", scalar_func_apply<double>([](double x) -> double { return gsl_sf_cos(x); })},
        {"sinc", scalar_func_apply<double>([](double x) -> double { return gsl_sf_sinc(x / M_PI); })},
        {"sinc_pi", scalar_func_apply<double>([](double x) -> double { return gsl_sf_sinc(x); })},
        {"erf", scalar_func_apply<double>([](double x) -> double { return gsl_sf_erf(x); })},
        {"erfc", scalar_func_apply<double>([](double x) -> double { return gsl_sf_erfc(x); })},
        {"tgamma", scalar_func_apply<double>([](double x) -> double { return gsl_sf_gamma(x); })},
        {"lgamma", scalar_func_apply<double>([](double x) -> double { return gsl_sf_lngamma(x); })},
        {"log", scalar_func_apply<double>([](double x) -> double { return gsl_sf_log(x); })},
        {"exp", scalar_func_apply<double>([](double x) -> double { return gsl_sf_exp(x); })},
        {"pow13", scalar_func_apply<double>([](double x) -> double { return gsl_sf_pow_int(x, 13); })},
        {"bessel_Y0", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_Y0(x); })},
        {"bessel_Y1", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_Y1(x); })},
        {"bessel_Y2", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_Yn(2, x); })},
        {"bessel_I0", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_I0(x); })},
        {"bessel_I1", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_I1(x); })},
        {"bessel_I2", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_In(2, x); })},
        {"bessel_J0", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_J0(x); })},
        {"bessel_J1", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_J1(x); })},
        {"bessel_J2", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_Jn(2, x); })},
        {"bessel_K0", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_K0(x); })},
        {"bessel_K1", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_K1(x); })},
        {"bessel_K2", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_Kn(2, x); })},
        {"bessel_j0", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_j0(x); })},
        {"bessel_j1", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_j1(x); })},
        {"bessel_j2", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_j2(x); })},
        {"bessel_y0", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_y0(x); })},
        {"bessel_y1", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_y1(x); })},
        {"bessel_y2", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_y2(x); })},
        {"hermite_0", scalar_func_apply<double>([](double x) -> double { return gsl_sf_hermite(0, x); })},
        {"hermite_1", scalar_func_apply<double>([](double x) -> double { return gsl_sf_hermite(1, x); })},
        {"hermite_2", scalar_func_apply<double>([](double x) -> double { return gsl_sf_hermite(2, x); })},
        {"hermite_3", scalar_func_apply<double>([](double x) -> double { return gsl_sf_hermite(3, x); })},
        {"riemann_zeta", scalar_func_apply<double>([](double x) -> double { return gsl_sf_zeta(x); })},
    };

    // FIXME: check accuracy of this and this+test_func
    std::unordered_map<std::string, scalar_function<cdouble>> gsl_complex_funs = {
        {"sin",
         scalar_func_apply<cdouble>([](cdouble z) -> cdouble { return gsl_complex_wrapper(z, gsl_sf_complex_sin_e); })},
        {"cos",
         scalar_func_apply<cdouble>([](cdouble z) -> cdouble { return gsl_complex_wrapper(z, gsl_sf_complex_cos_e); })},
        {"log",
         scalar_func_apply<cdouble>([](cdouble z) -> cdouble { return gsl_complex_wrapper(z, gsl_sf_complex_log_e); })},
        {"dilog", scalar_func_apply<cdouble>(
                      [](cdouble z) -> cdouble { return gsl_complex_wrapper(z, gsl_sf_complex_dilog_e); })},
        {"lgamma", scalar_func_apply<cdouble>(
                       [](cdouble z) -> cdouble { return gsl_complex_wrapper(z, gsl_sf_lngamma_complex_e); })},
    };

    std::unordered_map<std::string, scalar_function<double>> boost_funs = {
        {"sin_pi", scalar_func_apply<double>([](double x) -> double { return boost::math::sin_pi(x); })},
        {"cos_pi", scalar_func_apply<double>([](double x) -> double { return boost::math::cos_pi(x); })},
        {"tgamma", scalar_func_apply<double>([](double x) -> double { return boost::math::tgamma<double>(x); })},
        {"lgamma", scalar_func_apply<double>([](double x) -> double { return boost::math::lgamma<double>(x); })},
        {"digamma", scalar_func_apply<double>([](double x) -> double { return boost::math::digamma<double>(x); })},
        {"pow13", scalar_func_apply<double>([](double x) -> double { return boost::math::pow<13>(x); })},
        {"erf", scalar_func_apply<double>([](double x) -> double { return boost::math::erf(x); })},
        {"erfc", scalar_func_apply<double>([](double x) -> double { return boost::math::erfc(x); })},
        {"sinc_pi", scalar_func_apply<double>([](double x) -> double { return boost::math::sinc_pi(x); })},
        {"bessel_Y0", scalar_func_apply<double>([](double x) -> double { return boost::math::cyl_neumann(0, x); })},
        {"bessel_Y1", scalar_func_apply<double>([](double x) -> double { return boost::math::cyl_neumann(1, x); })},
        {"bessel_Y2", scalar_func_apply<double>([](double x) -> double { return boost::math::cyl_neumann(2, x); })},
        {"bessel_I0", scalar_func_apply<double>([](double x) -> double { return boost::math::cyl_bessel_i(0, x); })},
        {"bessel_I1", scalar_func_apply<double>([](double x) -> double { return boost::math::cyl_bessel_i(1, x); })},
        {"bessel_I2", scalar_func_apply<double>([](double x) -> double { return boost::math::cyl_bessel_i(2, x); })},
        {"bessel_J0", scalar_func_apply<double>([](double x) -> double { return boost::math::cyl_bessel_j(0, x); })},
        {"bessel_J1", scalar_func_apply<double>([](double x) -> double { return boost::math::cyl_bessel_j(1, x); })},
        {"bessel_J2", scalar_func_apply<double>([](double x) -> double { return boost::math::cyl_bessel_j(2, x); })},
        {"bessel_K0", scalar_func_apply<double>([](double x) -> double { return boost::math::cyl_bessel_k(0, x); })},
        {"bessel_K1", scalar_func_apply<double>([](double x) -> double { return boost::math::cyl_bessel_k(1, x); })},
        {"bessel_K2", scalar_func_apply<double>([](double x) -> double { return boost::math::cyl_bessel_k(2, x); })},
        {"bessel_j0", scalar_func_apply<double>([](double x) -> double { return boost::math::sph_bessel(0, x); })},
        {"bessel_j1", scalar_func_apply<double>([](double x) -> double { return boost::math::sph_bessel(1, x); })},
        {"bessel_j2", scalar_func_apply<double>([](double x) -> double { return boost::math::sph_bessel(2, x); })},
        {"bessel_y0", scalar_func_apply<double>([](double x) -> double { return boost::math::sph_neumann(0, x); })},
        {"bessel_y1", scalar_func_apply<double>([](double x) -> double { return boost::math::sph_neumann(1, x); })},
        {"bessel_y2", scalar_func_apply<double>([](double x) -> double { return boost::math::sph_neumann(2, x); })},
        {"hermite_0", scalar_func_apply<double>([](double x) -> double { return boost::math::hermite(0, x); })},
        {"hermite_1", scalar_func_apply<double>([](double x) -> double { return boost::math::hermite(1, x); })},
        {"hermite_2", scalar_func_apply<double>([](double x) -> double { return boost::math::hermite(2, x); })},
        {"hermite_3", scalar_func_apply<double>([](double x) -> double { return boost::math::hermite(3, x); })},
        {"riemann_zeta", scalar_func_apply<double>([](double x) -> double { return boost::math::zeta(x); })},
    };

    std::unordered_map<std::string, scalar_function<double>> std_funs = {
        {"tgamma", scalar_func_apply<double>([](double x) -> double { return std::tgamma(x); })},
        {"lgamma", scalar_func_apply<double>([](double x) -> double { return std::lgamma(x); })},
        {"sin", scalar_func_apply<double>([](double x) -> double { return std::sin(x); })},
        {"cos", scalar_func_apply<double>([](double x) -> double { return std::cos(x); })},
        {"tan", scalar_func_apply<double>([](double x) -> double { return std::tan(x); })},
        {"asin", scalar_func_apply<double>([](double x) -> double { return std::asin(x); })},
        {"acos", scalar_func_apply<double>([](double x) -> double { return std::acos(x); })},
        {"atan", scalar_func_apply<double>([](double x) -> double { return std::atan(x); })},
        {"asin", scalar_func_apply<double>([](double x) -> double { return std::asin(x); })},
        {"acos", scalar_func_apply<double>([](double x) -> double { return std::acos(x); })},
        {"atan", scalar_func_apply<double>([](double x) -> double { return std::atan(x); })},
        {"sinh", scalar_func_apply<double>([](double x) -> double { return std::sinh(x); })},
        {"cosh", scalar_func_apply<double>([](double x) -> double { return std::cosh(x); })},
        {"tanh", scalar_func_apply<double>([](double x) -> double { return std::tanh(x); })},
        {"asinh", scalar_func_apply<double>([](double x) -> double { return std::asinh(x); })},
        {"acosh", scalar_func_apply<double>([](double x) -> double { return std::acosh(x); })},
        {"atanh", scalar_func_apply<double>([](double x) -> double { return std::atanh(x); })},
        {"sin_pi", scalar_func_apply<double>([](double x) -> double { return std::sin(M_PI * x); })},
        {"cos_pi", scalar_func_apply<double>([](double x) -> double { return std::cos(M_PI * x); })},
        {"erf", scalar_func_apply<double>([](double x) -> double { return std::erf(x); })},
        {"erfc", scalar_func_apply<double>([](double x) -> double { return std::erfc(x); })},
        {"log", scalar_func_apply<double>([](double x) -> double { return std::log(x); })},
        {"log2", scalar_func_apply<double>([](double x) -> double { return std::log2(x); })},
        {"log10", scalar_func_apply<double>([](double x) -> double { return std::log10(x); })},
        {"exp", scalar_func_apply<double>([](double x) -> double { return std::exp(x); })},
        {"exp2", scalar_func_apply<double>([](double x) -> double { return std::exp2(x); })},
        {"exp10", scalar_func_apply<double>([](double x) -> double { return exp10(x); })},
        {"sqrt", scalar_func_apply<double>([](double x) -> double { return std::sqrt(x); })},
        {"rsqrt", scalar_func_apply<double>([](double x) -> double { return 1.0 / std::sqrt(x); })},
        {"pow3.5", scalar_func_apply<double>([](double x) -> double { return std::pow(x, 3.5); })},
        {"pow13", scalar_func_apply<double>([](double x) -> double { return std::pow(x, 13); })},
    };

    std::unordered_map<std::string, scalar_function<double>> amdlibm_funs = {
        {"sin", scalar_func_apply<double>([&amd_sin](double x) -> double { return amd_sin(x); })},
        {"cos", scalar_func_apply<double>([&amd_cos](double x) -> double { return amd_cos(x); })},
        {"tan", scalar_func_apply<double>([&amd_tan](double x) -> double { return amd_tan(x); })},
        {"sinh", scalar_func_apply<double>([&amd_sinh](double x) -> double { return amd_sinh(x); })},
        {"cosh", scalar_func_apply<double>([&amd_cosh](double x) -> double { return amd_cosh(x); })},
        {"tanh", scalar_func_apply<double>([&amd_tanh](double x) -> double { return amd_tanh(x); })},
        {"asin", scalar_func_apply<double>([&amd_asin](double x) -> double { return amd_asin(x); })},
        {"acos", scalar_func_apply<double>([&amd_acos](double x) -> double { return amd_acos(x); })},
        {"atan", scalar_func_apply<double>([&amd_atan](double x) -> double { return amd_atan(x); })},
        {"asinh", scalar_func_apply<double>([&amd_asinh](double x) -> double { return amd_asinh(x); })},
        {"acosh", scalar_func_apply<double>([&amd_acosh](double x) -> double { return amd_acosh(x); })},
        {"atanh", scalar_func_apply<double>([&amd_atanh](double x) -> double { return amd_atanh(x); })},
        {"log", scalar_func_apply<double>([&amd_log](double x) -> double { return amd_log(x); })},
        {"log2", scalar_func_apply<double>([&amd_log2](double x) -> double { return amd_log2(x); })},
        {"log10", scalar_func_apply<double>([&amd_log10](double x) -> double { return amd_log10(x); })},
        {"exp", scalar_func_apply<double>([&amd_exp](double x) -> double { return amd_exp(x); })},
        {"exp2", scalar_func_apply<double>([&amd_exp2](double x) -> double { return amd_exp2(x); })},
        {"exp10", scalar_func_apply<double>([&amd_exp10](double x) -> double { return amd_exp10(x); })},
        {"sqrt", scalar_func_apply<double>([&amd_sqrt](double x) -> double { return amd_sqrt(x); })},
        {"pow3.5", scalar_func_apply<double>([&amd_pow](double x) -> double { return amd_pow(x, 3.5); })},
        {"pow13", scalar_func_apply<double>([&amd_pow](double x) -> double { return amd_pow(x, 13); })},
    };

    Vec4d x;
    std::unordered_map<std::string, vec_function> amdlibm_funs_dx4 = {
        {"sin", vec_func_apply<Vec4d, double>([&amd_vrd4_sin](Vec4d x) -> Vec4d { return amd_vrd4_sin(x); })},
        {"cos", vec_func_apply<Vec4d, double>([&amd_vrd4_cos](Vec4d x) -> Vec4d { return amd_vrd4_cos(x); })},
        {"tan", vec_func_apply<Vec4d, double>([&amd_vrd4_tan](Vec4d x) -> Vec4d { return amd_vrd4_tan(x); })},
        {"log", vec_func_apply<Vec4d, double>([&amd_vrd4_log](Vec4d x) -> Vec4d { return amd_vrd4_log(x); })},
        {"log2", vec_func_apply<Vec4d, double>([&amd_vrd4_log2](Vec4d x) -> Vec4d { return amd_vrd4_log2(x); })},
        {"exp", vec_func_apply<Vec4d, double>([&amd_vrd4_exp](Vec4d x) -> Vec4d { return amd_vrd4_exp(x); })},
        {"exp2", vec_func_apply<Vec4d, double>([&amd_vrd4_exp2](Vec4d x) -> Vec4d { return amd_vrd4_exp2(x); })},
        {"pow3.5",
         vec_func_apply<Vec4d, double>([&amd_vrd4_pow](Vec4d x) -> Vec4d { return amd_vrd4_pow(x, Vec4d{3.5}); })},
        {"pow13",
         vec_func_apply<Vec4d, double>([&amd_vrd4_pow](Vec4d x) -> Vec4d { return amd_vrd4_pow(x, Vec4d{13}); })},
    };

    std::unordered_map<std::string, scalar_function<double>> sleef_funs = {
        {"sin_pi", scalar_func_apply<double>([](double x) -> double { return Sleef_sinpid1_u05purecfma(x); })},
        {"cos_pi", scalar_func_apply<double>([](double x) -> double { return Sleef_cospid1_u05purecfma(x); })},
        {"sin", scalar_func_apply<double>([](double x) -> double { return Sleef_sind1_u10purecfma(x); })},
        {"cos", scalar_func_apply<double>([](double x) -> double { return Sleef_cosd1_u10purecfma(x); })},
        {"tan", scalar_func_apply<double>([](double x) -> double { return Sleef_tand1_u10purecfma(x); })},
        {"sinh", scalar_func_apply<double>([](double x) -> double { return Sleef_sinhd1_u10purecfma(x); })},
        {"cosh", scalar_func_apply<double>([](double x) -> double { return Sleef_coshd1_u10purecfma(x); })},
        {"tanh", scalar_func_apply<double>([](double x) -> double { return Sleef_tanhd1_u10purecfma(x); })},
        {"asin", scalar_func_apply<double>([](double x) -> double { return Sleef_asind1_u10purecfma(x); })},
        {"acos", scalar_func_apply<double>([](double x) -> double { return Sleef_acosd1_u10purecfma(x); })},
        {"atan", scalar_func_apply<double>([](double x) -> double { return Sleef_atand1_u10purecfma(x); })},
        {"asinh", scalar_func_apply<double>([](double x) -> double { return Sleef_asinhd1_u10purecfma(x); })},
        {"acosh", scalar_func_apply<double>([](double x) -> double { return Sleef_acoshd1_u10purecfma(x); })},
        {"atanh", scalar_func_apply<double>([](double x) -> double { return Sleef_atanhd1_u10purecfma(x); })},
        {"log", scalar_func_apply<double>([](double x) -> double { return Sleef_logd1_u10purecfma(x); })},
        {"log2", scalar_func_apply<double>([](double x) -> double { return Sleef_log2d1_u10purecfma(x); })},
        {"log10", scalar_func_apply<double>([](double x) -> double { return Sleef_log10d1_u10purecfma(x); })},
        {"exp", scalar_func_apply<double>([](double x) -> double { return Sleef_expd1_u10purecfma(x); })},
        {"exp2", scalar_func_apply<double>([](double x) -> double { return Sleef_exp2d1_u10purecfma(x); })},
        {"exp10", scalar_func_apply<double>([](double x) -> double { return Sleef_exp10d1_u10purecfma(x); })},
        {"erf", scalar_func_apply<double>([](double x) -> double { return Sleef_erfd1_u10purecfma(x); })},
        {"erfc", scalar_func_apply<double>([](double x) -> double { return Sleef_erfcd1_u15purecfma(x); })},
        {"lgamma", scalar_func_apply<double>([](double x) -> double { return Sleef_lgammad1_u10purecfma(x); })},
        {"tgamma", scalar_func_apply<double>([](double x) -> double { return Sleef_tgammad1_u10purecfma(x); })},
        {"sqrt", scalar_func_apply<double>([](double x) -> double { return Sleef_sqrtd1_u05purecfma(x); })},
        {"pow3.5", scalar_func_apply<double>([](double x) -> double { return Sleef_powd1_u10purecfma(x, 3.5); })},
        {"pow13", scalar_func_apply<double>([](double x) -> double { return Sleef_powd1_u10purecfma(x, 13); })},
    };

    std::unordered_map<std::string, vec_function> sleef_funs_dx4 = {
        {"sin_pi", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_sinpid4_u05avx2(x); })},
        {"cos_pi", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_cospid4_u05avx2(x); })},
        {"sin", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_sind4_u10avx2(x); })},
        {"cos", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_cosd4_u10avx2(x); })},
        {"tan", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_tand4_u10avx2(x); })},
        {"sinh", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_sinhd4_u10avx2(x); })},
        {"cosh", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_coshd4_u10avx2(x); })},
        {"tanh", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_tanhd4_u10avx2(x); })},
        {"asin", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_asind4_u10avx2(x); })},
        {"acos", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_acosd4_u10avx2(x); })},
        {"atan", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_atand4_u10avx2(x); })},
        {"asinh", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_asinhd4_u10avx2(x); })},
        {"acosh", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_acoshd4_u10avx2(x); })},
        {"atanh", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_atanhd4_u10avx2(x); })},
        {"log", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_logd4_u10avx2(x); })},
        {"log2", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_log2d4_u10avx2(x); })},
        {"log10", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_log10d4_u10avx2(x); })},
        {"exp", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_expd4_u10avx2(x); })},
        {"exp2", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_exp2d4_u10avx2(x); })},
        {"exp10", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_exp10d4_u10avx2(x); })},
        {"erf", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_erfd4_u10avx2(x); })},
        {"erfc", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_erfcd4_u15avx2(x); })},
        {"lgamma", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_lgammad4_u10avx2(x); })},
        {"tlgamma", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_tgammad4_u10avx2(x); })},
        {"sqrt", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_sqrtd4_u05avx2(x); })},
        {"pow3.5", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_powd4_u10avx2(x, Vec4d{3.5}); })},
        {"pow13", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return Sleef_powd4_u10avx2(x, Vec4d{13}); })},
    };

#ifdef __AVX512F__
    std::unordered_map<std::string, vec_function> sleef_funs_dx8 = {
        {"sin_pi", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_sinpid8_u05avx512f(x); })},
        {"cos_pi", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_cospid8_u05avx512f(x); })},
        {"sin", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_sind8_u10avx512f(x); })},
        {"cos", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_cosd8_u10avx512f(x); })},
        {"tan", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_tand8_u10avx512f(x); })},
        {"sinh", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_sinhd8_u10avx512f(x); })},
        {"cosh", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_coshd8_u10avx512f(x); })},
        {"tanh", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_tanhd8_u10avx512f(x); })},
        {"asin", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_asind8_u10avx512f(x); })},
        {"acos", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_acosd8_u10avx512f(x); })},
        {"atan", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_atand8_u10avx512f(x); })},
        {"asinh", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_asinhd8_u10avx512f(x); })},
        {"acosh", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_acoshd8_u10avx512f(x); })},
        {"atanh", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_atanhd8_u10avx512f(x); })},
        {"log", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_logd8_u10avx512f(x); })},
        {"log2", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_log2d8_u10avx512f(x); })},
        {"log10", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_log10d8_u10avx512f(x); })},
        {"exp", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_expd8_u10avx512f(x); })},
        {"exp2", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_exp2d8_u10avx512f(x); })},
        {"exp10", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_exp10d8_u10avx512f(x); })},
        {"erf", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_erfd8_u10avx512f(x); })},
        {"erfc", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_erfcd8_u15avx512f(x); })},
        {"lgamma", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_lgammad8_u10avx512f(x); })},
        {"tlgamma", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_tgammad8_u10avx512f(x); })},
        {"sqrt", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_sqrtd8_u05avx512f(x); })},
        {"pow3.5",
         vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_powd8_u10avx512f(x, Vec8d{3.5}); })},
        {"pow13", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return Sleef_powd8_u10avx512f(x, Vec8d{13}); })},
    };
#endif

    std::unordered_map<std::string, vec_function> af_funs_dx4 = {
        {"sqrt", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return sqrt(x); })},
        {"sin", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return sin(x); })},
        {"cos", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return cos(x); })},
        {"tan", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return tan(x); })},
        {"sinh", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return sinh(x); })},
        {"cosh", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return cosh(x); })},
        {"tanh", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return tanh(x); })},
        {"asinh", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return asinh(x); })},
        {"acosh", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return acosh(x); })},
        {"atanh", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return atanh(x); })},
        {"asin", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return asin(x); })},
        {"acos", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return acos(x); })},
        {"atan", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return atan(x); })},
        {"exp", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return exp(x); })},
        {"exp2", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return exp2(x); })},
        {"exp10", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return exp10(x); })},
        {"log", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return log(x); })},
        {"log2", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return log2(x); })},
        {"log10", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return log10(x); })},
        {"pow3.5", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return pow(x, 3.5); })},
        {"pow13", vec_func_apply<Vec4d, double>([](Vec4d x) -> Vec4d { return pow_const(x, 13); })},
    };

#ifdef __AVX512F__
    std::unordered_map<std::string, vec_function> af_funs_dx8 = {
        {"sqrt", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return sqrt(x); })},
        {"sin", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return sin(x); })},
        {"cos", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return cos(x); })},
        {"tan", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return tan(x); })},
        {"sinh", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return sinh(x); })},
        {"cosh", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return cosh(x); })},
        {"tanh", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return tanh(x); })},
        {"asinh", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return asinh(x); })},
        {"acosh", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return acosh(x); })},
        {"atanh", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return atanh(x); })},
        {"asin", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return asin(x); })},
        {"acos", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return acos(x); })},
        {"atan", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return atan(x); })},
        {"exp", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return exp(x); })},
        {"exp2", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return exp2(x); })},
        {"exp10", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return exp10(x); })},
        {"log", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return log(x); })},
        {"log2", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return log2(x); })},
        {"log10", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return log10(x); })},
        {"pow3.5", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return pow(x, 3.5); })},
        {"pow13", vec_func_apply<Vec8d, double>([](Vec8d x) -> Vec8d { return pow_const(x, 13); })},
    };
#endif

    std::unordered_map<std::string, vec_function> sctl_funs_dx4 = {
        {"copy", sctl_apply<double, 4>([](const sctl_dx4 &x) { return x; })},
        {"exp", sctl_apply<double, 4>([](const sctl_dx4 &x) { return sctl::approx_exp<16>(x); })},
        {"sin", sctl_apply<double, 4>([](const sctl_dx4 &x) {
             sctl_dx4 sinx, cosx;
             sctl::approx_sincos<16>(sinx, cosx, x);
             return sinx;
         })},
        {"cos", sctl_apply<double, 4>([](const sctl_dx4 &x) {
             sctl_dx4 sinx, cosx;
             sctl::approx_sincos<16>(sinx, cosx, x);
             return cosx;
         })},
        {"rsqrt", sctl_apply<double, 4>([](const sctl_dx4 &x) { return sctl::approx_rsqrt<16>(x); })},
    };

    std::unordered_map<std::string, vec_function> sctl_funs_dx8 = {
        {"copy", sctl_apply<double, 8>([](const sctl_dx8 &x) { return x; })},
        {"exp", sctl_apply<double, 8>([](const sctl_dx8 &x) { return sctl::approx_exp<16>(x); })},
        {"sin", sctl_apply<double, 8>([](const sctl_dx8 &x) {
             sctl_dx8 sinx, cosx;
             sctl::approx_sincos<16>(sinx, cosx, x);
             return sinx;
         })},
        {"cos", sctl_apply<double, 8>([](const sctl_dx8 &x) {
             sctl_dx8 sinx, cosx;
             sctl::approx_sincos<16>(sinx, cosx, x);
             return cosx;
         })},
        {"rsqrt", sctl_apply<double, 8>([](const sctl_dx8 &x) { return sctl::approx_rsqrt<16>(x); })},
    };

    std::unordered_map<std::string, OPS::OPS> eigen_funs = {
        {"sin", OPS::SIN},         {"cos", OPS::COS},      {"tan", OPS::TAN},     {"sinh", OPS::SINH},
        {"cosh", OPS::COSH},       {"tanh", OPS::TANH},    {"exp", OPS::EXP},     {"log", OPS::LOG},
        {"log10", OPS::LOG10},     {"pow3.5", OPS::POW35}, {"pow13", OPS::POW13}, {"asin", OPS::ASIN},
        {"acos", OPS::ACOS},       {"atan", OPS::ATAN},    {"asinh", OPS::ASINH}, {"atanh", OPS::ATANH},
        {"acosh", OPS::ACOSH},     {"erf", OPS::ERF},      {"erfc", OPS::ERFC},   {"lgamma", OPS::LGAMMA},
        {"digamma", OPS::DIGAMMA}, {"ndtri", OPS::NDTRI},  {"sqrt", OPS::SQRT},   {"rsqrt", OPS::RSQRT},
    };

    std::set<std::string> fun_union;
    for (auto kv : amdlibm_funs)
        fun_union.insert(kv.first);
    for (auto kv : boost_funs)
        fun_union.insert(kv.first);
    for (auto kv : eigen_funs)
        fun_union.insert(kv.first);
    for (auto kv : fort_funs)
        fun_union.insert(kv.first);
    for (auto kv : gsl_funs)
        fun_union.insert(kv.first);
    for (auto kv : gsl_complex_funs)
        fun_union.insert(kv.first);
    for (auto kv : hank10x_funs)
        fun_union.insert(kv.first);
    for (auto kv : sleef_funs)
        fun_union.insert(kv.first);
    for (auto kv : std_funs)
        fun_union.insert(kv.first);

    for (auto kv : af_funs_dx4)
        fun_union.insert(kv.first);
    for (auto kv : amdlibm_funs_dx4)
        fun_union.insert(kv.first);
    for (auto kv : sctl_funs_dx4)
        fun_union.insert(kv.first);
    for (auto kv : sleef_funs_dx4)
        fun_union.insert(kv.first);
#ifdef __AVX512F__
    for (auto kv : af_funs_dx8)
        fun_union.insert(kv.first);
    for (auto kv : sctl_funs_dx8)
        fun_union.insert(kv.first);
    for (auto kv : sleef_funs_dx8)
        fun_union.insert(kv.first);
#endif

    std::set<std::string> keys_to_eval;
    if (input_keys.size() > 0)
        std::set_intersection(fun_union.begin(), fun_union.end(), input_keys.begin(), input_keys.end(),
                              std::inserter(keys_to_eval, keys_to_eval.end()));
    else
        keys_to_eval = fun_union;

    std::unordered_map<std::string, std::shared_ptr<baobzi::Baobzi>> baobzi_funs;
    std::unordered_map<std::string, std::function<double(double)>> potential_baobzi_funs{
        {"bessel_Y0", [](double x) -> double { return gsl_sf_bessel_Y0(x); }},
        {"bessel_Y1", [](double x) -> double { return gsl_sf_bessel_Y1(x); }},
        {"bessel_Y2", [](double x) -> double { return gsl_sf_bessel_Yn(2, x); }},
        {"bessel_I0", [](double x) -> double { return gsl_sf_bessel_I0(x); }},
        {"bessel_I1", [](double x) -> double { return gsl_sf_bessel_I1(x); }},
        {"bessel_I2", [](double x) -> double { return gsl_sf_bessel_In(2, x); }},
        {"bessel_J0", [](double x) -> double { return gsl_sf_bessel_J0(x); }},
        {"bessel_J1", [](double x) -> double { return gsl_sf_bessel_J1(x); }},
        {"bessel_J2", [](double x) -> double { return gsl_sf_bessel_Jn(2, x); }},
        {"hermite_0", [](double x) -> double { return gsl_sf_hermite(0, x); }},
        {"hermite_1", [](double x) -> double { return gsl_sf_hermite(1, x); }},
        {"hermite_2", [](double x) -> double { return gsl_sf_hermite(2, x); }},
        {"hermite_3", [](double x) -> double { return gsl_sf_hermite(3, x); }},
    };

    for (auto &key : keys_to_eval) {
        if (potential_baobzi_funs.count(key)) {
            std::cerr << "Creating baobzi function '" + key + "'.\n";
            baobzi_funs[key] = create_baobzi_func((void *)(&potential_baobzi_funs.at(key)), params[key].domain);
        }
    }

    const std::vector<std::pair<int, int>> run_sets = {{1024, 1e4}, {1024 * 1e4, 1}};
    for (auto &run_set : run_sets) {
        const auto &[n_eval, n_repeat] = run_set;
        std::cerr << "Running benchmark with input vector of length " << n_eval << " and " << n_repeat << " repeats.\n";
        Eigen::VectorXd vals = 0.5 * (Eigen::ArrayXd::Random(n_eval) + 1.0);
        Eigen::VectorX<cdouble> cvals = 0.5 * (Eigen::ArrayX<cdouble>::Random(n_eval) + std::complex<double>{1.0, 1.0});

        for (auto key : keys_to_eval) {
            std::cout << test_func(key, "std", std_funs, params, vals, n_repeat);
            std::cout << test_func(key, "fort", fort_funs, params, vals, n_repeat);
            std::cout << test_func(key, "amdlibm", amdlibm_funs, params, vals, n_repeat);
            std::cout << test_func(key, "boost", boost_funs, params, vals, n_repeat);
            std::cout << test_func(key, "gsl", gsl_funs, params, vals, n_repeat);
            std::cout << test_func(key, "gsl_complex", gsl_complex_funs, params, cvals, n_repeat);
            std::cout << test_func(key, "sleef", sleef_funs, params, vals, n_repeat);
            std::cout << test_func(key, "hank10x", hank10x_funs, params, cvals, n_repeat);
            std::cout << test_func(key, "baobzi", baobzi_funs, params, vals, n_repeat);
            std::cout << test_func(key, "eigen", eigen_funs, params, vals, n_repeat);
            std::cout << test_func(key, "amdlibm_dx4", amdlibm_funs_dx4, params, vals, n_repeat);
            std::cout << test_func(key, "agnerfog_dx4", af_funs_dx4, params, vals, n_repeat);
            std::cout << test_func(key, "sctl_dx4", sctl_funs_dx4, params, vals, n_repeat);
            std::cout << test_func(key, "sleef_dx4", sleef_funs_dx4, params, vals, n_repeat);
#ifdef __AVX512F__
            std::cout << test_func(key, "agnerfog_dx8", af_funs_dx8, params, vals, n_repeat);
            std::cout << test_func(key, "sctl_dx8", sctl_funs_dx8, params, vals, n_repeat);
            std::cout << test_func(key, "sleef_dx8", sleef_funs_dx8, params, vals, n_repeat);
#endif
            std::cout << "\n";
        }
    }
    return 0;
}
