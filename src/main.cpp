#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#include <sf_libraries.hpp>
#include <sf_utils.hpp>

#include <sqlite3.h>

#include <dlfcn.h>
#include <time.h>

const sf::utils::host_info_t host_info;
const sf::utils::library_info_t libraries_info[] = {
    {"sctl", sf::utils::get_sctl_version()},   {"baobzi", sf::utils::get_baobzi_version()},
    {"boost", sf::utils::get_boost_version()}, {"amdlibm", sf::utils::get_alm_version()},
    {"sleef", sf::utils::get_sleef_version()}, {"gsl", sf::utils::get_gsl_version()},
    {"agnerfog", sf::utils::get_af_version()}, {"baobzi", sf::utils::get_baobzi_version()},
    {"eigen", sf::utils::get_eigen_version()}, {"misc", "NA"},
};
const sf::utils::toolchain_info_t toolchain_info;

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
    VAL_T delta = upper - lower;
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
    cos,
    sin,
    tan,
    cosh,
    sinh,
    tanh,
    exp,
    log,
    log10,
    pow35,
    pow13,
    asin,
    acos,
    atan,
    asinh,
    acosh,
    atanh,
    erf,
    erfc,
    lgamma,
    digamma,
    ndtri,
    sqrt,
    rsqrt
};
}

#define EIGEN_CASE(OP)                                                                                                 \
    case OPS::OP: {                                                                                                    \
        res_eigen = x.array().OP();                                                                                    \
        break;                                                                                                         \
    }

template <typename VAL_T>
BenchResult<VAL_T> test_func(const std::string name, const std::string library_prefix,
                             const std::unordered_map<std::string, OPS::OPS> funs,
                             std::unordered_map<std::string, Params> params, const Eigen::VectorX<VAL_T> &vals_in,
                             size_t Nrepeat) {
    const std::string label = library_prefix + "_" + name;
    if (!funs.count(name))
        return BenchResult<VAL_T>(label);

    const Params &par = params[name];
    Eigen::VectorX<VAL_T> x = transform_domain(vals_in, par.domain.first, par.domain.second);

    BenchResult<VAL_T> res(label, x.size(), x.size() * Nrepeat, par);

    Eigen::VectorX<VAL_T> &res_eigen = res.res;

    OPS::OPS OP = funs.at(name);
    const struct timespec st = get_wtime();

    for (long k = 0; k < Nrepeat; k++)
        switch (OP) {
            EIGEN_CASE(cos)
            EIGEN_CASE(sin)
            EIGEN_CASE(tan)
            EIGEN_CASE(cosh)
            EIGEN_CASE(sinh)
            EIGEN_CASE(tanh)
            EIGEN_CASE(exp)
            EIGEN_CASE(log)
            EIGEN_CASE(log10)
            EIGEN_CASE(asin)
            EIGEN_CASE(acos)
            EIGEN_CASE(atan)
            EIGEN_CASE(asinh)
            EIGEN_CASE(acosh)
            EIGEN_CASE(atanh)
            EIGEN_CASE(erf)
            EIGEN_CASE(erfc)
            EIGEN_CASE(lgamma)
            EIGEN_CASE(digamma)
            EIGEN_CASE(ndtri)
            EIGEN_CASE(sqrt)
            EIGEN_CASE(rsqrt)
        case OPS::pow35: {
            res_eigen = x.array().pow(3.5);
            break;
        }
        case OPS::pow13: {
            res_eigen = x.array().pow(13);
            break;
        }
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

class Database {
  public:
    Database() = default;

    Database(std::string db_file) {
        if (std::filesystem::exists(db_file)) {
            sqlite3_open(db_file.c_str(), &db_);
            update_host_info();
            update_library_info();
            update_toolchain_info();
        } else
            throw std::runtime_error("DB file '" + db_file + "' does not exist.\n");
    }

    ~Database() { sqlite3_close(db_); }

    static std::string quote(const std::string &str) { return "\"" + str + "\""; }
    static int errcheck(char *err) {
        if (err != NULL) {
            printf("%s\n", err);
            sqlite3_free(err);
            return 1;
        }
        return 0;
    }

    bool update_host_info() {
        char *err = NULL;
        std::string sql = "INSERT OR IGNORE INTO hosts (cpuname,l1dcache,l1icache,l2cache,l3cache) VALUES(" +
                          quote(host_info.cpu_name) + "," + quote(host_info.L1d) + "," + quote(host_info.L1i) + "," +
                          quote(host_info.L2) + "," + quote(host_info.L3) + ");";
        sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err);
        return errcheck(err);
    }

    bool update_library_info() {
        bool iserr = false;
        for (auto &library_info : libraries_info) {
            char *err = NULL;
            std::string sql = "INSERT OR IGNORE INTO libraries (name,version) VALUES(" + quote(library_info.name) +
                              "," + quote(library_info.version) + ");";
            sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err);
            if (errcheck(err))
                iserr = true;
        }
        return iserr;
    }

    bool update_toolchain_info() {
        char *err = NULL;
        std::string sql = "INSERT OR IGNORE INTO toolchains (compiler,compilervers,libcvers) VALUES(" +
                          quote(toolchain_info.compiler) + "," + quote(toolchain_info.compilervers) + "," +
                          quote(toolchain_info.libcvers) + ");";
        sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err);
        return errcheck(err);
    }

  private:
    sqlite3 *db_ = nullptr;
};

int main(int argc, char *argv[]) {
    Database db("../sf_benchmarks.sqlite");

    std::cout << host_info.cpu_name << std::endl;
    std::cout << "    " + toolchain_info.compiler + ": " + toolchain_info.compilervers << std::endl;
    std::cout << "    libc: " + toolchain_info.libcvers << std::endl;
    for (auto &library_info : libraries_info)
        std::cout << "    " + library_info.name + ": " + library_info.version << std::endl;

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

    using C_FUN1F = float (*)(float);
    using C_FUN2F = float (*)(float, float);
    C_FUN1F amd_sinf = (C_FUN1F)dlsym(handle, "amd_sinf");
    C_FUN1F amd_cosf = (C_FUN1F)dlsym(handle, "amd_cosf");
    C_FUN1F amd_tanf = (C_FUN1F)dlsym(handle, "amd_tanf");
    C_FUN1F amd_sinhf = (C_FUN1F)dlsym(handle, "amd_sinhf");
    C_FUN1F amd_coshf = (C_FUN1F)dlsym(handle, "amd_coshf");
    C_FUN1F amd_tanhf = (C_FUN1F)dlsym(handle, "amd_tanhf");
    C_FUN1F amd_asinf = (C_FUN1F)dlsym(handle, "amd_asinf");
    C_FUN1F amd_acosf = (C_FUN1F)dlsym(handle, "amd_acosf");
    C_FUN1F amd_atanf = (C_FUN1F)dlsym(handle, "amd_atanf");
    C_FUN1F amd_asinhf = (C_FUN1F)dlsym(handle, "amd_asinhf");
    C_FUN1F amd_acoshf = (C_FUN1F)dlsym(handle, "amd_acoshf");
    C_FUN1F amd_atanhf = (C_FUN1F)dlsym(handle, "amd_atanhf");
    C_FUN1F amd_logf = (C_FUN1F)dlsym(handle, "amd_logf");
    C_FUN1F amd_log2f = (C_FUN1F)dlsym(handle, "amd_log2f");
    C_FUN1F amd_log10f = (C_FUN1F)dlsym(handle, "amd_log10f");
    C_FUN1F amd_expf = (C_FUN1F)dlsym(handle, "amd_expf");
    C_FUN1F amd_exp2f = (C_FUN1F)dlsym(handle, "amd_exp2f");
    C_FUN1F amd_exp10f = (C_FUN1F)dlsym(handle, "amd_exp10f");
    C_FUN1F amd_sqrtf = (C_FUN1F)dlsym(handle, "amd_sqrtf");
    C_FUN2F amd_powf = (C_FUN2F)dlsym(handle, "amd_powf");

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

    using C_FX8_FUN1F = Vec8f (*)(Vec8f);
    using C_FX8_FUN2F = Vec8f (*)(Vec8f, Vec8f);
    C_FX8_FUN1F amd_vrs8_sinf = (C_FX8_FUN1F)dlsym(handle, "amd_vrs8_sinf");
    C_FX8_FUN1F amd_vrs8_cosf = (C_FX8_FUN1F)dlsym(handle, "amd_vrs8_cosf");
    C_FX8_FUN1F amd_vrs8_tanf = (C_FX8_FUN1F)dlsym(handle, "amd_vrs8_tanf");
    C_FX8_FUN1F amd_vrs8_logf = (C_FX8_FUN1F)dlsym(handle, "amd_vrs8_logf");
    C_FX8_FUN1F amd_vrs8_log2f = (C_FX8_FUN1F)dlsym(handle, "amd_vrs8_log2f");
    C_FX8_FUN1F amd_vrs8_expf = (C_FX8_FUN1F)dlsym(handle, "amd_vrs8_expf");
    C_FX8_FUN1F amd_vrs8_exp2f = (C_FX8_FUN1F)dlsym(handle, "amd_vrs8_exp2f");
    C_FX8_FUN2F amd_vrs8_powf = (C_FX8_FUN2F)dlsym(handle, "amd_vrs8_powf");

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

    std::unordered_map<std::string, multi_eval_func<double>> fort_funs = {
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

    std::unordered_map<std::string, multi_eval_func<double>> gsl_funs = {
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
    std::unordered_map<std::string, multi_eval_func<cdouble>> gsl_complex_funs = {
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

    std::unordered_map<std::string, multi_eval_func<float>> boost_funs_fx1 = {
        {"sin_pi", scalar_func_apply<float>([](float x) -> float { return boost::math::sin_pi(x); })},
        {"cos_pi", scalar_func_apply<float>([](float x) -> float { return boost::math::cos_pi(x); })},
        {"tgamma", scalar_func_apply<float>([](float x) -> float { return boost::math::tgamma<float>(x); })},
        {"lgamma", scalar_func_apply<float>([](float x) -> float { return boost::math::lgamma<float>(x); })},
        {"digamma", scalar_func_apply<float>([](float x) -> float { return boost::math::digamma<float>(x); })},
        {"pow13", scalar_func_apply<float>([](float x) -> float { return boost::math::pow<13>(x); })},
        {"erf", scalar_func_apply<float>([](float x) -> float { return boost::math::erf(x); })},
        {"erfc", scalar_func_apply<float>([](float x) -> float { return boost::math::erfc(x); })},
        {"sinc_pi", scalar_func_apply<float>([](float x) -> float { return boost::math::sinc_pi(x); })},
        {"bessel_Y0", scalar_func_apply<float>([](float x) -> float { return boost::math::cyl_neumann(0, x); })},
        {"bessel_Y1", scalar_func_apply<float>([](float x) -> float { return boost::math::cyl_neumann(1, x); })},
        {"bessel_Y2", scalar_func_apply<float>([](float x) -> float { return boost::math::cyl_neumann(2, x); })},
        {"bessel_I0", scalar_func_apply<float>([](float x) -> float { return boost::math::cyl_bessel_i(0, x); })},
        {"bessel_I1", scalar_func_apply<float>([](float x) -> float { return boost::math::cyl_bessel_i(1, x); })},
        {"bessel_I2", scalar_func_apply<float>([](float x) -> float { return boost::math::cyl_bessel_i(2, x); })},
        {"bessel_J0", scalar_func_apply<float>([](float x) -> float { return boost::math::cyl_bessel_j(0, x); })},
        {"bessel_J1", scalar_func_apply<float>([](float x) -> float { return boost::math::cyl_bessel_j(1, x); })},
        {"bessel_J2", scalar_func_apply<float>([](float x) -> float { return boost::math::cyl_bessel_j(2, x); })},
        {"bessel_K0", scalar_func_apply<float>([](float x) -> float { return boost::math::cyl_bessel_k(0, x); })},
        {"bessel_K1", scalar_func_apply<float>([](float x) -> float { return boost::math::cyl_bessel_k(1, x); })},
        {"bessel_K2", scalar_func_apply<float>([](float x) -> float { return boost::math::cyl_bessel_k(2, x); })},
        {"bessel_j0", scalar_func_apply<float>([](float x) -> float { return boost::math::sph_bessel(0, x); })},
        {"bessel_j1", scalar_func_apply<float>([](float x) -> float { return boost::math::sph_bessel(1, x); })},
        {"bessel_j2", scalar_func_apply<float>([](float x) -> float { return boost::math::sph_bessel(2, x); })},
        {"bessel_y0", scalar_func_apply<float>([](float x) -> float { return boost::math::sph_neumann(0, x); })},
        {"bessel_y1", scalar_func_apply<float>([](float x) -> float { return boost::math::sph_neumann(1, x); })},
        {"bessel_y2", scalar_func_apply<float>([](float x) -> float { return boost::math::sph_neumann(2, x); })},
        {"hermite_0", scalar_func_apply<float>([](float x) -> float { return boost::math::hermite(0, x); })},
        {"hermite_1", scalar_func_apply<float>([](float x) -> float { return boost::math::hermite(1, x); })},
        {"hermite_2", scalar_func_apply<float>([](float x) -> float { return boost::math::hermite(2, x); })},
        {"hermite_3", scalar_func_apply<float>([](float x) -> float { return boost::math::hermite(3, x); })},
        {"riemann_zeta", scalar_func_apply<float>([](float x) -> float { return boost::math::zeta(x); })},
    };

    std::unordered_map<std::string, multi_eval_func<double>> boost_funs_dx1 = {
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

    std::unordered_map<std::string, multi_eval_func<float>> std_funs_fx1 = {
        {"tgamma", scalar_func_apply<float>([](float x) -> float { return std::tgamma(x); })},
        {"lgamma", scalar_func_apply<float>([](float x) -> float { return std::lgamma(x); })},
        {"sin", scalar_func_apply<float>([](float x) -> float { return std::sin(x); })},
        {"cos", scalar_func_apply<float>([](float x) -> float { return std::cos(x); })},
        {"tan", scalar_func_apply<float>([](float x) -> float { return std::tan(x); })},
        {"asin", scalar_func_apply<float>([](float x) -> float { return std::asin(x); })},
        {"acos", scalar_func_apply<float>([](float x) -> float { return std::acos(x); })},
        {"atan", scalar_func_apply<float>([](float x) -> float { return std::atan(x); })},
        {"asin", scalar_func_apply<float>([](float x) -> float { return std::asin(x); })},
        {"acos", scalar_func_apply<float>([](float x) -> float { return std::acos(x); })},
        {"atan", scalar_func_apply<float>([](float x) -> float { return std::atan(x); })},
        {"sinh", scalar_func_apply<float>([](float x) -> float { return std::sinh(x); })},
        {"cosh", scalar_func_apply<float>([](float x) -> float { return std::cosh(x); })},
        {"tanh", scalar_func_apply<float>([](float x) -> float { return std::tanh(x); })},
        {"asinh", scalar_func_apply<float>([](float x) -> float { return std::asinh(x); })},
        {"acosh", scalar_func_apply<float>([](float x) -> float { return std::acosh(x); })},
        {"atanh", scalar_func_apply<float>([](float x) -> float { return std::atanh(x); })},
        {"sin_pi", scalar_func_apply<float>([](float x) -> float { return std::sin(M_PI * x); })},
        {"cos_pi", scalar_func_apply<float>([](float x) -> float { return std::cos(M_PI * x); })},
        {"erf", scalar_func_apply<float>([](float x) -> float { return std::erf(x); })},
        {"erfc", scalar_func_apply<float>([](float x) -> float { return std::erfc(x); })},
        {"log", scalar_func_apply<float>([](float x) -> float { return std::log(x); })},
        {"log2", scalar_func_apply<float>([](float x) -> float { return std::log2(x); })},
        {"log10", scalar_func_apply<float>([](float x) -> float { return std::log10(x); })},
        {"exp", scalar_func_apply<float>([](float x) -> float { return std::exp(x); })},
        {"exp2", scalar_func_apply<float>([](float x) -> float { return std::exp2(x); })},
        {"exp10", scalar_func_apply<float>([](float x) -> float { return exp10(x); })},
        {"sqrt", scalar_func_apply<float>([](float x) -> float { return std::sqrt(x); })},
        {"rsqrt", scalar_func_apply<float>([](float x) -> float { return 1.0 / std::sqrt(x); })},
        {"pow3.5", scalar_func_apply<float>([](float x) -> float { return std::pow(x, 3.5); })},
        {"pow13", scalar_func_apply<float>([](float x) -> float { return std::pow(x, 13); })},
    };

    std::unordered_map<std::string, multi_eval_func<double>> std_funs_dx1 = {
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

    std::unordered_map<std::string, multi_eval_func<float>> amdlibm_funs_fx1 = {
        {"sin", scalar_func_apply<float>([&amd_sinf](float x) -> float { return amd_sinf(x); })},
        {"cos", scalar_func_apply<float>([&amd_cosf](float x) -> float { return amd_cosf(x); })},
        {"tan", scalar_func_apply<float>([&amd_tanf](float x) -> float { return amd_tanf(x); })},
        {"sinh", scalar_func_apply<float>([&amd_sinhf](float x) -> float { return amd_sinhf(x); })},
        {"cosh", scalar_func_apply<float>([&amd_coshf](float x) -> float { return amd_coshf(x); })},
        {"tanh", scalar_func_apply<float>([&amd_tanhf](float x) -> float { return amd_tanhf(x); })},
        {"asin", scalar_func_apply<float>([&amd_asinf](float x) -> float { return amd_asinf(x); })},
        {"acos", scalar_func_apply<float>([&amd_acosf](float x) -> float { return amd_acosf(x); })},
        {"atan", scalar_func_apply<float>([&amd_atanf](float x) -> float { return amd_atanf(x); })},
        {"asinh", scalar_func_apply<float>([&amd_asinhf](float x) -> float { return amd_asinhf(x); })},
        {"acosh", scalar_func_apply<float>([&amd_acoshf](float x) -> float { return amd_acoshf(x); })},
        {"atanh", scalar_func_apply<float>([&amd_atanhf](float x) -> float { return amd_atanhf(x); })},
        {"log", scalar_func_apply<float>([&amd_logf](float x) -> float { return amd_logf(x); })},
        {"log2", scalar_func_apply<float>([&amd_log2f](float x) -> float { return amd_log2f(x); })},
        {"log10", scalar_func_apply<float>([&amd_log10f](float x) -> float { return amd_log10f(x); })},
        {"exp", scalar_func_apply<float>([&amd_expf](float x) -> float { return amd_expf(x); })},
        {"exp2", scalar_func_apply<float>([&amd_exp2f](float x) -> float { return amd_exp2f(x); })},
        {"exp10", scalar_func_apply<float>([&amd_exp10f](float x) -> float { return amd_exp10f(x); })},
        {"sqrt", scalar_func_apply<float>([&amd_sqrtf](float x) -> float { return amd_sqrtf(x); })},
        {"pow3.5", scalar_func_apply<float>([&amd_powf](float x) -> float { return amd_powf(x, 3.5); })},
        {"pow13", scalar_func_apply<float>([&amd_powf](float x) -> float { return amd_powf(x, 13); })},
    };

    std::unordered_map<std::string, multi_eval_func<double>> amdlibm_funs_dx1 = {
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

    std::unordered_map<std::string, multi_eval_func<double>> amdlibm_funs_dx4 = {
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

    std::unordered_map<std::string, multi_eval_func<float>> amdlibm_funs_fx8 = {
        {"sin", vec_func_apply<Vec8f, float>([&amd_vrs8_sinf](Vec8f x) -> Vec8f { return amd_vrs8_sinf(x); })},
        {"cos", vec_func_apply<Vec8f, float>([&amd_vrs8_cosf](Vec8f x) -> Vec8f { return amd_vrs8_cosf(x); })},
        {"tan", vec_func_apply<Vec8f, float>([&amd_vrs8_tanf](Vec8f x) -> Vec8f { return amd_vrs8_tanf(x); })},
        {"log", vec_func_apply<Vec8f, float>([&amd_vrs8_logf](Vec8f x) -> Vec8f { return amd_vrs8_logf(x); })},
        {"log2", vec_func_apply<Vec8f, float>([&amd_vrs8_log2f](Vec8f x) -> Vec8f { return amd_vrs8_log2f(x); })},
        {"exp", vec_func_apply<Vec8f, float>([&amd_vrs8_expf](Vec8f x) -> Vec8f { return amd_vrs8_expf(x); })},
        {"exp2", vec_func_apply<Vec8f, float>([&amd_vrs8_exp2f](Vec8f x) -> Vec8f { return amd_vrs8_exp2f(x); })},
        {"pow3.5",
         vec_func_apply<Vec8f, float>([&amd_vrs8_powf](Vec8f x) -> Vec8f { return amd_vrs8_powf(x, Vec8f{3.5}); })},
        {"pow13",
         vec_func_apply<Vec8f, float>([&amd_vrs8_powf](Vec8f x) -> Vec8f { return amd_vrs8_powf(x, Vec8f{13}); })},
    };

    std::unordered_map<std::string, multi_eval_func<float>> sleef_funs_fx1 = {
        {"sin_pi", scalar_func_apply<float>([](float x) -> float { return Sleef_sinpif1_u05purecfma(x); })},
        {"cos_pi", scalar_func_apply<float>([](float x) -> float { return Sleef_cospif1_u05purecfma(x); })},
        {"sin", scalar_func_apply<float>([](float x) -> float { return Sleef_sinf1_u10purecfma(x); })},
        {"cos", scalar_func_apply<float>([](float x) -> float { return Sleef_cosf1_u10purecfma(x); })},
        {"tan", scalar_func_apply<float>([](float x) -> float { return Sleef_tanf1_u10purecfma(x); })},
        {"sinh", scalar_func_apply<float>([](float x) -> float { return Sleef_sinhf1_u10purecfma(x); })},
        {"cosh", scalar_func_apply<float>([](float x) -> float { return Sleef_coshf1_u10purecfma(x); })},
        {"tanh", scalar_func_apply<float>([](float x) -> float { return Sleef_tanhf1_u10purecfma(x); })},
        {"asin", scalar_func_apply<float>([](float x) -> float { return Sleef_asinf1_u10purecfma(x); })},
        {"acos", scalar_func_apply<float>([](float x) -> float { return Sleef_acosf1_u10purecfma(x); })},
        {"atan", scalar_func_apply<float>([](float x) -> float { return Sleef_atanf1_u10purecfma(x); })},
        {"asinh", scalar_func_apply<float>([](float x) -> float { return Sleef_asinhf1_u10purecfma(x); })},
        {"acosh", scalar_func_apply<float>([](float x) -> float { return Sleef_acoshf1_u10purecfma(x); })},
        {"atanh", scalar_func_apply<float>([](float x) -> float { return Sleef_atanhf1_u10purecfma(x); })},
        {"log", scalar_func_apply<float>([](float x) -> float { return Sleef_logf1_u10purecfma(x); })},
        {"log2", scalar_func_apply<float>([](float x) -> float { return Sleef_log2f1_u10purecfma(x); })},
        {"log10", scalar_func_apply<float>([](float x) -> float { return Sleef_log10f1_u10purecfma(x); })},
        {"exp", scalar_func_apply<float>([](float x) -> float { return Sleef_expf1_u10purecfma(x); })},
        {"exp2", scalar_func_apply<float>([](float x) -> float { return Sleef_exp2f1_u10purecfma(x); })},
        {"exp10", scalar_func_apply<float>([](float x) -> float { return Sleef_exp10f1_u10purecfma(x); })},
        {"erf", scalar_func_apply<float>([](float x) -> float { return Sleef_erff1_u10purecfma(x); })},
        {"erfc", scalar_func_apply<float>([](float x) -> float { return Sleef_erfcf1_u15purecfma(x); })},
        {"lgamma", scalar_func_apply<float>([](float x) -> float { return Sleef_lgammaf1_u10purecfma(x); })},
        {"tgamma", scalar_func_apply<float>([](float x) -> float { return Sleef_tgammaf1_u10purecfma(x); })},
        {"sqrt", scalar_func_apply<float>([](float x) -> float { return Sleef_sqrtf1_u05purecfma(x); })},
        {"pow3.5", scalar_func_apply<float>([](float x) -> float { return Sleef_powf1_u10purecfma(x, 3.5); })},
        {"pow13", scalar_func_apply<float>([](float x) -> float { return Sleef_powf1_u10purecfma(x, 13); })},
    };

    std::unordered_map<std::string, multi_eval_func<double>> sleef_funs_dx1 = {
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

    std::unordered_map<std::string, multi_eval_func<float>> sleef_funs_fx8 = {
        {"sin_pi", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_sinpif8_u05avx2(x); })},
        {"cos_pi", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_cospif8_u05avx2(x); })},
        {"sin", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_sinf8_u10avx2(x); })},
        {"cos", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_cosf8_u10avx2(x); })},
        {"tan", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_tanf8_u10avx2(x); })},
        {"sinh", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_sinhf8_u10avx2(x); })},
        {"cosh", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_coshf8_u10avx2(x); })},
        {"tanh", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_tanhf8_u10avx2(x); })},
        {"asin", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_asinf8_u10avx2(x); })},
        {"acos", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_acosf8_u10avx2(x); })},
        {"atan", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_atanf8_u10avx2(x); })},
        {"asinh", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_asinhf8_u10avx2(x); })},
        {"acosh", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_acoshf8_u10avx2(x); })},
        {"atanh", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_atanhf8_u10avx2(x); })},
        {"log", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_logf8_u10avx2(x); })},
        {"log2", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_log2f8_u10avx2(x); })},
        {"log10", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_log10f8_u10avx2(x); })},
        {"exp", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_expf8_u10avx2(x); })},
        {"exp2", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_exp2f8_u10avx2(x); })},
        {"exp10", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_exp10f8_u10avx2(x); })},
        {"erf", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_erff8_u10avx2(x); })},
        {"erfc", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_erfcf8_u15avx2(x); })},
        {"lgamma", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_lgammaf8_u10avx2(x); })},
        {"tlgamma", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_tgammaf8_u10avx2(x); })},
        {"sqrt", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_sqrtf8_u05avx2(x); })},
        {"pow3.5", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_powf8_u10avx2(x, Vec8f{3.5}); })},
        {"pow13", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return Sleef_powf8_u10avx2(x, Vec8f{13}); })},
    };

    std::unordered_map<std::string, multi_eval_func<double>> sleef_funs_dx4 = {
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
    std::unordered_map<std::string, multi_eval_func<float>> sleef_funs_fx16 = {
        {"sin_pi", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_sinpif16_u05avx512f(x); })},
        {"cos_pi", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_cospif16_u05avx512f(x); })},
        {"sin", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_sinf16_u10avx512f(x); })},
        {"cos", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_cosf16_u10avx512f(x); })},
        {"tan", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_tanf16_u10avx512f(x); })},
        {"sinh", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_sinhf16_u10avx512f(x); })},
        {"cosh", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_coshf16_u10avx512f(x); })},
        {"tanh", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_tanhf16_u10avx512f(x); })},
        {"asin", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_asinf16_u10avx512f(x); })},
        {"acos", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_acosf16_u10avx512f(x); })},
        {"atan", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_atanf16_u10avx512f(x); })},
        {"asinh", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_asinhf16_u10avx512f(x); })},
        {"acosh", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_acoshf16_u10avx512f(x); })},
        {"atanh", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_atanhf16_u10avx512f(x); })},
        {"log", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_logf16_u10avx512f(x); })},
        {"log2", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_log2f16_u10avx512f(x); })},
        {"log10", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_log10f16_u10avx512f(x); })},
        {"exp", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_expf16_u10avx512f(x); })},
        {"exp2", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_exp2f16_u10avx512f(x); })},
        {"exp10", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_exp10f16_u10avx512f(x); })},
        {"erf", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_erff16_u10avx512f(x); })},
        {"erfc", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_erfcf16_u15avx512f(x); })},
        {"lgamma", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_lgammaf16_u10avx512f(x); })},
        {"tlgamma", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_tgammaf16_u10avx512f(x); })},
        {"sqrt", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_sqrtf16_u05avx512f(x); })},
        {"pow3.5",
         vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_powf16_u10avx512f(x, Vec16f{3.5}); })},
        {"pow13",
         vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return Sleef_powf16_u10avx512f(x, Vec16f{13}); })},
    };

    std::unordered_map<std::string, multi_eval_func<double>> sleef_funs_dx8 = {
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

    std::unordered_map<std::string, multi_eval_func<float>> af_funs_fx8 = {
        {"sqrt", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return sqrt(x); })},
        {"sin", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return sin(x); })},
        {"cos", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return cos(x); })},
        {"tan", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return tan(x); })},
        {"sinh", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return sinh(x); })},
        {"cosh", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return cosh(x); })},
        {"tanh", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return tanh(x); })},
        {"asinh", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return asinh(x); })},
        {"acosh", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return acosh(x); })},
        {"atanh", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return atanh(x); })},
        {"asin", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return asin(x); })},
        {"acos", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return acos(x); })},
        {"atan", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return atan(x); })},
        {"exp", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return exp(x); })},
        {"exp2", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return exp2(x); })},
        {"exp10", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return exp10(x); })},
        {"log", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return log(x); })},
        {"log2", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return log2(x); })},
        {"log10", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return log10(x); })},
        {"pow3.5", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return pow(x, 3.5); })},
        {"pow13", vec_func_apply<Vec8f, float>([](Vec8f x) -> Vec8f { return pow_const(x, 13); })},
    };

    std::unordered_map<std::string, multi_eval_func<double>> af_funs_dx4 = {
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
    std::unordered_map<std::string, multi_eval_func<float>> af_funs_fx16 = {
        {"sqrt", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return sqrt(x); })},
        {"sin", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return sin(x); })},
        {"cos", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return cos(x); })},
        {"tan", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return tan(x); })},
        {"sinh", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return sinh(x); })},
        {"cosh", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return cosh(x); })},
        {"tanh", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return tanh(x); })},
        {"asinh", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return asinh(x); })},
        {"acosh", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return acosh(x); })},
        {"atanh", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return atanh(x); })},
        {"asin", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return asin(x); })},
        {"acos", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return acos(x); })},
        {"atan", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return atan(x); })},
        {"exp", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return exp(x); })},
        {"exp2", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return exp2(x); })},
        {"exp10", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return exp10(x); })},
        {"log", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return log(x); })},
        {"log2", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return log2(x); })},
        {"log10", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return log10(x); })},
        {"pow3.5", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return pow(x, 3.5); })},
        {"pow13", vec_func_apply<Vec16f, float>([](Vec16f x) -> Vec16f { return pow_const(x, 13); })},
    };

    std::unordered_map<std::string, multi_eval_func<double>> af_funs_dx8 = {
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

    std::unordered_map<std::string, multi_eval_func<float>> sctl_funs_fx8 = {
        {"copy", sctl_apply<float, 8>([](const sctl_fx8 &x) { return x; })},
        {"exp", sctl_apply<float, 8>([](const sctl_fx8 &x) { return sctl::approx_exp<7>(x); })},
        {"sin", sctl_apply<float, 8>([](const sctl_fx8 &x) {
             sctl_fx8 sinx, cosx;
             sctl::approx_sincos<7>(sinx, cosx, x);
             return sinx;
         })},
        {"cos", sctl_apply<float, 8>([](const sctl_fx8 &x) {
             sctl_fx8 sinx, cosx;
             sctl::approx_sincos<7>(sinx, cosx, x);
             return cosx;
         })},
        {"rsqrt", sctl_apply<float, 8>([](const sctl_fx8 &x) { return sctl::approx_rsqrt<7>(x); })},
    };

    std::unordered_map<std::string, multi_eval_func<float>> sctl_funs_fx16 = {
        {"copy", sctl_apply<float, 16>([](const sctl_fx16 &x) { return x; })},
        {"exp", sctl_apply<float, 16>([](const sctl_fx16 &x) { return sctl::approx_exp<7>(x); })},
        {"sin", sctl_apply<float, 16>([](const sctl_fx16 &x) {
             sctl_fx16 sinx, cosx;
             sctl::approx_sincos<7>(sinx, cosx, x);
             return sinx;
         })},
        {"cos", sctl_apply<float, 16>([](const sctl_fx16 &x) {
             sctl_fx16 sinx, cosx;
             sctl::approx_sincos<7>(sinx, cosx, x);
             return cosx;
         })},
        {"rsqrt", sctl_apply<float, 16>([](const sctl_fx16 &x) { return sctl::approx_rsqrt<7>(x); })},
    };

    std::unordered_map<std::string, multi_eval_func<double>> sctl_funs_dx4 = {
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

    std::unordered_map<std::string, multi_eval_func<double>> sctl_funs_dx8 = {
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
        {"sin", OPS::sin},         {"cos", OPS::cos},      {"tan", OPS::tan},     {"sinh", OPS::sinh},
        {"cosh", OPS::cosh},       {"tanh", OPS::tanh},    {"exp", OPS::exp},     {"log", OPS::log},
        {"log10", OPS::log10},     {"pow3.5", OPS::pow35}, {"pow13", OPS::pow13}, {"asin", OPS::asin},
        {"acos", OPS::acos},       {"atan", OPS::atan},    {"asinh", OPS::asinh}, {"atanh", OPS::atanh},
        {"acosh", OPS::acosh},     {"erf", OPS::erf},      {"erfc", OPS::erfc},   {"lgamma", OPS::lgamma},
        {"digamma", OPS::digamma}, {"ndtri", OPS::ndtri},  {"sqrt", OPS::sqrt},   {"rsqrt", OPS::rsqrt},
    };

    std::set<std::string> fun_union;
    for (auto kv : amdlibm_funs_fx1)
        fun_union.insert(kv.first);
    for (auto kv : amdlibm_funs_dx1)
        fun_union.insert(kv.first);
    for (auto kv : boost_funs_fx1)
        fun_union.insert(kv.first);
    for (auto kv : boost_funs_dx1)
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
    for (auto kv : sleef_funs_fx1)
        fun_union.insert(kv.first);
    for (auto kv : sleef_funs_dx1)
        fun_union.insert(kv.first);
    for (auto kv : std_funs_fx1)
        fun_union.insert(kv.first);
    for (auto kv : std_funs_dx1)
        fun_union.insert(kv.first);

    for (auto kv : af_funs_dx4)
        fun_union.insert(kv.first);
    for (auto kv : amdlibm_funs_dx4)
        fun_union.insert(kv.first);
    for (auto kv : amdlibm_funs_fx8)
        fun_union.insert(kv.first);
    for (auto kv : sctl_funs_dx4)
        fun_union.insert(kv.first);
    for (auto kv : sleef_funs_dx4)
        fun_union.insert(kv.first);
    for (auto kv : sleef_funs_fx8)
        fun_union.insert(kv.first);
#ifdef __AVX512F__
    for (auto kv : af_funs_dx8)
        fun_union.insert(kv.first);
    for (auto kv : sctl_funs_dx8)
        fun_union.insert(kv.first);
    for (auto kv : sleef_funs_dx8)
        fun_union.insert(kv.first);
    for (auto kv : sleef_funs_fx16)
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

    std::vector<std::pair<int, int>> run_sets;
    for (uint8_t shift = 0; shift <= 14; shift += 2)
        run_sets.push_back({1 << (10 + shift), 1 << (14 - shift)});

    for (auto &run_set : run_sets) {
        const auto &[n_eval, n_repeat] = run_set;
        std::cerr << "Running benchmark with input vector of length " << n_eval << " and " << n_repeat << " repeats.\n";
        Eigen::VectorXd vals = 0.5 * (Eigen::ArrayXd::Random(n_eval) + 1.0);
        Eigen::VectorXf fvals = vals.cast<float>();
        Eigen::VectorX<cdouble> cvals = 0.5 * (Eigen::ArrayX<cdouble>::Random(n_eval) + std::complex<double>{1.0, 1.0});

        for (auto key : keys_to_eval) {
            std::cout << test_func(key, "boost_fx1", boost_funs_fx1, params, fvals, n_repeat);
            std::cout << test_func(key, "std_fx1", std_funs_fx1, params, fvals, n_repeat);
            std::cout << test_func(key, "amdlibm_fx1", amdlibm_funs_fx1, params, fvals, n_repeat);
            std::cout << test_func(key, "amdlibm_fx8", amdlibm_funs_fx8, params, fvals, n_repeat);
            std::cout << test_func(key, "sleef_fx1", sleef_funs_fx1, params, fvals, n_repeat);
            std::cout << test_func(key, "sleef_fx8", sleef_funs_fx8, params, fvals, n_repeat);
            std::cout << test_func(key, "af_fx8", af_funs_fx8, params, fvals, n_repeat);
            std::cout << test_func(key, "sctl_fx8", sctl_funs_fx8, params, fvals, n_repeat);
            std::cout << test_func(key, "eigen_fxx", eigen_funs, params, fvals, n_repeat);
#ifdef __AVX512F__
            std::cout << test_func(key, "agnerfog_fx16", af_funs_fx16, params, fvals, n_repeat);
            std::cout << test_func(key, "sctl_fx16", sctl_funs_fx16, params, fvals, n_repeat);
            std::cout << test_func(key, "sleef_fx16", sleef_funs_fx16, params, fvals, n_repeat);
#endif

            std::cout << test_func(key, "std_dx1", std_funs_dx1, params, vals, n_repeat);
            std::cout << test_func(key, "fort_dx1", fort_funs, params, vals, n_repeat);
            std::cout << test_func(key, "amdlibm_dx1", amdlibm_funs_dx1, params, vals, n_repeat);
            std::cout << test_func(key, "boost_dx1", boost_funs_dx1, params, vals, n_repeat);
            std::cout << test_func(key, "gsl_dx1", gsl_funs, params, vals, n_repeat);
            std::cout << test_func(key, "gsl_cdx1", gsl_complex_funs, params, cvals, n_repeat);
            std::cout << test_func(key, "sleef_dx1", sleef_funs_dx1, params, vals, n_repeat);
            std::cout << test_func(key, "hank10x_dx1", hank10x_funs, params, cvals, n_repeat);
            std::cout << test_func(key, "baobzi_dx1", baobzi_funs, params, vals, n_repeat);
            std::cout << test_func(key, "eigen_dxx", eigen_funs, params, vals, n_repeat);
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
