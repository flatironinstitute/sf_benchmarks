#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <iostream>
#include <set>
#include <sstream>
#include <type_traits>
#include <unordered_map>

#include <Eigen/Core>
#include <boost/math/special_functions.hpp>
#include <cmath>
#include <gsl/gsl_sf.h>
#include <sctl.hpp>
#include <sleef.h>
#include <unsupported/Eigen/SpecialFunctions>

#include <time.h>

struct timespec get_wtime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

double get_wtime_diff(const struct timespec *ts, const struct timespec *tf) {
    return (tf->tv_sec - ts->tv_sec) + (tf->tv_nsec - ts->tv_nsec) * 1E-9;
}

typedef __m256d sleef_dx4;
typedef __m512d sleef_dx8;
typedef sctl::Vec<double, 4> sctl_dx4;
typedef sctl::Vec<double, 8> sctl_dx8;

typedef std::function<double(double)> fun_dx1;
typedef std::function<sctl_dx4::VData(const sctl_dx4::VData &)> sctl_fun_dx4;
typedef std::function<sctl_dx8::VData(const sctl_dx8::VData &)> sctl_fun_dx8;
typedef std::function<sleef_dx4(sleef_dx4)> sleef_fun_dx4;
typedef std::function<sleef_dx8(sleef_dx8)> sleef_fun_dx8;

class BenchResult {
  public:
    std::vector<double> res;
    double eval_time = 0.0;
    std::string label;

    BenchResult(const std::string &label_) : label(label_){};
    BenchResult(const std::string &label_, std::size_t size) : res(size), label(label_){};

    double &operator[](int i) { return res[i]; }
    double Mevals() const { return res.size() / eval_time / 1E6; }

    friend std::ostream &operator<<(std::ostream &, const BenchResult &);
};

std::ostream &operator<<(std::ostream &os, const BenchResult &br) {
    double mean = 0.0;
    for (const auto &v : br.res)
        mean += v;
    mean /= br.res.size();

    using std::left;
    using std::setw;
    if (br.res.size()) {
        os.precision(6);
        os << left << setw(20) << br.label + ": " << left << setw(15) << br.Mevals();
        os.precision(15);
        os << left << setw(15) << mean;
    } else
        os << left << setw(20) << br.label + ": " << setw(15) << "NA" << setw(15) << "NA";
    return os;
}

template <typename FUN_T>
BenchResult test_func(const std::string name, const std::string library_prefix,
                      const std::unordered_map<std::string, FUN_T> funs, const std::vector<double> &vals) {
    const std::string label = library_prefix + "_" + name;
    if (!funs.count(name))
        return BenchResult(label);

    BenchResult res(label, vals.size());
    const FUN_T &f = funs.at(name);

    // Load a big thing to clear cache. No idea why compiler isn't optimizing this away.
    std::vector<char> c(200 * 1024 * 1024);
    for (int j = 0; j < c.size(); j++)
        c[j] = j;

    const struct timespec st = get_wtime();
    if constexpr (std::is_same_v<FUN_T, fun_dx1>) {
        for (std::size_t i = 0; i < vals.size(); ++i)
            res[i] = f(vals[i]);
    } else if constexpr (std::is_same_v<FUN_T, sctl_fun_dx4>) {
        for (std::size_t i = 0; i < vals.size(); i += 4) {
            sctl_dx4 x = sctl_dx4::Load(vals.data() + i);
            sctl_dx4(f(x.get())).Store(res.res.data() + i);
        }
    } else if constexpr (std::is_same_v<FUN_T, sctl_fun_dx8>) {
        for (std::size_t i = 0; i < vals.size(); i += 8) {
            sctl_dx8 x = sctl_dx8::Load(vals.data() + i);
            sctl_dx8(f(x.get())).Store(res.res.data() + i);
        }
    } else if constexpr (std::is_same_v<FUN_T, sleef_fun_dx4>) {
        for (std::size_t i = 0; i < vals.size(); i += 4) {
            sctl_dx4 x = sctl_dx4::Load(vals.data() + i);
            _mm256_store_pd(res.res.data() + i, f(x.get().v));
        }
    } else if constexpr (std::is_same_v<FUN_T, sleef_fun_dx8>) {
        for (std::size_t i = 0; i < vals.size(); i += 8) {
            sctl_dx8 x = sctl_dx8::Load(vals.data() + i);
            _mm512_store_pd(res.res.data() + i, f(x.get().v));
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
BenchResult test_func(const std::string name, const std::string library_prefix,
                      const std::unordered_map<std::string, OPS::OPS> funs, const std::vector<double> &vals) {
    Eigen::Map<const Eigen::ArrayXd> x(vals.data(), vals.size());

    const std::string label = library_prefix + "_" + name;
    if (!funs.count(name))
        return BenchResult(label);

    BenchResult res(label, vals.size());

    Eigen::Map<Eigen::VectorXd> res_eigen(res.res.data(), vals.size());

    OPS::OPS OP = funs.at(name);
    const struct timespec st = get_wtime();

    switch (OP) {
    case OPS::COS:
        res_eigen = x.cos();
        break;
    case OPS::SIN:
        res_eigen = x.sin();
        break;
    case OPS::TAN:
        res_eigen = x.tan();
        break;
    case OPS::COSH:
        res_eigen = x.cosh();
        break;
    case OPS::SINH:
        res_eigen = x.sinh();
        break;
    case OPS::TANH:
        res_eigen = x.tanh();
        break;
    case OPS::EXP:
        res_eigen = x.exp();
        break;
    case OPS::LOG:
        res_eigen = x.log();
        break;
    case OPS::LOG10:
        res_eigen = x.log10();
        break;
    case OPS::POW35:
        res_eigen = x.pow(3.5);
        break;
    case OPS::POW13:
        res_eigen = x.pow(13);
        break;
    case OPS::ASIN:
        res_eigen = x.asin();
        break;
    case OPS::ACOS:
        res_eigen = x.acos();
        break;
    case OPS::ATAN:
        res_eigen = x.atan();
        break;
    case OPS::ASINH:
        res_eigen = x.asinh();
        break;
    case OPS::ACOSH:
        res_eigen = x.acosh();
        break;
    case OPS::ATANH:
        res_eigen = x.atanh();
        break;
    case OPS::ERF:
        res_eigen = x.erf();
        break;
    case OPS::ERFC:
        res_eigen = x.erfc();
        break;
    case OPS::LGAMMA:
        res_eigen = x.lgamma();
        break;
    case OPS::DIGAMMA:
        res_eigen = x.digamma();
        break;
    case OPS::NDTRI:
        res_eigen = x.ndtri();
        break;
    case OPS::SQRT:
        res_eigen = x.sqrt();
        break;
    case OPS::RSQRT:
        res_eigen = x.rsqrt();
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

int main(int argc, char *argv[]) {
    std::set<std::string> input_keys = parse_args(argc - 1, argv + 1);

    std::unordered_map<std::string, fun_dx1> gsl_funs = {
        {"sin_pi", gsl_sf_sin_pi},
        {"cos_pi", gsl_sf_cos_pi},
        {"sin", gsl_sf_sin},
        {"cos", gsl_sf_cos},
        {"sinc", gsl_sf_sinc},
        {"sinc_pi", [](double x) -> double { return gsl_sf_sinc(M_PI * x); }},
        {"erf", gsl_sf_erf},
        {"erfc", gsl_sf_erfc},
        {"tgamma", gsl_sf_gamma},
        {"lgamma", gsl_sf_lngamma},
        {"log", gsl_sf_log},
        {"exp", gsl_sf_exp},
        {"pow13", [](double x) -> double { return gsl_sf_pow_int(x, 13); }},
        {"bessel_Y0", gsl_sf_bessel_Y0},
        {"bessel_Y1", gsl_sf_bessel_Y1},
        {"bessel_Y2", [](double x) -> double { return gsl_sf_bessel_Yn(2, x); }},
        {"bessel_I0", gsl_sf_bessel_I0},
        {"bessel_I1", gsl_sf_bessel_I1},
        {"bessel_I2", [](double x) -> double { return gsl_sf_bessel_In(2, x); }},
        {"bessel_J0", gsl_sf_bessel_J0},
        {"bessel_J1", gsl_sf_bessel_J1},
        {"bessel_J2", [](double x) -> double { return gsl_sf_bessel_Jn(2, x); }},
        {"bessel_K0", gsl_sf_bessel_K0},
        {"bessel_K1", gsl_sf_bessel_K1},
        {"bessel_K2", [](double x) -> double { return gsl_sf_bessel_Kn(2, x); }},
        {"bessel_j0", gsl_sf_bessel_j0},
        {"bessel_j1", gsl_sf_bessel_j1},
        {"bessel_j2", gsl_sf_bessel_j2},
        {"bessel_y0", gsl_sf_bessel_y0},
        {"bessel_y1", gsl_sf_bessel_y1},
        {"bessel_y2", gsl_sf_bessel_y2},
        {"hermite_0", [](double x) -> double { return gsl_sf_hermite(0, x); }},
        {"hermite_1", [](double x) -> double { return gsl_sf_hermite(1, x); }},
        {"hermite_2", [](double x) -> double { return gsl_sf_hermite(2, x); }},
        {"hermite_3", [](double x) -> double { return gsl_sf_hermite(3, x); }},
        {"riemann_zeta", gsl_sf_zeta},
    };
    std::unordered_map<std::string, fun_dx1> boost_funs = {
        {"sin_pi", [](double x) -> double { return boost::math::sin_pi(x); }},
        {"cos_pi", [](double x) -> double { return boost::math::cos_pi(x); }},
        {"tgamma", [](double x) -> double { return boost::math::tgamma<double>(x); }},
        {"lgamma", [](double x) -> double { return boost::math::lgamma<double>(x); }},
        {"digamma", [](double x) -> double { return boost::math::digamma<double>(x); }},
        {"pow13", [](double x) -> double { return boost::math::pow<13>(x); }},
        {"erf", [](double x) -> double { return boost::math::erf(x); }},
        {"erfc", [](double x) -> double { return boost::math::erfc(x); }},
        {"sinc_pi", [](double x) -> double { return boost::math::sinc_pi(x); }},
        {"bessel_Y0", [](double x) -> double { return boost::math::cyl_neumann(0, x); }},
        {"bessel_Y1", [](double x) -> double { return boost::math::cyl_neumann(1, x); }},
        {"bessel_Y2", [](double x) -> double { return boost::math::cyl_neumann(2, x); }},
        {"bessel_I0", [](double x) -> double { return boost::math::cyl_bessel_i(0, x); }},
        {"bessel_I1", [](double x) -> double { return boost::math::cyl_bessel_i(1, x); }},
        {"bessel_I2", [](double x) -> double { return boost::math::cyl_bessel_i(2, x); }},
        {"bessel_J0", [](double x) -> double { return boost::math::cyl_bessel_j(0, x); }},
        {"bessel_J1", [](double x) -> double { return boost::math::cyl_bessel_j(1, x); }},
        {"bessel_J2", [](double x) -> double { return boost::math::cyl_bessel_j(2, x); }},
        {"bessel_K0", [](double x) -> double { return boost::math::cyl_bessel_k(0, x); }},
        {"bessel_K1", [](double x) -> double { return boost::math::cyl_bessel_k(1, x); }},
        {"bessel_K2", [](double x) -> double { return boost::math::cyl_bessel_k(2, x); }},
        {"bessel_j0", [](double x) -> double { return boost::math::sph_bessel(0, x); }},
        {"bessel_j1", [](double x) -> double { return boost::math::sph_bessel(1, x); }},
        {"bessel_j2", [](double x) -> double { return boost::math::sph_bessel(2, x); }},
        {"bessel_y0", [](double x) -> double { return boost::math::sph_neumann(0, x); }},
        {"bessel_y1", [](double x) -> double { return boost::math::sph_neumann(1, x); }},
        {"bessel_y2", [](double x) -> double { return boost::math::sph_neumann(2, x); }},
        {"hermite_0", [](double x) -> double { return boost::math::hermite(0, x); }},
        {"hermite_1", [](double x) -> double { return boost::math::hermite(1, x); }},
        {"hermite_2", [](double x) -> double { return boost::math::hermite(2, x); }},
        {"hermite_3", [](double x) -> double { return boost::math::hermite(3, x); }},
        {"hermite_3", [](double x) -> double { return boost::math::hermite(3, x); }},
        {"riemann_zeta", [](double x) -> double { return boost::math::zeta(x); }},
    };
    std::unordered_map<std::string, fun_dx1> std_funs = {
        {"tgamma", [](double x) -> double { return std::tgamma(x); }},
        {"lgamma", [](double x) -> double { return std::lgamma(x); }},
        {"sin", [](double x) -> double { return std::sin(x); }},
        {"cos", [](double x) -> double { return std::cos(x); }},
        {"tan", [](double x) -> double { return std::tan(x); }},
        {"asin", [](double x) -> double { return std::asin(x); }},
        {"acos", [](double x) -> double { return std::acos(x); }},
        {"atan", [](double x) -> double { return std::atan(x); }},
        {"asin", [](double x) -> double { return std::asin(x); }},
        {"acos", [](double x) -> double { return std::acos(x); }},
        {"atan", [](double x) -> double { return std::atan(x); }},
        {"sinh", [](double x) -> double { return std::sinh(x); }},
        {"cosh", [](double x) -> double { return std::cosh(x); }},
        {"tanh", [](double x) -> double { return std::tanh(x); }},
        {"asinh", [](double x) -> double { return std::asinh(x); }},
        {"acosh", [](double x) -> double { return std::acosh(x); }},
        {"atanh", [](double x) -> double { return std::atanh(x); }},
        {"sin_pi", [](double x) -> double { return std::sin(M_PI * x); }},
        {"cos_pi", [](double x) -> double { return std::cos(M_PI * x); }},
        {"erf", [](double x) -> double { return std::erf(x); }},
        {"erfc", [](double x) -> double { return std::erfc(x); }},
        {"log", [](double x) -> double { return std::log(x); }},
        {"log2", [](double x) -> double { return std::log2(x); }},
        {"log10", [](double x) -> double { return std::log10(x); }},
        {"exp", [](double x) -> double { return std::exp(x); }},
        {"exp2", [](double x) -> double { return std::exp2(x); }},
        {"exp10", [](double x) -> double { return exp10(x); }},
        {"sqrt", [](double x) -> double { return std::sqrt(x); }},
        {"rsqrt", [](double x) -> double { return 1.0 / std::sqrt(x); }},
        {"pow3.5", [](double x) -> double { return std::pow(x, 3.5); }},
        {"pow13", [](double x) -> double { return std::pow(x, 13); }},
    };
    std::unordered_map<std::string, fun_dx1> sleef_funs = {
        {"sin_pi", Sleef_sinpid1_u05purecfma},
        {"cos_pi", Sleef_cospid1_u05purecfma},
        {"sin", Sleef_sind1_u10purecfma},
        {"cos", Sleef_cosd1_u10purecfma},
        {"tan", Sleef_tand1_u10purecfma},
        {"sinh", Sleef_sinhd1_u10purecfma},
        {"cosh", Sleef_coshd1_u10purecfma},
        {"tanh", Sleef_tanhd1_u10purecfma},
        {"asin", Sleef_asind1_u10purecfma},
        {"acos", Sleef_acosd1_u10purecfma},
        {"atan", Sleef_atand1_u10purecfma},
        {"asinh", Sleef_asinhd1_u10purecfma},
        {"acosh", Sleef_acoshd1_u10purecfma},
        {"atanh", Sleef_atanhd1_u10purecfma},
        {"log", Sleef_logd1_u10purecfma},
        {"log", Sleef_log2d1_u10purecfma},
        {"log10", Sleef_log10d1_u10purecfma},
        {"exp", Sleef_expd1_u10purecfma},
        {"exp2", Sleef_exp2d1_u10purecfma},
        {"exp10", Sleef_exp10d1_u10purecfma},
        {"erf", Sleef_erfd1_u10purecfma},
        {"erfc", Sleef_erfcd1_u15purecfma},
        {"lgamma", Sleef_lgammad1_u10purecfma},
        {"tgamma", Sleef_tgammad1_u10purecfma},
        {"sqrt", Sleef_sqrtd1_u05purecfma},
        {"pow3.5", [](double x) -> double { return Sleef_powd1_u10purecfma(x, 3.5); }},
        {"pow13", [](double x) -> double { return Sleef_powd1_u10purecfma(x, 13); }},
    };
    std::unordered_map<std::string, sleef_fun_dx4> sleef_funs_dx4 = {
        {"sin_pi", Sleef_sinpid4_u05avx2},
        {"cos_pi", Sleef_cospid4_u05avx2},
        {"sin", Sleef_sind4_u10avx2},
        {"cos", Sleef_cosd4_u10avx2},
        {"tan", Sleef_tand4_u10avx2},
        {"sinh", Sleef_sinhd4_u10avx2},
        {"cosh", Sleef_coshd4_u10avx2},
        {"tanh", Sleef_tanhd4_u10avx2},
        {"asin", Sleef_asind4_u10avx2},
        {"acos", Sleef_acosd4_u10avx2},
        {"atan", Sleef_atand4_u10avx2},
        {"asinh", Sleef_asinhd4_u10avx2},
        {"acosh", Sleef_acoshd4_u10avx2},
        {"atanh", Sleef_atanhd4_u10avx2},
        {"log", Sleef_logd4_u10avx2},
        {"log", Sleef_log2d4_u10avx2},
        {"log10", Sleef_log10d4_u10avx2},
        {"exp", Sleef_expd4_u10avx2},
        {"exp2", Sleef_exp2d4_u10avx2},
        {"exp10", Sleef_exp10d4_u10avx2},
        {"erf", Sleef_erfd4_u10avx2},
        {"erfc", Sleef_erfcd4_u15avx2},
        {"lgamma", Sleef_lgammad4_u10avx2},
        {"tgamma", Sleef_tgammad4_u10avx2},
        {"sqrt", Sleef_sqrtd4_u05avx2},
        {"pow3.5", [](sleef_dx4 x) -> sleef_dx4 { return Sleef_powd4_u10avx2(x, sleef_dx4{3.5}); }},
        {"pow13", [](sleef_dx4 x) -> sleef_dx4 { return Sleef_powd4_u10avx2(x, sleef_dx4{13}); }},
    };
    std::unordered_map<std::string, sleef_fun_dx8> sleef_funs_dx8 = {
        {"sin_pi", Sleef_sinpid8_u05avx512f},
        {"cos_pi", Sleef_cospid8_u05avx512f},
        {"sin", Sleef_sind8_u10avx512f},
        {"cos", Sleef_cosd8_u10avx512f},
        {"tan", Sleef_tand8_u10avx512f},
        {"sinh", Sleef_sinhd8_u10avx512f},
        {"cosh", Sleef_coshd8_u10avx512f},
        {"tanh", Sleef_tanhd8_u10avx512f},
        {"asin", Sleef_asind8_u10avx512f},
        {"acos", Sleef_acosd8_u10avx512f},
        {"atan", Sleef_atand8_u10avx512f},
        {"asinh", Sleef_asinhd8_u10avx512f},
        {"acosh", Sleef_acoshd8_u10avx512f},
        {"atanh", Sleef_atanhd8_u10avx512f},
        {"log", Sleef_logd8_u10avx512f},
        {"log", Sleef_log2d8_u10avx512f},
        {"log10", Sleef_log10d8_u10avx512f},
        {"exp", Sleef_expd8_u10avx512f},
        {"exp2", Sleef_exp2d8_u10avx512f},
        {"exp10", Sleef_exp10d8_u10avx512f},
        {"erf", Sleef_erfd8_u10avx512f},
        {"erfc", Sleef_erfcd8_u15avx512f},
        {"lgamma", Sleef_lgammad8_u10avx512f},
        {"tgamma", Sleef_tgammad8_u10avx512f},
        {"sqrt", Sleef_sqrtd8_u05avx512f},
        {"pow3.5", [](sleef_dx8 x) -> sleef_dx8 { return Sleef_powd8_u10avx512f(x, sleef_dx8{3.5}); }},
        {"pow13", [](sleef_dx8 x) -> sleef_dx8 { return Sleef_powd8_u10avx512f(x, sleef_dx8{13}); }},
    };
    std::unordered_map<std::string, sctl_fun_dx4> sctl_funs_dx4 = {
        {"exp", sctl::exp_intrin<sctl_dx4::VData>},
    };
    std::unordered_map<std::string, sctl_fun_dx8> sctl_funs_dx8 = {
        {"exp", sctl::exp_intrin<sctl_dx8::VData>},
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
    for (auto kv : gsl_funs)
        fun_union.insert(kv.first);
    for (auto kv : boost_funs)
        fun_union.insert(kv.first);
    for (auto kv : std_funs)
        fun_union.insert(kv.first);
    for (auto kv : sctl_funs_dx4)
        fun_union.insert(kv.first);
    for (auto kv : sctl_funs_dx8)
        fun_union.insert(kv.first);
    for (auto kv : sleef_funs)
        fun_union.insert(kv.first);
    for (auto kv : sleef_funs_dx4)
        fun_union.insert(kv.first);
    for (auto kv : sleef_funs_dx8)
        fun_union.insert(kv.first);
    for (auto kv : eigen_funs)
        fun_union.insert(kv.first);

    std::set<std::string> keys_to_eval;
    if (input_keys.size() > 0)
        std::set_intersection(fun_union.begin(), fun_union.end(), input_keys.begin(), input_keys.end(),
                              std::inserter(keys_to_eval, keys_to_eval.end()));
    else
        keys_to_eval = fun_union;

    std::vector<double> vals(1000000);
    srand(100);
    for (auto &val : vals)
        val = rand() / (double)RAND_MAX;

    for (auto key : keys_to_eval) {
        std::cout << test_func(key, "std", std_funs, vals) << std::endl;
        std::cout << test_func(key, "boost", boost_funs, vals) << std::endl;
        std::cout << test_func(key, "gsl", gsl_funs, vals) << std::endl;
        std::cout << test_func(key, "sleef", sleef_funs, vals) << std::endl;
        std::cout << test_func(key, "sleef_dx4", sleef_funs_dx4, vals) << std::endl;
        std::cout << test_func(key, "sleef_dx8", sleef_funs_dx8, vals) << std::endl;
        std::cout << test_func(key, "sctl_dx4", sctl_funs_dx4, vals) << std::endl;
        std::cout << test_func(key, "sctl_dx8", sctl_funs_dx8, vals) << std::endl;
        std::cout << test_func(key, "eigen", eigen_funs, vals);
        std::cout << "\n\n";
    }

    return 0;
}
