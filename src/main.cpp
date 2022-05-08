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

const sf::utils::host_info_t host_info;
const sf::utils::library_info_t libraries_info[] = {
    {"sctl", sf::utils::get_sctl_version()},   {"baobzi", sf::utils::get_baobzi_version()},
    {"boost", sf::utils::get_boost_version()}, {"amdlibm", sf::utils::get_alm_version()},
    {"sleef", sf::utils::get_sleef_version()}, {"gsl", sf::utils::get_gsl_version()},
    {"agnerfog", sf::utils::get_af_version()}, {"baobzi", sf::utils::get_baobzi_version()},
    {"eigen", sf::utils::get_eigen_version()}, {"misc", "NA"},
};
const sf::utils::toolchain_info_t toolchain_info;

class Params {
  public:
    std::pair<double, double> domain{0.0, 1.0};
};

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

    sf::utils::timer timer;
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
    timer.stop();

    res.eval_time = timer.elapsed();

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

    sf::utils::timer timer;
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

    timer.stop();
    res.eval_time = timer.elapsed();

    return res;
}

std::set<std::string> parse_args(int argc, char *argv[]) {
    std::set<std::string> res;
    for (int i = 0; i < argc; ++i)
        res.insert(argv[i]);

    return res;
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

    auto &amdlibm_funs_dx1 = sf::functions::amd::get_funs_dx1();
    auto &amdlibm_funs_dx4 = sf::functions::amd::get_funs_dx4();
    auto &amdlibm_funs_fx1 = sf::functions::amd::get_funs_fx1();
    auto &amdlibm_funs_fx8 = sf::functions::amd::get_funs_fx8();

    auto &gsl_funs = sf::functions::gsl::get_funs_dx1();
    auto &gsl_complex_funs = sf::functions::gsl::get_funs_cdx1();

    auto &boost_funs_fx1 = sf::functions::boost::get_funs_fx1();
    auto &boost_funs_dx1 = sf::functions::boost::get_funs_dx1();

    auto &stl_funs_fx1 = sf::functions::stl::get_funs_fx1();
    auto &stl_funs_dx1 = sf::functions::stl::get_funs_dx1();

    auto &sleef_funs_dx1 = sf::functions::sleef::get_funs_dx1();
    auto &sleef_funs_dx4 = sf::functions::sleef::get_funs_dx4();
    auto &sleef_funs_dx8 = sf::functions::sleef::get_funs_dx8();
    auto &sleef_funs_fx1 = sf::functions::sleef::get_funs_fx1();
    auto &sleef_funs_fx8 = sf::functions::sleef::get_funs_fx8();
    auto &sleef_funs_fx16 = sf::functions::sleef::get_funs_fx16();

    auto &af_funs_dx4 = sf::functions::af::get_funs_dx4();
    auto &af_funs_dx8 = sf::functions::af::get_funs_dx8();
    auto &af_funs_fx8 = sf::functions::af::get_funs_fx8();
    auto &af_funs_fx16 = sf::functions::af::get_funs_fx16();

    auto &sctl_funs_dx4 = sf::functions::SCTL::get_funs_dx4();
    auto &sctl_funs_dx8 = sf::functions::SCTL::get_funs_dx8();
    auto &sctl_funs_fx8 = sf::functions::SCTL::get_funs_fx8();
    auto &sctl_funs_fx16 = sf::functions::SCTL::get_funs_fx16();

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
    for (auto kv : stl_funs_fx1)
        fun_union.insert(kv.first);
    for (auto kv : stl_funs_dx1)
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
    for (auto kv : af_funs_dx8)
        fun_union.insert(kv.first);
    for (auto kv : sctl_funs_dx8)
        fun_union.insert(kv.first);
    for (auto kv : sleef_funs_dx8)
        fun_union.insert(kv.first);
    for (auto kv : sleef_funs_fx16)
        fun_union.insert(kv.first);

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
            std::cout << test_func(key, "stl_fx1", stl_funs_fx1, params, fvals, n_repeat);
            std::cout << test_func(key, "amdlibm_fx1", amdlibm_funs_fx1, params, fvals, n_repeat);
            std::cout << test_func(key, "amdlibm_fx8", amdlibm_funs_fx8, params, fvals, n_repeat);
            std::cout << test_func(key, "sleef_fx1", sleef_funs_fx1, params, fvals, n_repeat);
            std::cout << test_func(key, "sleef_fx8", sleef_funs_fx8, params, fvals, n_repeat);
            std::cout << test_func(key, "af_fx8", af_funs_fx8, params, fvals, n_repeat);
            std::cout << test_func(key, "sctl_fx8", sctl_funs_fx8, params, fvals, n_repeat);
            std::cout << test_func(key, "eigen_fxx", eigen_funs, params, fvals, n_repeat);
            std::cout << test_func(key, "agnerfog_fx16", af_funs_fx16, params, fvals, n_repeat);
            std::cout << test_func(key, "sctl_fx16", sctl_funs_fx16, params, fvals, n_repeat);
            std::cout << test_func(key, "sleef_fx16", sleef_funs_fx16, params, fvals, n_repeat);

            std::cout << test_func(key, "stl_dx1", stl_funs_dx1, params, vals, n_repeat);
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
            std::cout << test_func(key, "agnerfog_dx8", af_funs_dx8, params, vals, n_repeat);
            std::cout << test_func(key, "sctl_dx8", sctl_funs_dx8, params, vals, n_repeat);
            std::cout << test_func(key, "sleef_dx8", sleef_funs_dx8, params, vals, n_repeat);

            std::cout << "\n";
        }
    }
    return 0;
}
