#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <boost/math/special_functions.hpp>
#include <cmath>
#include <gsl/gsl_sf.h>
#include <sctl.hpp>
#include <sleef.h>

#include <time.h>

struct timespec get_wtime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

double get_wtime_diff(const struct timespec *ts, const struct timespec *tf) {
    return (tf->tv_sec - ts->tv_sec) + (tf->tv_nsec - ts->tv_nsec) * 1E-9;
}

typedef std::function<double(double)> fun_1d;

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

BenchResult test_func(const std::string name, const std::string library_prefix,
                      const std::unordered_map<std::string, fun_1d> funs, const std::vector<double> &vals) {
    const std::string label = library_prefix + "_" + name;
    if (!funs.count(name))
        return BenchResult(label);

    const fun_1d &f = funs.at(name);
    BenchResult res(label, vals.size());

    const struct timespec st = get_wtime();
    for (std::size_t i = 0; i < vals.size(); ++i)
        res[i] = f(vals[i]);
    const struct timespec ft = get_wtime();

    res.eval_time = get_wtime_diff(&st, &ft);

    return res;
}

int main(int argc, char *argv[]) {
    std::unordered_map<std::string, fun_1d> gsl_funs = {
        {"sin_pi", gsl_sf_sin_pi},
        {"cos_pi", gsl_sf_cos_pi},
        {"erf", gsl_sf_erf},
        {"erfc", gsl_sf_erfc},
        {"tgamma", gsl_sf_gamma},
        {"lgamma", gsl_sf_lngamma},
        {"log", gsl_sf_log},
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
    std::unordered_map<std::string, fun_1d> boost_funs = {
        {"sin_pi", [](double x) -> double { return boost::math::sin_pi(x); }},
        {"cos_pi", [](double x) -> double { return boost::math::cos_pi(x); }},
        {"tgamma", [](double x) -> double { return boost::math::tgamma<double>(x); }},
        {"lgamma", [](double x) -> double { return boost::math::lgamma<double>(x); }},
        {"log", [](double x) -> double { return boost::math::log1p(x - 1); }},
        {"pow13", [](double x) -> double { return boost::math::pow<13>(x); }},
        {"erf", [](double x) -> double { return boost::math::erf(x); }},
        {"erfc", [](double x) -> double { return boost::math::erfc(x); }},
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
    std::unordered_map<std::string, fun_1d> std_funs = {
        {"tgamma", [](double x) -> double { return std::tgamma(x); }},
        {"lgamma", [](double x) -> double { return std::lgamma(x); }},
        {"erf", [](double x) -> double { return std::erf(x); }},
        {"erfc", [](double x) -> double { return std::erfc(x); }},
        {"log", [](double x) -> double { return std::log(x); }},
        {"log10", [](double x) -> double { return std::log10(x); }},
        {"pow3.5", [](double x) -> double { return std::pow(x, 3.5); }},
        {"pow13", [](double x) -> double { return std::pow(x, 13); }},
    };
    std::unordered_map<std::string, fun_1d> sctl_funs;
    std::unordered_map<std::string, fun_1d> sleef_funs = {
        {"sin_pi", Sleef_sinpid1_u05purecfma},
        {"cos_pi", Sleef_cospid1_u05purecfma},
        {"log", Sleef_logd1_u10purecfma},
        {"log10", Sleef_log10d1_u10purecfma},
        {"erf", Sleef_erfd1_u10purecfma},
        {"erfc", Sleef_erfcd1_u15purecfma},
        {"lgamma", Sleef_lgammad1_u10purecfma},
        {"tgamma", Sleef_tgammad1_u10purecfma},
        {"pow3.5", [](double x) -> double { return Sleef_powd1_u10purecfma(x, 3.5); }},
        {"pow13", [](double x) -> double { return Sleef_powd1_u10purecfma(x, 13); }},
    };

    std::unordered_set<std::string> fun_union;
    for (auto kv : gsl_funs)
        fun_union.insert(kv.first);
    for (auto kv : boost_funs)
        fun_union.insert(kv.first);
    for (auto kv : std_funs)
        fun_union.insert(kv.first);
    for (auto kv : sctl_funs)
        fun_union.insert(kv.first);
    for (auto kv : sleef_funs)
        fun_union.insert(kv.first);

    std::vector<double> vals(1000000);
    srand(100);
    for (auto &val : vals)
        val = rand() / (double)RAND_MAX;

    for (auto key : fun_union) {
        std::cout << test_func(key, "gsl", gsl_funs, vals) << std::endl;
        std::cout << test_func(key, "boost", boost_funs, vals) << std::endl;
        std::cout << test_func(key, "sleef", sleef_funs, vals) << std::endl;
        std::cout << test_func(key, "std", std_funs, vals) << std::endl;
        std::cout << test_func(key, "sctl", sctl_funs, vals) << std::endl;
        std::cout << "\n\n";
    }
}
