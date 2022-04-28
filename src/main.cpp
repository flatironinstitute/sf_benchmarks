#include "boost/math/special_functions/math_fwd.hpp"
#include "gsl/gsl_sf_gamma.h"
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

typedef double (*fun_1d)(double);

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

    fun_1d f = funs.at(name);
    BenchResult res(label, vals.size());

    const struct timespec st = get_wtime();
    for (std::size_t i = 0; i < vals.size(); ++i)
        res[i] = f(vals[i]);
    const struct timespec ft = get_wtime();

    res.eval_time = get_wtime_diff(&st, &ft);

    return res;
}

double boost_bessel_Y0(double x) { return boost::math::cyl_neumann(0, x); }
double boost_bessel_K0(double x) { return boost::math::cyl_bessel_k(0, x); }
double boost_bessel_I0(double x) { return boost::math::cyl_bessel_i(0, x); }
double boost_bessel_J0(double x) { return boost::math::cyl_bessel_j(0, x); }
double boost_bessel_j0(double x) { return boost::math::sph_bessel(0, x); }
double boost_bessel_y0(double x) { return boost::math::sph_neumann(0, x); }

int main(int argc, char *argv[]) {
    std::unordered_map<std::string, fun_1d> gsl_funs = {{"tgamma", gsl_sf_gamma},
                                                        {"lgamma", gsl_sf_lngamma},
                                                        {"bessel_Y0", gsl_sf_bessel_Y0},
                                                        {"bessel_I0", gsl_sf_bessel_I0},
                                                        {"bessel_J0", gsl_sf_bessel_J0},
                                                        {"bessel_K0", gsl_sf_bessel_K0},
                                                        {"bessel_j0", gsl_sf_bessel_j0},
                                                        {"bessel_y0", gsl_sf_bessel_y0}
    };
    std::unordered_map<std::string, fun_1d> boost_funs = {{"tgamma", boost::math::tgamma<double>},
                                                          {"lgamma", boost::math::lgamma},
                                                          {"erf", boost::math::erf},
                                                          {"erfc", boost::math::erfc},
                                                          {"bessel_Y0", boost_bessel_Y0},
                                                          {"bessel_I0", boost_bessel_I0},
                                                          {"bessel_J0", boost_bessel_J0},
                                                          {"bessel_K0", boost_bessel_K0},
                                                          {"bessel_j0", boost_bessel_j0},
                                                          {"bessel_y0", boost_bessel_y0},
    };
    std::unordered_map<std::string, fun_1d> std_funs = {{"tgamma", std::tgamma},
                                                        {"lgamma", std::lgamma},
                                                        {"erf", std::erf},
                                                        {"erfc", std::erfc},
    };
    std::unordered_map<std::string, fun_1d> sctl_funs;
    std::unordered_map<std::string, fun_1d> sleef_funs;

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
