#ifndef SF_LIBRARIES_HPP
#define SF_LIBRARIES_HPP

#include <sf_benchmarks.hpp>

#include <string>
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

#include <boost/version.hpp>
#include <gnu/libc-version.h>
#include <gsl/gsl_version.h>

extern "C" {
void hank103_(double _Complex *, double _Complex *, double _Complex *, int *);
void fort_bessel_jn_(int *, double *, double *);
void fort_bessel_yn_(int *, double *, double *);
}

namespace sf::functions {
namespace af {
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx8();
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx4();
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx16();
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx8();
} // namespace af

namespace amd {
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx1();
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx8();
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx1();
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx4();
} // namespace amd

namespace baobzi {
std::unordered_map<std::string, std::shared_ptr<::baobzi::Baobzi>> &
get_funs_dx1(std::set<std::string> &keys_to_eval, std::unordered_map<std::string, configuration_t> &configs);
}

namespace boost {
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx1();
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx1();
} // namespace boost

// https://eigen.tuxfamily.org/dox/group__CoeffwiseMathFunctions.html
namespace eigen {
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

std::unordered_map<std::string, OPS> &get_funs();
} // namespace eigen

namespace fort {
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx1();
} // namespace fort

namespace gsl {
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx1();
std::unordered_map<std::string, multi_eval_func<cdouble>> &get_funs_cdx1();
} // namespace gsl

namespace misc {
std::unordered_map<std::string, fun_cdx1_x2> &get_funs_cdx1_x2();
} // namespace misc

namespace SCTL {
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx8();
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx4();
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx16();
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx8();
} // namespace SCTL

namespace sleef {
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx1();
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx1();
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx8();
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx4();
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx16();
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx8();
} // namespace sleef

namespace stl {
std::unordered_map<::std::string, multi_eval_func<float>> &get_funs_fx1();
std::unordered_map<::std::string, multi_eval_func<double>> &get_funs_dx1();
} // namespace stl

} // namespace sf::functions

#endif
