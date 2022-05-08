#ifndef SF_LIBRARIES_HPP
#define SF_LIBRARIES_HPP

extern "C" {
void hank103_(double _Complex *, double _Complex *, double _Complex *, int *);
void fort_bessel_jn_(int *, double *, double *);
void fort_bessel_yn_(int *, double *, double *);
}

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

#include <sf_eval.hpp>

namespace sf::functions {
namespace amd {
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx1();
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx8();
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx1();
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx4();
} // namespace amd

namespace gsl {
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx1();
std::unordered_map<std::string, multi_eval_func<cdouble>> &get_funs_cdx1();
} // namespace gsl

namespace boost {
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx1();
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx1();
} // namespace boost

namespace stl {
std::unordered_map<::std::string, multi_eval_func<float>> &get_funs_fx1();
std::unordered_map<::std::string, multi_eval_func<double>> &get_funs_dx1();
} // namespace stl

} // namespace sf::functions

#endif
