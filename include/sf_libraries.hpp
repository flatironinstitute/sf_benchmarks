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

#endif
