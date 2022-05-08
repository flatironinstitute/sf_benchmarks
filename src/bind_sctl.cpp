#include <dlfcn.h>
#include <string>
#include <unordered_map>

#include <sf_libraries.hpp>

namespace sf::functions::SCTL {

std::unordered_map<std::string, multi_eval_func<float>> funs_fx8 = {
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

std::unordered_map<std::string, multi_eval_func<double>> funs_dx4 = {
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

#ifdef __AVX512F__
std::unordered_map<std::string, multi_eval_func<float>> funs_fx16 = {
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

std::unordered_map<std::string, multi_eval_func<double>> funs_dx8 = {
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
#else
std::unordered_map<std::string, multi_eval_func<float>> funs_fx16;
std::unordered_map<std::string, multi_eval_func<double>> funs_dx8;
#endif

std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx8() { return funs_fx8; }
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx4() { return funs_dx4; }
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx16() { return funs_fx16; }
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx8() { return funs_dx8; }

} // namespace sf::functions::SCTL
