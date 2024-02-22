#include <sf_libraries.hpp>

namespace sf::functions::SCTL {

std::unordered_map<std::string, multi_eval_func<float>> funs_fx8 = {
    {"memcpy", sctl_apply<float, 8>([](const sctl_fx8 &x) { return x; })},
    {"memset", sctl_apply<float, 8>([](const sctl_fx8 &x) -> sctl_fx8 { return (sctl_fx8::VData)Vec8f{0.0}; })},
    {"exp", sctl_apply<float, 8>([](const sctl_fx8 &x) { return exp(x); })},
    {"log", sctl_apply<float, 8>([](const sctl_fx8 &x) { return log(x); })},
    {"sin", sctl_apply<float, 8>([](const sctl_fx8 &x) {
         sctl_fx8 sinx, cosx;
         sincos(sinx, cosx, x);
         return sinx;
     })},
    {"cos", sctl_apply<float, 8>([](const sctl_fx8 &x) {
         sctl_fx8 sinx, cosx;
         sincos(sinx, cosx, x);
         return cosx;
     })},
    {"rsqrt", sctl_apply<float, 8>([](const sctl_fx8 &x) { return sctl::approx_rsqrt<7>(x); })},
};

std::unordered_map<std::string, multi_eval_func<double>> funs_dx4 = {
    {"memcpy", sctl_apply<double, 4>([](const sctl_dx4 &x) { return x; })},
    {"memset", sctl_apply<double, 4>([](const sctl_dx4 &x) -> sctl_dx4 { return (sctl_dx4::VData)Vec4d{0.0}; })},
    {"exp", sctl_apply<double, 4>([](const sctl_dx4 &x) { return exp(x); })},
    {"log", sctl_apply<double, 4>([](const sctl_dx4 &x) { return log(x); })},
    {"sin", sctl_apply<double, 4>([](const sctl_dx4 &x) {
         sctl_dx4 sinx, cosx;
         sincos(sinx, cosx, x);
         return sinx;
     })},
    {"cos", sctl_apply<double, 4>([](const sctl_dx4 &x) {
         sctl_dx4 sinx, cosx;
         sincos(sinx, cosx, x);
         return cosx;
     })},
    {"rsqrt", sctl_apply<double, 4>([](const sctl_dx4 &x) { return sctl::approx_rsqrt<16>(x); })},
};

#ifdef __AVX512F__
std::unordered_map<std::string, multi_eval_func<float>> funs_fx16 = {
    {"memcpy", sctl_apply<float, 16>([](const sctl_fx16 &x) { return x; })},
    {"memset", sctl_apply<float, 16>([](const sctl_fx16 &x) -> sctl_fx16 { return (sctl_fx16::VData)Vec16f{0.0}; })},
    {"exp", sctl_apply<float, 16>([](const sctl_fx16 &x) { return exp(x); })},
    {"log", sctl_apply<float, 16>([](const sctl_fx16 &x) { return log(x); })},
    {"sin", sctl_apply<float, 16>([](const sctl_fx16 &x) {
         sctl_fx16 sinx, cosx;
         sincos(sinx, cosx, x);
         return sinx;
     })},
    {"cos", sctl_apply<float, 16>([](const sctl_fx16 &x) {
         sctl_fx16 sinx, cosx;
         sincos(sinx, cosx, x);
         return cosx;
     })},
    {"rsqrt", sctl_apply<float, 16>([](const sctl_fx16 &x) { return sctl::approx_rsqrt<7>(x); })},
};

std::unordered_map<std::string, multi_eval_func<double>> funs_dx8 = {
    {"memcpy", sctl_apply<double, 8>([](const sctl_dx8 &x) { return x; })},
    {"memset", sctl_apply<double, 8>([](const sctl_dx8 &x) -> sctl_dx8 { return (sctl_dx8::VData)Vec8d{0.0}; })},
    {"exp", sctl_apply<double, 8>([](const sctl_dx8 &x) { return exp(x); })},
    {"log", sctl_apply<double, 8>([](const sctl_dx8 &x) { return log(x); })},
    {"sin", sctl_apply<double, 8>([](const sctl_dx8 &x) {
         sctl_dx8 sinx, cosx;
         sincos(sinx, cosx, x);
         return sinx;
     })},
    {"cos", sctl_apply<double, 8>([](const sctl_dx8 &x) {
         sctl_dx8 sinx, cosx;
         sincos(sinx, cosx, x);
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
