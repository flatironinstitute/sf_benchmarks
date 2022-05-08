#include <dlfcn.h>
#include <string>
#include <unordered_map>

#include <sf_libraries.hpp>

namespace sf::functions::misc {
std::unordered_map<std::string, fun_cdx1_x2> funs_cdx1_x2 = {
    {"hank103", [](cdouble z) -> std::pair<cdouble, cdouble> {
         cdouble h0, h1;
         int ifexpon = 1;
         hank103_((double _Complex *)&z, (double _Complex *)&h0, (double _Complex *)&h1, &ifexpon);
         return {h0, h1};
     }}};

std::unordered_map<std::string, fun_cdx1_x2> &get_funs_cdx1_x2() { return funs_cdx1_x2; }
} // namespace sf::functions::misc
