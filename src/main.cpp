#include <iostream>
#include <math.h>
#include <sleef.h>
#include <cmath>

int main(int argc, char *argv[]) {
    std::cout << "hi.\n";
    std::cout << cosh(1.0) << std::endl;
    std::cout << Sleef_cosh_u10(1.0) << std::endl;
}
