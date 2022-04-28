# Benchmarks of various special function libraries

Some preliminary results on my workstation. Certainly not
definitive. All inputs are random on the interval [0,1] for example.

```
function           Mevals/s
---------------------------
sleef_exp10:        117.049
std_exp10:          43.6351

sleef_exp2:         148.75
std_exp2:           80.4371

std_log2:           91.8398

sleef_tan:          66.1653
std_tan:            45.1175

sleef_pow3.5:       20.7117
std_pow3.5:         20.2079

gsl_bessel_K0:      32.9456
boost_bessel_K0:    19.5298

gsl_bessel_J2:      8.90478
boost_bessel_J2:    6.43712

gsl_exp:            91.1578
sleef_exp:          123.877
std_exp:            117.519

gsl_bessel_y1:      15.632
boost_bessel_y1:    1.18714

gsl_bessel_J1:      23.0943
boost_bessel_J1:    51.6073

gsl_bessel_K1:      32.5786
boost_bessel_K1:    17.8249

gsl_pow13:          123.008
boost_pow13:        528.369
sleef_pow13:        20.4049
std_pow13:          20.2538

gsl_sin:            25.2001
sleef_sin:          94.4576
std_sin:            95.5203

gsl_bessel_K2:      7.37997
boost_bessel_K2:    8.18884

gsl_riemann_zeta:   20.1686
boost_riemann_zeta: 41.652

gsl_erf:            15.8742
boost_erf:          20.198
sleef_erf:          46.7152
std_erf:            142.557

gsl_erfc:           14.5249
boost_erfc:         19.9042
sleef_erfc:         29.2883
std_erfc:           102.42

gsl_bessel_y2:      20.5555
boost_bessel_y2:    1.20367

gsl_log:            42.3061
sleef_log:          75.1404
std_log:            47.7767

sleef_log10:        73.5039
std_log10:          46.1167

gsl_hermite_1:      228.896
boost_hermite_1:    365.688

gsl_cos_pi:         35.9388
boost_cos_pi:       18.673
sleef_cos_pi:       48.7208

gsl_hermite_0:      228.985
boost_hermite_0:    601.573

gsl_hermite_3:      73.8649
boost_hermite_3:    303.748

gsl_bessel_j2:      145.642
boost_bessel_j2:    4.22192

gsl_lgamma:         15.7713
boost_lgamma:       17.4234
sleef_lgamma:       9.05995
std_lgamma:         33.2423

gsl_cos:            25.6028
sleef_cos:          73.8453
std_cos:            110.27

gsl_bessel_I1:      23.4346
boost_bessel_I1:    45.2869

gsl_bessel_I0:      21.7577
boost_bessel_I0:    49.8896

gsl_bessel_Y0:      9.55031
boost_bessel_Y0:    14.8271

gsl_sin_pi:         36.994
boost_sin_pi:       9.54709
sleef_sin_pi:       46.5576

gsl_bessel_Y1:      9.32079
boost_bessel_Y1:    13.0276

gsl_bessel_Y2:      8.62195
boost_bessel_Y2:    6.53967

gsl_bessel_j0:      73.3119
boost_bessel_j0:    12.887

gsl_tgamma:         10.0375
boost_tgamma:       2.15392
sleef_tgamma:       10.9437
std_tgamma:         21.0858

gsl_bessel_j1:      46.9584
boost_bessel_j1:    4.17285

gsl_hermite_2:      81.0531
boost_hermite_2:    359.855

gsl_bessel_I2:      9.79727
boost_bessel_I2:    4.98674

gsl_bessel_y0:      23.1249
boost_bessel_y0:    1.22006

gsl_bessel_J0:      22.5602
boost_bessel_J0:    59.7366
```
