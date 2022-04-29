# Benchmarks of various special function libraries

Some preliminary results on my workstation. Certainly not
definitive. All inputs are random on the interval [0,1] for example.

```
Model name:          Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz
CPU MHz:             3400.000
CPU max MHz:         3700.0000
CPU min MHz:         1200.0000

gcc 11.2.0 (-march=native -O3)
rocky8
glibc 2.28
```


```
function           Mevals/s
---------------------------
sleef_acos:         71.6999
sleef_dx4_acos:     241.049
sleef_dx8_acos:     320.492
std_acos:           57.3163
eigen_acos:         62.8068

sleef_acosh:        25.0122
sleef_dx4_acosh:    86.5068
sleef_dx8_acosh:    122.141
std_acosh:          191.635
eigen_acosh:        250.278

sleef_asin:         71.4394
sleef_dx4_asin:     281.538
sleef_dx8_asin:     352.384
std_asin:           57.4311
eigen_asin:         64.5833

sleef_asinh:        21.432
sleef_dx4_asinh:    68.4559
sleef_dx8_asinh:    107.937
std_asinh:          35.7119
eigen_asinh:        37.6585

sleef_atan:         68.8296
sleef_dx4_atan:     200.972
sleef_dx8_atan:     317.848
std_atan:           46.1974
eigen_atan:         48.9877

sleef_atanh:        35.0551
sleef_dx4_atanh:    131.014
sleef_dx8_atanh:    203.459
std_atanh:          40.5397
eigen_atanh:        43.0846

gsl_bessel_I0:      21.7342
boost_bessel_I0:    46.8539

gsl_bessel_I1:      23.4131
boost_bessel_I1:    43.3671

gsl_bessel_I2:      9.77897
boost_bessel_I2:    5.39706

gsl_bessel_J0:      22.0006
boost_bessel_J0:    30.9796

gsl_bessel_J1:      23.6527
boost_bessel_J1:    28.7143

gsl_bessel_J2:      8.88124
boost_bessel_J2:    11.5437

gsl_bessel_K0:      33.0013
boost_bessel_K0:    17.0373

gsl_bessel_K1:      32.6505
boost_bessel_K1:    14.7856

gsl_bessel_K2:      7.30499
boost_bessel_K2:    7.99836

gsl_bessel_Y0:      9.52245
boost_bessel_Y0:    11.4978

gsl_bessel_Y1:      9.20491
boost_bessel_Y1:    10.6936

gsl_bessel_Y2:      8.53053
boost_bessel_Y2:    5.41445

gsl_bessel_j0:      71.5352
boost_bessel_j0:    13.0046

gsl_bessel_j1:      46.3495
boost_bessel_j1:    4.0281

gsl_bessel_j2:      144.448
boost_bessel_j2:    4.02913

gsl_bessel_y0:      23.0755
boost_bessel_y0:    1.20235

gsl_bessel_y1:      15.6073
boost_bessel_y1:    1.19668

gsl_bessel_y2:      20.487
boost_bessel_y2:    1.19282

gsl_cos:            26.3003
sleef_cos:          72.2803
sleef_dx4_cos:      293.337
sleef_dx8_cos:      469.349
std_cos:            108.517
eigen_cos:          122.823

gsl_cos_pi:         36.0626
boost_cos_pi:       18.8987
sleef_cos_pi:       48.5551
std_cos_pi:         57.6042

sleef_cosh:         41.9157
sleef_dx4_cosh:     173.75
sleef_dx8_cosh:     264.679
std_cosh:           64.7764
eigen_cosh:         72.6633

boost_digamma:      50.4184
eigen_digamma:      40.8368

gsl_erf:            15.8702
boost_erf:          19.7833
sleef_erf:          43.984
sleef_dx4_erf:      65.8141
sleef_dx8_erf:      126.975
std_erf:            143.094
eigen_erf:          165.795

gsl_erfc:           14.5503
boost_erfc:         20.0656
sleef_erfc:         29.284
sleef_dx4_erfc:     53.8045
sleef_dx8_erfc:     101.147
std_erfc:           103.064
eigen_erfc:         114.846

gsl_exp:            89.6022
sleef_exp:          123.37
sleef_dx4_exp:      419.93
sleef_dx8_exp:      700.815
std_exp:            114.539
sctl_dx4_exp:       504.484
eigen_exp:          554.791

sleef_exp10:        117.271
sleef_dx4_exp10:    382.509
sleef_dx8_exp10:    676.026
std_exp10:          43.451

sleef_exp2:         149.201
sleef_dx4_exp2:     490.376
sleef_dx8_exp2:     766.285
std_exp2:           78.7596

gsl_hermite_0:      203.921
boost_hermite_0:    493.139

gsl_hermite_1:      216.866
boost_hermite_1:    335.733

gsl_hermite_2:      79.4564
boost_hermite_2:    334.357

gsl_hermite_3:      70.4948
boost_hermite_3:    289.07

gsl_lgamma:         15.7623
boost_lgamma:       16.9472
sleef_lgamma:       11.1511
sleef_dx4_lgamma:   31.5217
sleef_dx8_lgamma:   56.3401
std_lgamma:         34.1914
eigen_lgamma:       35.4162

gsl_log:            42.2871
sleef_log:          74.9692
sleef_dx4_log:      247.194
sleef_dx8_log:      500.408
std_log:            47.2692
eigen_log:          501.388

sleef_log10:        73.2453
sleef_dx4_log10:    239.379
sleef_dx8_log10:    495.085
std_log10:          45.7676
eigen_log10:        48.0926

std_log2:           92.8415

eigen_ndtri:        16.1078

gsl_pow13:          143.758
boost_pow13:        449.842
sleef_pow13:        20.4139
sleef_dx4_pow13:    75.3856
sleef_dx8_pow13:    148.3
std_pow13:          20.1888
eigen_pow13:        36.9479

sleef_pow3.5:       20.265
sleef_dx4_pow3.5:   75.3571
sleef_dx8_pow3.5:   148.695
std_pow3.5:         20.204
eigen_pow3.5:       40.8764

gsl_riemann_zeta:   20.3502
boost_riemann_zeta: 43.4612

std_rsqrt:          331.774
eigen_rsqrt:        879.624

gsl_sin:            25.8494
sleef_sin:          94.1602
sleef_dx4_sin:      304.679
sleef_dx8_sin:      313.43
std_sin:            95.8646
eigen_sin:          103.567

gsl_sin_pi:         36.7527
boost_sin_pi:       9.26593
sleef_sin_pi:       46.8486
std_sin_pi:         60.2729

gsl_sinc:           19.7882

gsl_sinc_pi:        27.9802
boost_sinc_pi:      83.1254

sleef_sinh:         41.4466
sleef_dx4_sinh:     163.931
sleef_dx8_sinh:     264.686
std_sinh:           53.5038
eigen_sinh:         58.9807

sleef_sqrt:         122.865
sleef_dx4_sqrt:     368.929
sleef_dx8_sqrt:     595.62
std_sqrt:           473.587
eigen_sqrt:         877.73

sleef_tan:          65.4528
sleef_dx4_tan:      230.961
sleef_dx8_tan:      357.14
std_tan:            45.25
eigen_tan:          46.611

sleef_tanh:         34.2783
sleef_dx4_tanh:     138.901
sleef_dx8_tanh:     212.637
std_tanh:           52.3251
eigen_tanh:         55.9111

gsl_tgamma:         10.0273
boost_tgamma:       2.14009
sleef_tgamma:       10.8103
sleef_dx4_tgamma:   33.5636
sleef_dx8_tgamma:   63.076
std_tgamma:         21.0316
```
