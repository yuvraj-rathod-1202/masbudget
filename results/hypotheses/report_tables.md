## t_main

| Mix | Budget ×c_min | MCKP Q | Naive (Random+Best) Q | Naive gap % | Random+QL gap % | Best decentralized | Best dec. Q | Best dec. gap % [95% CI] | Best dec. gap on q-hat objective % |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| uniform | 1.000 | 0.681 | 0.168 | 75.397 | 0.000 | Easiest-first + Shadow price | 0.681 | 0.0 [0.0, 0.0] | -0.000 |
| uniform | 1.500 | 0.680 | 0.092 | 86.449 | 0.739 | Easiest-first + Shapley share | 0.681 | -0.3 [-1.3, 0.7] | 0.095 |
| uniform | 3.000 | 0.697 | 0.155 | 77.825 | 3.192 | Hardest-first + Shadow price | 0.697 | 0.0 [-0.8, 0.8] | 0.473 |
| uniform | 6.000 | 0.730 | 0.206 | 71.696 | 1.586 | Marginal-ROI priority + Shadow price | 0.732 | -0.3 [-1.3, 0.5] | 0.511 |
| uniform | 12.000 | 0.781 | 0.361 | 53.762 | 0.751 | Random + Shadow price | 0.780 | 0.1 [-0.3, 0.6] | 0.695 |
| uniform | 25.000 | 0.802 | 0.632 | 21.224 | -0.291 | Easiest-first + Shapley share | 0.804 | -0.3 [-0.8, 0.2] | 0.127 |
| uniform | 50.000 | 0.797 | 0.797 | 0.000 | -0.859 | Easiest-first + Shapley share | 0.804 | -0.9 [-1.5, -0.2] | 0.000 |
| uniform | 120.000 | 0.797 | 0.797 | 0.000 | -0.859 | Easiest-first + Shadow price | 0.804 | -0.9 [-1.5, -0.2] | 0.000 |
| easy_heavy | 1.000 | 0.902 | 0.259 | 71.287 | 0.000 | Easiest-first + Shadow price | 0.902 | 0.0 [0.0, 0.0] | 0.000 |
| easy_heavy | 1.500 | 0.906 | 0.289 | 68.111 | 0.656 | Hardest-first + Shadow price | 0.904 | 0.2 [-0.2, 0.6] | 0.282 |
| easy_heavy | 3.000 | 0.923 | 0.431 | 53.391 | 2.480 | Marginal-ROI priority + Shadow price | 0.922 | 0.1 [-0.2, 0.4] | 0.345 |
| easy_heavy | 6.000 | 0.939 | 0.651 | 30.618 | 2.375 | Easiest-first + Shadow price | 0.932 | 0.8 [0.4, 1.1] | 0.757 |
| easy_heavy | 12.000 | 0.935 | 0.921 | 1.497 | 0.551 | Easiest-first + Shadow price | 0.936 | -0.1 [-0.5, 0.3] | 0.002 |
| easy_heavy | 25.000 | 0.936 | 0.936 | 0.000 | -0.323 | Easiest-first + Shapley share | 0.939 | -0.3 [-0.6, -0.0] | 0.000 |
| easy_heavy | 50.000 | 0.936 | 0.936 | 0.000 | -0.323 | Easiest-first + Shapley share | 0.939 | -0.3 [-0.6, -0.0] | 0.000 |
| easy_heavy | 120.000 | 0.936 | 0.936 | 0.000 | -0.323 | Easiest-first + Shadow price | 0.939 | -0.3 [-0.6, -0.0] | 0.000 |
| hard_heavy | 1.000 | 0.293 | 0.067 | 76.968 | 0.000 | Easiest-first + Shadow price | 0.293 | 0.0 [0.0, 0.0] | 0.000 |
| hard_heavy | 1.500 | 0.306 | 0.029 | 90.433 | 4.329 | Marginal-ROI priority + ΔU/ΔC ratio | 0.305 | 0.2 [-0.5, 0.8] | 0.384 |
| hard_heavy | 3.000 | 0.316 | 0.043 | 86.166 | 7.503 | Hardest-first + ΔU/ΔC ratio | 0.312 | 1.1 [-0.5, 2.7] | 1.031 |
| hard_heavy | 6.000 | 0.337 | 0.082 | 75.595 | 31.443 | Marginal-ROI priority + Shadow price | 0.343 | -2.2 [-4.5, 0.2] | 2.864 |
| hard_heavy | 12.000 | 0.409 | 0.122 | 69.996 | 21.024 | Marginal-ROI priority + Shadow price | 0.428 | -5.2 [-7.8, -2.5] | 1.312 |
| hard_heavy | 25.000 | 0.529 | 0.224 | 57.432 | -0.449 | Easiest-first + Quasi-linear | 0.531 | -0.4 [-0.7, -0.2] | 0.054 |
| hard_heavy | 50.000 | 0.512 | 0.413 | 19.318 | -3.808 | Easiest-first + Quasi-linear | 0.531 | -3.8 [-5.1, -2.5] | 0.804 |
| hard_heavy | 120.000 | 0.513 | 0.513 | 0.000 | -3.745 | Easiest-first + Shadow price | 0.531 | -3.7 [-5.4, -2.1] | 0.000 |

## t_arrival

| Mix | Budget | Round-robin FCFS | Random | Hardest-first | Easiest-first | Marginal-ROI priority | HTF − Random (pp), p | ETF − Random (pp), p |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| uniform | tight (×1.5–6) | 27.3 | 27.3 | 35.2 | 11.2 | 35.3 | +7.9, p=0.0001 | -16.1, p=0.0001 |
| uniform | loose (×12–50) | 7.2 | 7.1 | 13.2 | 3.3 | 13.3 | +6.1, p=0.0001 | -3.8, p=0.0001 |
| easy_heavy | tight (×1.5–6) | 15.6 | 14.6 | 28.7 | 1.4 | 28.9 | +14.1, p=0.0001 | -13.3, p=0.0001 |
| easy_heavy | loose (×12–50) | 0.6 | 0.6 | 2.1 | 0.5 | 2.1 | +1.4, p=0.0001 | -0.1, p=0.0001 |
| hard_heavy | tight (×1.5–6) | 25.9 | 26.1 | 25.3 | 14.2 | 24.8 | -0.8, p=0.202 | -12.0, p=0.0001 |
| hard_heavy | loose (×12–50) | 19.7 | 19.2 | 26.6 | 13.1 | 26.4 | +7.4, p=0.0001 | -6.1, p=0.0001 |

## t_utility

| Mix | Budget | Best-model | Quasi-linear | Shadow price | Shapley share | ΔU/ΔC ratio |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| uniform | tight (×1.5–6) | 78.7 | 1.8 (-76.8*) | 0.6 (-78.0*) | 2.7 (-76.0*) | 52.9 (-25.8*) |
| uniform | loose (×12–50) | 25.0 | -0.1 (-25.1*) | -0.2 (-25.2*) | -0.1 (-25.1*) | 10.9 (-14.1*) |
| easy_heavy | tight (×1.5–6) | 50.7 | 1.8 (-48.9*) | 0.7 (-50.0*) | 1.3 (-49.4*) | 18.6 (-32.1*) |
| easy_heavy | loose (×12–50) | 0.5 | -0.0 (-0.5*) | 0.1 (-0.4) | -0.0 (-0.5*) | 2.6 (+2.1*) |
| hard_heavy | tight (×1.5–6) | 84.1 | 14.4 (-69.6*) | 1.4 (-82.7*) | 8.1 (-76.0*) | 22.7 (-61.4*) |
| hard_heavy | loose (×12–50) | 48.9 | 5.6 (-43.3*) | 2.1 (-46.8*) | 5.8 (-43.1*) | 33.7 (-15.2*) |

## t_models

| Model | Cost $/call | pass@1 easy | pass@1 medium | pass@1 hard | Δpass hard vs cheapest | Δpass medium vs cheapest | Δpass(hard)/Δ$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| deepseek-v3.1-terminus | 0.0011 | 1.0000 | 0.8835 | 0.1364 | 0.0000 | 0.0000 | nan |
| deepseek-v3-0324 | 0.0014 | 1.0000 | 0.8523 | 0.1477 | 0.0114 | -0.0312 | 35.2729 |
| gemini-2.5-flash | 0.0058 | 1.0000 | 0.7358 | 0.0653 | -0.0710 | -0.1477 | -15.0812 |
| claude-sonnet-4 | 0.0135 | 1.0000 | 0.6847 | 0.0597 | -0.0767 | -0.1989 | -6.2175 |
| deepseek-r1-0528 | 0.0289 | 1.0000 | 0.9773 | 0.4261 | 0.2898 | 0.0938 | 10.4432 |
| gemini-2.5-pro | 0.1352 | 1.0000 | 0.9403 | 0.4347 | 0.2983 | 0.0568 | 2.2256 |

## t_fmr

| Utility | Budget ×c_min | Q first decile | Q last decile | Starved last decile % | Agent FMR (diff, pp) | Agent starvation % | Jain |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Best-model | 3 | 0.758 | 0.000 | 100.000 | 16.300 | 100.000 | 0.638 |
| Best-model | 6 | 0.796 | 0.000 | 100.000 | 18.900 | 100.000 | 0.760 |
| Best-model | 25 | 0.796 | 0.066 | 90.200 | 18.100 | 76.400 | 0.945 |
| Quasi-linear | 3 | 0.626 | 0.682 | 0.000 | -2.000 | 0.000 | 0.957 |
| Quasi-linear | 6 | 0.692 | 0.684 | 6.000 | 0.400 | 5.400 | 0.966 |
| Quasi-linear | 25 | 0.796 | 0.810 | 0.000 | -2.100 | 0.000 | 0.978 |
| Shadow price | 3 | 0.758 | 0.682 | 0.000 | -0.200 | 0.000 | 0.961 |
| Shadow price | 6 | 0.762 | 0.684 | 0.000 | 0.700 | 0.000 | 0.968 |
| Shadow price | 25 | 0.804 | 0.810 | 0.000 | -1.500 | 0.000 | 0.979 |
| Shapley share | 3 | 0.636 | 0.688 | 0.600 | -2.000 | 0.600 | 0.954 |
| Shapley share | 6 | 0.636 | 0.694 | 0.000 | -2.100 | 0.000 | 0.953 |
| Shapley share | 25 | 0.796 | 0.810 | 0.000 | -2.100 | 0.000 | 0.978 |
| ΔU/ΔC ratio | 3 | 0.680 | 0.000 | 100.000 | 24.600 | 100.000 | 0.828 |
| ΔU/ΔC ratio | 6 | 0.680 | 0.000 | 100.000 | 20.000 | 94.800 | 0.921 |
| ΔU/ΔC ratio | 25 | 0.680 | 0.724 | 0.000 | -2.900 | 0.000 | 0.962 |

## fmr_note



## t_info

| Mix | Budget ×c_min | MCKP Q (expected) | Best dec. Q (expected) | Best dec. gap % (expected) | MCKP Q (oracle) | Best dec. Q (oracle) | Best dec. gap % (oracle) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| uniform | 3 | 0.697 | 0.697 | 0.029 | 0.832 | 0.828 | 0.457 |
| uniform | 12 | 0.781 | 0.780 | 0.154 | 0.860 | 0.860 | -0.000 |
| uniform | 50 | 0.797 | 0.804 | -0.828 | 0.860 | 0.860 | -0.000 |
| easy_heavy | 3 | 0.923 | 0.922 | 0.108 | 0.954 | 0.953 | 0.021 |
| easy_heavy | 12 | 0.935 | 0.936 | -0.107 | 0.956 | 0.956 | 0.000 |
| easy_heavy | 50 | 0.936 | 0.939 | -0.321 | 0.956 | 0.956 | 0.000 |
| hard_heavy | 3 | 0.316 | 0.312 | 1.265 | 0.534 | 0.460 | 13.927 |
| hard_heavy | 12 | 0.409 | 0.428 | -4.699 | 0.648 | 0.646 | 0.309 |
| hard_heavy | 50 | 0.512 | 0.531 | -3.669 | 0.661 | 0.661 | -0.000 |

## t_scale

| Mix | Budget ×c_min | N=5: mean dec. gap % | N=5: agent starve % | N=10: mean dec. gap % | N=10: agent starve % | N=20: mean dec. gap % | N=20: agent starve % | N=50: mean dec. gap % | N=50: agent starve % | Δgap N50−N5 (pp), p |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| uniform | 3 | 23.0 | 100.0 | 28.1 | 100.0 | 29.0 | 100.0 | 29.9 | 100.0 | +6.9, p=0.0001 |
| uniform | 12 | 13.0 | 98.8 | 14.4 | 99.8 | 15.1 | 100.0 | 15.2 | 100.0 | +2.2, p=0.0001 |
| easy_heavy | 3 | 17.0 | 86.0 | 18.3 | 99.4 | 19.4 | 99.9 | 20.0 | 100.0 | +3.0, p=0.0001 |
| easy_heavy | 12 | 2.6 | 1.2 | 2.8 | 12.2 | 2.1 | 0.0 | 1.4 | 1.6 | -1.3, p=0.0021 |
| hard_heavy | 3 | 20.1 | 100.0 | 20.9 | 100.0 | 22.5 | 100.0 | 24.2 | 100.0 | +4.1, p=0.0012 |
| hard_heavy | 12 | 27.0 | 100.0 | 29.0 | 100.0 | 29.2 | 100.0 | 30.9 | 100.0 | +3.9, p=0.0001 |

## t_disparity

| Mix | Pool | MCKP Q (tight ×1.5–6) | MCKP Q (loose ×12–50) | Naive gap % tight | Mean dec. gap % tight | Mean dec. gap % loose | Best dec. gap % tight |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| uniform | flat | 0.677 | 0.677 | 0.000 | 0.000 | 0.000 | 0.000 |
| uniform | steep | 0.706 | 0.793 | 77.693 | 29.337 | 7.928 | -0.101 |
| easy_heavy | flat | 0.902 | 0.902 | 0.000 | 0.000 | 0.000 | 0.000 |
| easy_heavy | steep | 0.923 | 0.936 | 50.321 | 19.266 | 0.595 | 0.158 |
| hard_heavy | flat | 0.304 | 0.304 | 0.000 | 0.000 | 0.000 | 0.000 |
| hard_heavy | steep | 0.316 | 0.483 | 82.358 | 29.293 | 15.291 | -0.674 |

## t_noise

| Label noise ε | Budget ×c_min | MCKP Q | Random+Best Q | ETF+Shadow Q | HTF+Shadow Q | Best dec. | Best dec. Q |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.000 | 3 | 0.697 | 0.155 | 0.686 | 0.697 | Hardest-first + Shadow price | 0.697 |
| 0.000 | 12 | 0.781 | 0.361 | 0.769 | 0.775 | Random + Shadow price | 0.780 |
| 0.000 | 50 | 0.797 | 0.797 | 0.802 | 0.799 | Easiest-first + Quasi-linear | 0.804 |
| 0.200 | 3 | 0.693 | 0.172 | 0.684 | 0.691 | Random + Shadow price | 0.693 |
| 0.200 | 12 | 0.766 | 0.355 | 0.759 | 0.759 | Random + Shadow price | 0.762 |
| 0.200 | 50 | 0.785 | 0.785 | 0.790 | 0.788 | Easiest-first + Quasi-linear | 0.792 |
| 0.400 | 3 | 0.688 | 0.176 | 0.682 | 0.689 | Marginal-ROI priority + Shadow price | 0.691 |
| 0.400 | 12 | 0.753 | 0.341 | 0.747 | 0.744 | Round-robin FCFS + Shadow price | 0.748 |
| 0.400 | 50 | 0.775 | 0.775 | 0.779 | 0.778 | Easiest-first + Quasi-linear | 0.781 |

## tuned

{"best_model": {}, "quasi_linear": {"lambda_cost": 1000}, "dynamic_shadow_price": {"alpha": 0.5, "p0": 100}, "shapley_budget_share": {"gamma": 1.0, "lambda_cost": 1000, "num_samples": 100, "seed": 42}, "marginal_roi_ratio": {}}

## t_calibration

| Utility | Mean Q, calibrated per budget | Mean Q, best single setting | Mean regret, param from adjacent budget (x2) | Mean regret, param from any other budget | Worst-case regret |
|:---|:---:|:---:|:---:|:---:|:---:|
| Quasi-linear | 0.741 | 0.675 | 0.051 | 0.186 | 0.510 |
| Shadow price | 0.743 | 0.675 | 0.055 | 0.174 | 0.555 |
| Shapley share | 0.737 | 0.675 | 0.041 | 0.179 | 0.510 |

## t_burst

| Utility | Budget x c_min | First-quartile Q minus last-quartile Q (pp) | Runs where last quartile gets 0 | Agent starve % | Jain | Gap % |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Best-model | 3 | 65.500 | 100% | 77.200 | 0.249 | 74.123 |
| Best-model | 6 | 79.300 | 100% | 74.000 | 0.283 | 70.267 |
| Best-model | 25 | 66.500 | 38% | 24.000 | 0.795 | 21.612 |
| Quasi-linear | 3 | 1.800 | 0% | 0.000 | 0.957 | 3.192 |
| Quasi-linear | 6 | 4.700 | 4% | 1.000 | 0.958 | 1.322 |
| Quasi-linear | 25 | 0.900 | 0% | 0.000 | 0.978 | -0.291 |
| Shadow price | 3 | 10.700 | 0% | 0.000 | 0.959 | 0.396 |
| Shadow price | 6 | 12.300 | 0% | 0.000 | 0.959 | 1.252 |
| Shadow price | 25 | -0.200 | 0% | 0.000 | 0.978 | 0.193 |
| Shapley share | 3 | 1.800 | 0% | 0.200 | 0.953 | 2.042 |
| Shapley share | 6 | 1.400 | 0% | 0.000 | 0.953 | 6.311 |
| Shapley share | 25 | 0.900 | 0% | 0.000 | 0.978 | -0.291 |
| ΔU/ΔC ratio | 3 | 72.800 | 100% | 65.000 | 0.431 | 55.969 |
| ΔU/ΔC ratio | 6 | 70.300 | 96% | 30.400 | 0.732 | 28.148 |
| ΔU/ΔC ratio | 25 | 2.600 | 0% | 0.000 | 0.962 | 11.861 |

## t_league

| Paradigm | Policy | Mean Q | Gap % | Task starve % | Agent starve % | Jain | Budget used % |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Centralized | Cascade† | 0.728 | -7.334 | 0.000 | 0.000 | 0.952 | 83.448 |
| Centralized | Predictive router | 0.691 | -0.110 | 0.000 | 0.000 | 0.944 | 82.542 |
| Centralized | MCKP (optimal) | 0.691 | 0.000 | 0.000 | 0.000 | 0.943 | 83.680 |
| Decentralized | Marginal-ROI priority + Shadow price | 0.691 | -0.109 | 0.000 | 0.000 | 0.944 | 69.941 |
| Decentralized | Hardest-first + Shadow price | 0.690 | 0.271 | 0.000 | 0.000 | 0.944 | 69.901 |
| Decentralized | Random + Shadow price | 0.689 | 0.385 | 0.000 | 0.000 | 0.944 | 68.751 |
| Decentralized | Round-robin FCFS + Shadow price | 0.688 | 0.520 | 0.000 | 0.000 | 0.943 | 68.883 |
| Centralized | Equal quota | 0.688 | 0.531 | 0.000 | 0.000 | 0.941 | 67.884 |
| Decentralized | Easiest-first + Quasi-linear | 0.687 | 0.925 | 1.855 | 8.083 | 0.943 | 57.485 |
| Decentralized | Easiest-first + Shadow price | 0.687 | 0.881 | 0.000 | 0.000 | 0.943 | 65.649 |
| Decentralized | Easiest-first + Shapley share | 0.685 | 1.265 | 0.669 | 3.975 | 0.941 | 54.148 |
| Decentralized | Random + Shapley share | 0.682 | 2.029 | 1.521 | 4.158 | 0.939 | 54.148 |
| Decentralized | Round-robin FCFS + Shapley share | 0.682 | 2.141 | 1.538 | 4.250 | 0.940 | 54.149 |
| Decentralized | Random + Quasi-linear | 0.680 | 2.735 | 3.360 | 8.492 | 0.938 | 57.483 |
| Decentralized | Round-robin FCFS + Quasi-linear | 0.679 | 2.962 | 3.378 | 8.833 | 0.941 | 57.485 |
| Decentralized | Marginal-ROI priority + Shapley share | 0.678 | 3.015 | 1.668 | 4.217 | 0.937 | 54.150 |
| Decentralized | Hardest-first + Shapley share | 0.677 | 3.261 | 1.668 | 4.250 | 0.936 | 54.150 |
| Decentralized | Marginal-ROI priority + Quasi-linear | 0.667 | 5.444 | 3.675 | 9.975 | 0.930 | 57.491 |
| Decentralized | Hardest-first + Quasi-linear | 0.665 | 6.100 | 4.423 | 10.075 | 0.928 | 57.490 |
| Centralized | Cheapest model | 0.626 | 11.090 | 0.000 | 0.000 | 0.919 | 28.979 |
| Decentralized | Easiest-first + Best-model | 0.619 | 14.178 | 25.893 | 50.683 | 0.918 | 84.291 |
| Decentralized | Easiest-first + ΔU/ΔC ratio | 0.598 | 16.735 | 16.625 | 31.150 | 0.913 | 56.897 |
| Decentralized | Random + ΔU/ΔC ratio | 0.515 | 28.114 | 19.940 | 37.033 | 0.872 | 56.899 |
| Decentralized | Round-robin FCFS + ΔU/ΔC ratio | 0.515 | 28.183 | 19.648 | 37.667 | 0.896 | 56.895 |
| Decentralized | Hardest-first + ΔU/ΔC ratio | 0.465 | 32.523 | 17.917 | 33.000 | 0.844 | 56.787 |
| Decentralized | Marginal-ROI priority + ΔU/ΔC ratio | 0.464 | 32.545 | 17.917 | 33.008 | 0.841 | 56.787 |
| Decentralized | Random + Best-model | 0.419 | 45.299 | 45.519 | 69.108 | 0.767 | 84.417 |
| Decentralized | Round-robin FCFS + Best-model | 0.414 | 45.618 | 45.811 | 70.858 | 0.823 | 84.412 |
| Decentralized | Hardest-first + Best-model | 0.293 | 62.032 | 55.647 | 72.150 | 0.608 | 84.462 |
| Decentralized | Marginal-ROI priority + Best-model | 0.287 | 63.057 | 56.485 | 72.292 | 0.588 | 84.455 |