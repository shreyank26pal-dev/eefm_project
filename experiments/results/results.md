# FinTwin experimental results

## E1 Survey replication and overhead-aware revaluation (Standard tier)

| Method | NPV | IRR | Payback |
|---|---|---|---|
| Survey CIE-1 (reported) | $684,814 | 60.3% | 1.50 yr |
| FinTwin static (same assumptions) | $684,881 | 60.2% | 1.50 yr |
| + hidden overheads, expected @ WACC | $657,649 | 51.5% | 2.14 yr (disc.) |
| + Monte Carlo @ r(σ)=13.5% (mean) | $375,387 | 40.0% (median) | 3 yr (median disc.) |

σ = 0.345; P(NPV<0) = 0.28%; 90% NPV band $147.6k – $631.9k; CVaR5% $96.1k; break-even benefit realisation 53.7%.

PW of overheads: energy $8.6k, refresh $63.1k, compliance $68.2k, opex $171.7k, ramp_loss $154.8k

## E2 Twin-measured vs survey-assumed benefits (5 seeds, 365 days)

| Benefit | Survey assumption | Twin steady state | Twin under stress |
|---|---|---|---|
| B1_working_capital | $184.1k | $84.4k ± $1.9k | $92.6k ± $2.0k |
| B2_credit_spread | $81.8k | $93.8k ± $0.9k | $91.1k ± $0.6k |
| B3_failure_avoidance | $78.5k | $4.9k ± $16.0k | $197.5k ± $47.8k |
| total | $344.4k | $183.1k ± $16.0k | $381.2k ± $46.5k |

| Operational metric | Baseline (steady) | Twin (steady) | Baseline (stress) | Twin (stress) |
|---|---|---|---|---|
| avg_ccc | 55.1 | 36.3 | 57.6 | 36.7 |
| avg_dio | 48.2 | 34.1 | 50.6 | 34.5 |
| avg_dso | 42.0 | 37.2 | 42.0 | 37.2 |
| avg_loan_rate | 14.10% | 10.35% | 14.10% | 10.46% |
| failures | 39.4 | 36.4 | 68.2 | 39.0 |
| replacements | 48.0 | 66.4 | 43.4 | 66.0 |
| derates | 0.0 | 0.0 | 0.0 | 39.8 |
| maint_cost | $412.2k | $407.3k | $665.9k | $468.4k |

With measured benefits (shock-year extra $198.1k, p=0.30): static NPV $73.6k, MC mean $-73.3k, P(loss) 71.2%, median IRR 10.6%.

## E4 ML models

### Failure-probability model (AI4I 2020, 10,000 records, 25% stratified hold-out, threshold tuned on out-of-fold train predictions; C_FN=$25k, C_FP=$1.5k)

| Model | ROC-AUC | PR-AUC | F1@0.5 | Cost threshold | Recall@thr | Expected cost | Saving vs run-to-failure |
|---|---|---|---|---|---|---|---|
| LogisticRegression | 0.940 | 0.502 | 0.349 | 0.06 | 0.812 | $727,000 | 65.8% |
| RandomForest | 0.972 | 0.845 | 0.795 | 0.20 | 0.788 | $465,000 | 78.1% |
| HistGradientBoosting | 0.963 | 0.835 | 0.784 | 0.08 | 0.835 | $407,000 | 80.8% |

### Invoice settlement (DSO) model (6,000 synthetic invoices, 25% hold-out)

| Model | MAE (days) | R² |
|---|---|---|
| Naive (terms + past delay) | 5.84 | 0.742 |
| LinearRegression | 2.89 | 0.935 |
| GradientBoosting | 2.90 | 0.935 |

Counterfactual twin-verified-delivery effect: −6.32 days per invoice.

## E5 Maintenance policy: fixed time-based interval vs twin Eq. 7 rule (6 machines, 5 seeds)

| Interval (wear min) | Scenario | Time-based cost | Twin cost | Time-based failures | Twin failures |
|---|---|---|---|---|---|
| 120 | steady | $445.2k | $407.3k | 36.4 | 36.4 |
| 120 | overload | $895.0k | $516.2k | 88.4 | 40.6 |
| 160 | steady | $412.2k | $407.3k | 36.6 | 36.4 |
| 160 | overload | $871.9k | $516.2k | 89.2 | 40.6 |
| 200 | steady | $412.2k | $407.3k | 39.4 | 36.4 |
| 200 | overload | $869.2k | $516.2k | 91.4 | 40.6 |
| 240 | steady | $627.6k | $407.3k | 68.0 | 36.4 |
| 240 | overload | $1,032.0k | $516.2k | 113.2 | 40.6 |

## E6 Computational performance (single CPU core)

- 365-day dual-world twin with 3 shocks: 2.20 s
- Tier comparison (3 × 5,000 runs + sensitivity): 0.07 s

| MC runs | Time (s) | Mean NPV | Std. error |
|---|---|---|---|
| 1,000 | 0.006 | $375.7k | $4.8k |
| 5,000 | 0.024 | $375.4k | $2.1k |
| 20,000 | 0.088 | $371.1k | $1.1k |

## E3 Twin fidelity right-sizing (Gap 2)

### (a) Survey-assumed benefits

| Tier | Capex | Static NPV | Risk-adj. mean NPV | P5 NPV | P(loss) | Median IRR | r(σ) | PW overheads | CO₂ t/yr |
|---|---|---|---|---|---|---|---|---|---|
| lite | $150.0k | $370.7k | $258.2k | $158.8k | 0.0% | 62.0% | 12.7% | $23.3k | 0.8 |
| standard | $450.0k | $684.9k | $375.4k | $147.6k | 0.3% | 40.0% | 13.5% | $140.0k | 12.3 |
| high | $900.0k | $243.3k | $-273.3k | $-547.3k | 92.5% | 3.9% | 17.0% | $341.4k | 67.2 |

Recommended: **standard**; static NPV alone would choose **standard**.

### (b) Twin-measured benefits (E2)

| Tier | Capex | Static NPV | Risk-adj. mean NPV | P5 NPV | P(loss) | Median IRR | r(σ) | PW overheads | CO₂ t/yr |
|---|---|---|---|---|---|---|---|---|---|
| lite | $150.0k | $97.4k | $52.6k | $-31.0k | 18.4% | 27.4% | 15.8% | $23.3k | 0.8 |
| standard | $450.0k | $73.6k | $-73.3k | $-285.0k | 71.2% | 10.6% | 18.5% | $140.0k | 12.3 |
| high | $900.0k | $-485.3k | $-731.5k | $-984.1k | 100.0% | -24.3% | 25.1% | $341.4k | 67.2 |

Recommended: **lite**; static NPV alone would choose **lite**.
