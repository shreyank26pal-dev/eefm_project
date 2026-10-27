"""
Reproducible experiments for the project report.
Run from backend/:   python -m experiments.run_experiments
Outputs: experiments/results/*.json, results.md and figures/*.png
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fdt import economics as ee
from fdt import ml
from fdt.twin import BusinessProfile, Shock, simulate
from fdt.valuation import ValuationInput, compare_tiers, monte_carlo

OUT = os.path.join(os.path.dirname(__file__), "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
TW, BA, NG = "#0f7b6c", "#d9822b", "#b4413c"
STRESS = [Shock("overload", 60, 30), Shock("supplier_delay", 150, 40), Shock("customer_stress", 240, 60)]
SEEDS = [11, 12, 13, 14, 15]
k = lambda x: f"${x/1e3:,.1f}k"
md = ["# FinTwin experimental results\n"]


def e1_replication():
    """E1: replicate survey case study, then add overheads, then risk."""
    static = monte_carlo(ValuationInput(include_overheads=False))
    aware = monte_carlo(ValuationInput())
    flows = static["static"]["flows"]
    res = {
        "survey_reported": {"npv": 684_814, "irr": 0.603, "payback": 1.50},
        "fintwin_static": {"npv": static["static"]["npv"], "irr": static["static"]["irr"],
                           "payback": static["static"]["payback"], "flows": flows},
        "overhead_aware_expected": {k_: aware["expected_overhead_aware"][k_] for k_ in
                                    ("npv", "irr", "discounted_payback", "annual_worth")},
        "monte_carlo": {k_: aware["mc"][k_] for k_ in ("npv_mean", "npv_p5", "npv_p95", "prob_loss",
                                                        "cvar5", "irr_median", "payback_median")},
        "sigma": aware["sigma"], "r_sigma": aware["r_sigma"], "overheads_pw": aware["overheads_pw"],
        "breakeven": aware["breakeven"], "tornado": aware["tornado"],
    }
    md.append("## E1 Survey replication and overhead-aware revaluation (Standard tier)\n")
    md.append("| Method | NPV | IRR | Payback |\n|---|---|---|---|")
    md.append(f"| Survey CIE-1 (reported) | $684,814 | 60.3% | 1.50 yr |")
    md.append(f"| FinTwin static (same assumptions) | ${res['fintwin_static']['npv']:,.0f} | {res['fintwin_static']['irr']:.1%} | {res['fintwin_static']['payback']:.2f} yr |")
    oa = res["overhead_aware_expected"]
    md.append(f"| + hidden overheads, expected @ WACC | ${oa['npv']:,.0f} | {oa['irr']:.1%} | {oa['discounted_payback']:.2f} yr (disc.) |")
    m = res["monte_carlo"]
    md.append(f"| + Monte Carlo @ r(σ)={res['r_sigma']:.1%} (mean) | ${m['npv_mean']:,.0f} | {m['irr_median']:.1%} (median) | {m['payback_median']:.0f} yr (median disc.) |\n")
    md.append(f"σ = {res['sigma']:.3f}; P(NPV<0) = {m['prob_loss']:.2%}; 90% NPV band {k(m['npv_p5'])} – {k(m['npv_p95'])}; CVaR5% {k(m['cvar5'])}; "
              f"break-even benefit realisation {res['breakeven']['benefit_realisation_needed']:.1%}.\n")
    md.append("PW of overheads: " + ", ".join(f"{a} {k(b)}" for a, b in res["overheads_pw"].items()) + "\n")

    # figure: NPV histogram + bridge
    vi = ValuationInput()
    r = monte_carlo(vi)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    e = np.array(r["mc"]["hist"]["edges"]); c = r["mc"]["hist"]["counts"]
    ax[0].bar((e[:-1] + e[1:]) / 2 / 1e3, c, width=np.diff(e) / 1e3, color=[NG if x < 0 else TW for x in e[:-1]])
    ax[0].axvline(r["static"]["npv"] / 1e3, color=BA, ls="--", label="static survey NPV")
    ax[0].axvline(r["mc"]["npv_mean"] / 1e3, color="k", ls=":", label="MC mean @ r(σ)")
    ax[0].set_xlabel("NPV ($k)"); ax[0].set_ylabel("simulations"); ax[0].legend(frameon=False)
    ax[0].set_title("(a) NPV distribution, 5,000 runs")
    yrs = np.arange(1, 6)
    ax[1].fill_between(yrs, np.array(r["mc"]["cf_fan"]["p5"]) / 1e3, np.array(r["mc"]["cf_fan"]["p95"]) / 1e3, color=TW, alpha=.15, label="P5–P95")
    ax[1].plot(yrs, np.array(r["mc"]["cf_fan"]["p50"]) / 1e3, color=TW, marker="o", label="median")
    ax[1].plot(yrs, np.array(r["static"]["flows"][1:]) / 1e3, color=BA, ls="--", label="static survey")
    ax[1].set_xlabel("year"); ax[1].set_ylabel("net cash flow ($k)"); ax[1].legend(frameon=False)
    ax[1].set_title("(b) Annual net cash flow fan")
    fig.savefig(os.path.join(FIG, "fig_e1_npv.png")); plt.close(fig)
    return res


def e2_measured_benefits():
    """E2: replace assumed benefits with twin-measured ones."""
    prof = BusinessProfile()
    steady = [simulate(prof, 365, [], s) for s in SEEDS]
    stress = [simulate(prof, 365, STRESS, s) for s in SEEDS]
    avg = lambda runs, key: float(np.mean([x["benefits_annualised"][key] for x in runs]))
    sd = lambda runs, key: float(np.std([x["benefits_annualised"][key] for x in runs]))
    keys = ["B1_working_capital", "B2_credit_spread", "B3_failure_avoidance", "total"]
    meas = {"steady": {key: [avg(steady, key), sd(steady, key)] for key in keys},
            "stress": {key: [avg(stress, key), sd(stress, key)] for key in keys}}
    summ = lambda runs, w, f: float(np.mean([x["summary"][w][f] for x in runs]))
    ops = {}
    for name, runs in (("steady", steady), ("stress", stress)):
        ops[name] = {w: {f: summ(runs, w, f) for f in ("avg_ccc", "avg_dio", "avg_dso", "avg_loan_rate",
                                                       "failures", "replacements", "derates", "maint_cost")}
                     for w in ("baseline", "twin")}
    shock_benefit = max(meas["stress"]["total"][0] - meas["steady"]["total"][0], 0)
    vi = ValuationInput(b1_working_capital=meas["steady"]["B1_working_capital"][0],
                        b2_credit_spread=meas["steady"]["B2_credit_spread"][0],
                        b3_failure_avoidance=max(meas["steady"]["B3_failure_avoidance"][0], 0),
                        shock_benefit=shock_benefit)
    r = monte_carlo(vi)
    res = {"measured": meas, "operations": ops, "shock_benefit": shock_benefit,
           "valuation_measured": {"static_npv": r["static"]["npv"], "mc_mean": r["mc"]["npv_mean"],
                                  "prob_loss": r["mc"]["prob_loss"], "irr_median": r["mc"]["irr_median"],
                                  "npv_p5": r["mc"]["npv_p5"]}}
    md.append("## E2 Twin-measured vs survey-assumed benefits (5 seeds, 365 days)\n")
    md.append("| Benefit | Survey assumption | Twin steady state | Twin under stress |\n|---|---|---|---|")
    survey = {"B1_working_capital": 184_110, "B2_credit_spread": 81_750, "B3_failure_avoidance": 78_500, "total": 344_360}
    for key in keys:
        md.append(f"| {key} | {k(survey[key])} | {k(meas['steady'][key][0])} ± {k(meas['steady'][key][1])} | {k(meas['stress'][key][0])} ± {k(meas['stress'][key][1])} |")
    md.append("")
    md.append("| Operational metric | Baseline (steady) | Twin (steady) | Baseline (stress) | Twin (stress) |\n|---|---|---|---|---|")
    for f in ("avg_ccc", "avg_dio", "avg_dso", "avg_loan_rate", "failures", "replacements", "derates", "maint_cost"):
        vals = [ops["steady"]["baseline"][f], ops["steady"]["twin"][f], ops["stress"]["baseline"][f], ops["stress"]["twin"][f]]
        fmt = (lambda v: f"{v:.2%}") if f == "avg_loan_rate" else (lambda v: k(v)) if f == "maint_cost" else (lambda v: f"{v:.1f}")
        md.append(f"| {f} | " + " | ".join(fmt(v) for v in vals) + " |")
    vm = res["valuation_measured"]
    md.append(f"\nWith measured benefits (shock-year extra {k(shock_benefit)}, p=0.30): static NPV {k(vm['static_npv'])}, "
              f"MC mean {k(vm['mc_mean'])}, P(loss) {vm['prob_loss']:.1%}, median IRR {vm['irr_median']:.1%}.\n")

    # figure: CCC + cumulative maintenance under stress, seed 11
    s = stress[0]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    d = np.arange(1, 366)
    for a in ax:
        for sh in STRESS:
            a.axvspan(sh.start, sh.start + sh.length, color=NG, alpha=.07)
    ax[0].plot(d, s["series"]["baseline"]["ccc"], color=BA, lw=1, label="baseline")
    ax[0].plot(d, s["series"]["twin"]["ccc"], color=TW, lw=1, label="twin")
    ax[0].set_title("(a) Cash conversion cycle (days)"); ax[0].legend(frameon=False)
    ax[1].plot(d, np.array(s["series"]["baseline"]["loan_rate"]) * 100, color=BA, lw=1)
    ax[1].plot(d, np.array(s["series"]["twin"]["loan_rate"]) * 100, color=TW, lw=1)
    ax[1].set_title("(b) Loan rate, Eq. 9 (%)")
    ax[2].plot(d, np.cumsum(s["series"]["baseline"]["maint_cost"]) / 1e3, color=BA, lw=1.4)
    ax[2].plot(d, np.cumsum(s["series"]["twin"]["maint_cost"]) / 1e3, color=TW, lw=1.4)
    ax[2].set_title("(c) Cumulative failure+maint. cost ($k)")
    for a in ax: a.set_xlabel("day")
    fig.savefig(os.path.join(FIG, "fig_e2_twin.png")); plt.close(fig)
    return res


def e3_tiers(e2):
    m = e2["measured"]["steady"]
    measured_vi = ValuationInput(b1_working_capital=m["B1_working_capital"][0],
                                 b2_credit_spread=m["B2_credit_spread"][0],
                                 b3_failure_avoidance=max(m["B3_failure_avoidance"][0], 0),
                                 shock_benefit=e2["shock_benefit"])
    res = {"survey_benefits": compare_tiers(ValuationInput()),
           "measured_benefits": compare_tiers(measured_vi)}
    md.append("## E3 Twin fidelity right-sizing (Gap 2)\n")
    for label, title in (("survey_benefits", "(a) Survey-assumed benefits"), ("measured_benefits", "(b) Twin-measured benefits (E2)")):
        md.append(f"### {title}\n")
        md.append("| Tier | Capex | Static NPV | Risk-adj. mean NPV | P5 NPV | P(loss) | Median IRR | r(σ) | PW overheads | CO₂ t/yr |\n|---|---|---|---|---|---|---|---|---|---|")
        for r in res[label]["rows"]:
            md.append(f"| {r['tier']} | {k(r['capex'])} | {k(r['static_npv'])} | {k(r['risk_adj_npv_mean'])} | {k(r['npv_p5'])} | {r['prob_loss']:.1%} | {r['irr_median']:.1%} | {r['r_sigma']:.1%} | {k(r['overhead_pw'])} | {r['co2_t']:.1f} |")
        md.append(f"\nRecommended: **{res[label]['recommended']}**; static NPV alone would choose **{res[label]['static_choice']}**.\n")
    rows = res["survey_benefits"]["rows"]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    x = np.arange(len(rows)); w = .27
    ax.bar(x - w, [r["static_npv"] / 1e3 for r in rows], w, color=BA, label="static NPV")
    ax.bar(x, [r["risk_adj_npv_mean"] / 1e3 for r in rows], w, color=TW, label="risk-adj. mean NPV")
    ax.bar(x + w, [r["npv_p5"] / 1e3 for r in rows], w, color="#9bb7c4", label="P5 NPV")
    ax.axhline(0, color="k", lw=.6); ax.set_xticks(x, [r["tier"] for r in rows]); ax.set_ylabel("$k")
    ax.legend(frameon=False, fontsize=8); ax.set_title("Static vs risk-adjusted NPV by twin tier")
    fig.savefig(os.path.join(FIG, "fig_e3_tiers.png")); plt.close(fig)
    return res


def e4_ml():
    path = os.path.join(OUT, "ml_results.json")
    res = json.load(open(path))
    md.append("## E4 ML models\n")
    md.append("### Failure-probability model (AI4I 2020, 10,000 records, 25% stratified hold-out, threshold tuned on out-of-fold train predictions; C_FN=$25k, C_FP=$1.5k)\n")
    md.append("| Model | ROC-AUC | PR-AUC | F1@0.5 | Cost threshold | Recall@thr | Expected cost | Saving vs run-to-failure |\n|---|---|---|---|---|---|---|---|")
    for n, r in res["failure_model"].items():
        md.append(f"| {n} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['f1_at_0.5']:.3f} | {r['cost_opt_threshold']:.2f} | {r['recall_at_cost_threshold']:.3f} | ${r['expected_cost']:,.0f} | {r['saving_vs_run_to_failure_pct']:.1f}% |")
    md.append("\n### Invoice settlement (DSO) model (6,000 synthetic invoices, 25% hold-out)\n")
    md.append("| Model | MAE (days) | R² |\n|---|---|---|")
    for n, r in res["dso_model"].items():
        if isinstance(r, dict):
            md.append(f"| {n} | {r['mae_days']:.2f} | {r['r2']:.3f} |")
    md.append(f"\nCounterfactual twin-verified-delivery effect: −{res['dso_model']['twin_effect_days']:.2f} days per invoice.\n")
    return res


def e5_policy_sweep():
    rows = []
    for tb in (120, 160, 200, 240):
        for scen, shocks in (("steady", []), ("overload", [Shock("overload", 60, 30), Shock("overload", 220, 30)])):
            runs = [simulate(BusinessProfile(time_based_wear=tb), 365, shocks, s) for s in SEEDS]
            rows.append({"interval": tb, "scenario": scen,
                         "baseline_cost": float(np.mean([r["summary"]["baseline"]["maint_cost"] for r in runs])),
                         "twin_cost": float(np.mean([r["summary"]["twin"]["maint_cost"] for r in runs])),
                         "baseline_fail": float(np.mean([r["summary"]["baseline"]["failures"] for r in runs])),
                         "twin_fail": float(np.mean([r["summary"]["twin"]["failures"] for r in runs]))})
    md.append("## E5 Maintenance policy: fixed time-based interval vs twin Eq. 7 rule (6 machines, 5 seeds)\n")
    md.append("| Interval (wear min) | Scenario | Time-based cost | Twin cost | Time-based failures | Twin failures |\n|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['interval']} | {r['scenario']} | {k(r['baseline_cost'])} | {k(r['twin_cost'])} | {r['baseline_fail']:.1f} | {r['twin_fail']:.1f} |")
    md.append("")
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for scen, col in (("steady", TW), ("overload", NG)):
        sub = [r for r in rows if r["scenario"] == scen]
        ax.plot([r["interval"] for r in sub], [r["baseline_cost"] / 1e3 for r in sub], marker="o", color=col, label=f"time-based ({scen})")
        ax.axhline(np.mean([r["twin_cost"] for r in sub]) / 1e3, color=col, ls="--", label=f"twin Eq. 7 ({scen})")
    ax.set_xlabel("fixed replacement interval (tool-wear min)"); ax.set_ylabel("annual cost ($k)")
    ax.legend(frameon=False, fontsize=7.5); ax.set_title("Policy cost vs fixed interval")
    fig.savefig(os.path.join(FIG, "fig_e5_policy.png")); plt.close(fig)
    return rows


def e6_performance():
    t = {}
    t0 = time.perf_counter(); simulate(BusinessProfile(), 365, STRESS, 11); t["twin_365d_s"] = time.perf_counter() - t0
    for n in (1000, 5000, 20000):
        t0 = time.perf_counter(); r = monte_carlo(ValuationInput(n_sims=n)); dt = time.perf_counter() - t0
        t[f"mc_{n}"] = {"seconds": dt, "npv_mean": r["mc"]["npv_mean"], "std_error": r["mc"]["npv_std"] / np.sqrt(n)}
    t0 = time.perf_counter(); compare_tiers(ValuationInput()); t["tiers_s"] = time.perf_counter() - t0
    md.append("## E6 Computational performance (single CPU core)\n")
    md.append(f"- 365-day dual-world twin with 3 shocks: {t['twin_365d_s']:.2f} s\n- Tier comparison (3 × 5,000 runs + sensitivity): {t['tiers_s']:.2f} s")
    md.append("\n| MC runs | Time (s) | Mean NPV | Std. error |\n|---|---|---|---|")
    for n in (1000, 5000, 20000):
        x = t[f"mc_{n}"]
        md.append(f"| {n:,} | {x['seconds']:.3f} | {k(x['npv_mean'])} | {k(x['std_error'])} |")
    md.append("")
    return t


if __name__ == "__main__":
    if not os.path.exists(os.path.join(OUT, "ml_results.json")):
        import runpy
        runpy.run_module("fdt.ml", run_name="__main__")
    results = {"E1": e1_replication(), "E2": e2_measured_benefits(),
               "E4": e4_ml(), "E5": e5_policy_sweep(), "E6": e6_performance()}
    results["E3"] = e3_tiers(results["E2"])
    with open(os.path.join(OUT, "experiments.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    with open(os.path.join(OUT, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))
