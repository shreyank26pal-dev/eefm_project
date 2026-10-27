"""
Overhead-aware stochastic capital budgeting for a Financial Digital Twin.

Extends the deterministic case study of our survey (Sec. 5, Eq. 17-21) by
  1. Monte Carlo simulation of benefit realisation, revenue growth and
     disruption shocks  -> NPV / IRR distributions, P(loss), VaR, CVaR
  2. Volatility-adjusted discount rate r(sigma)  (Survey Eq. 5)
  3. Explicit "hidden" twin overheads that the survey flags as Research Gap 2
     but does not price: edge energy (Xu & Liu [A1] wattage), hardware refresh,
     integration ramp-up, compliance/cyber cost, OPEX escalation
  4. Fidelity-tier selection: Lite vs Standard vs High-fidelity twin,
     i.e. "how much twin is economically worth building?"
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import economics as ee

# ---------------------------------------------------------------------------
# Twin fidelity tiers
# ---------------------------------------------------------------------------
TIERS = {
    "lite": {
        "label": "Lite twin (daily batch sync, RFID + ERP, cloud ML)",
        "hardware": 70_000, "software": 80_000, "fixed_opex": 20_000,
        "edge_nodes": 6, "watts": 15,
        "capture": {"B1": 0.55, "B2": 0.40, "B3": 0.30, "shock": 0.35},
        "refresh_year": None, "refresh_frac": 0.0,
    },
    "standard": {
        "label": "Standard twin (IoT edge + 66 W edge AI + smart-contract factoring) = survey case",
        "hardware": 280_000, "software": 170_000, "fixed_opex": 42_900,
        "edge_nodes": 20, "watts": 66,
        "capture": {"B1": 1.0, "B2": 1.0, "B3": 1.0, "shock": 1.0},
        "refresh_year": 3, "refresh_frac": 0.30,
    },
    "high": {
        "label": "High-fidelity twin (6G + federated GNN + ZKP, sub-ms sync)",
        "hardware": 600_000, "software": 300_000, "fixed_opex": 95_000,
        "edge_nodes": 60, "watts": 120,
        "capture": {"B1": 1.15, "B2": 1.20, "B3": 1.25, "shock": 1.40},
        "refresh_year": 3, "refresh_frac": 0.35,
    },
}


@dataclass
class ValuationInput:
    tier: str = "standard"
    horizon: int = 5
    # benefits (gross, per year, at full capture) - default = survey Eq. 12-14
    b1_working_capital: float = 184_110
    b2_credit_spread: float = 81_750
    b3_failure_avoidance: float = 78_500
    # extra benefit in a disruption year (from twin shock simulation)
    shock_benefit: float = 150_000
    shock_prob: float = 0.30
    # cost of capital (Survey Eq. 5): rf + beta(rm-rf) = 10% base WACC
    rf: float = 0.065
    rm: float = 0.135
    beta: float = 0.50
    gamma: float = 0.10
    # uncertainty
    realisation_low: float = 0.55
    realisation_mode: float = 0.90
    realisation_high: float = 1.15
    growth_mean: float = 0.04
    growth_sd: float = 0.03
    # hidden overheads (Research Gap 2)
    include_overheads: bool = True
    energy_price: float = 0.12        # $/kWh
    pue: float = 1.5                  # cooling / power usage effectiveness
    energy_escalation: float = 0.05
    opex_escalation: float = 0.03
    compliance_frac: float = 0.04     # cyber/compliance as % of capex per year
    ramp_year1: float = 0.60          # share of benefit realised in year 1
    ramp_year2: float = 0.90
    grid_co2: float = 0.71            # kg CO2 per kWh (India grid avg, CEA)
    n_sims: int = 5000
    seed: int = 2026
    overrides: dict = field(default_factory=dict)   # custom tier costs

    def wacc(self):
        return self.rf + self.beta * (self.rm - self.rf)


def _tier(vi: ValuationInput) -> dict:
    t = dict(TIERS[vi.tier])
    t["capture"] = dict(t["capture"])
    t.update(vi.overrides or {})
    t["capex"] = t["hardware"] + t["software"]
    return t


def energy_kwh(t: dict) -> float:
    return t["edge_nodes"] * t["watts"] * 8760 / 1000


def deterministic_flows(vi: ValuationInput, scale: float = 1.0) -> tuple[list[float], dict]:
    """Expected-value cash flows (no randomness) - used for static NPV,
    break-even and tornado sensitivity."""
    t = _tier(vi)
    c = t["capture"]
    T = vi.horizon
    gross = (vi.b1_working_capital * c["B1"] + vi.b2_credit_spread * c["B2"]
             + vi.b3_failure_avoidance * c["B3"])
    flows, parts = [-t["capex"]], {"energy": [], "refresh": [], "compliance": [], "opex": [], "ramp_loss": []}
    kwh = energy_kwh(t)
    for y in range(1, T + 1):
        if vi.include_overheads:
            ramp = vi.ramp_year1 if y == 1 else vi.ramp_year2 if y == 2 else 1.0
            grow = (1 + vi.growth_mean) ** (y - 1)
            ben = gross * scale * ramp * grow + vi.shock_prob * vi.shock_benefit * c["shock"] * scale
            energy = kwh * vi.pue * vi.energy_price * (1 + vi.energy_escalation) ** (y - 1)
            opex = t["fixed_opex"] * (1 + vi.opex_escalation) ** (y - 1)
            comp = vi.compliance_frac * t["capex"]
            refresh = t["hardware"] * t["refresh_frac"] if t["refresh_year"] == y else 0.0
            parts["ramp_loss"].append(gross * scale * grow * (1 - ramp))
        else:   # survey static assumptions: flat benefit, flat 45k opex incl. energy
            ben = gross * scale
            energy = kwh * vi.pue * vi.energy_price
            opex, comp, refresh = t["fixed_opex"], 0.0, 0.0
            parts["ramp_loss"].append(0.0)
        parts["energy"].append(energy); parts["opex"].append(opex)
        parts["compliance"].append(comp); parts["refresh"].append(refresh)
        flows.append(ben - energy - opex - comp - refresh)
    return flows, parts


def monte_carlo(vi: ValuationInput) -> dict:
    t = _tier(vi)
    c = t["capture"]
    rng = np.random.default_rng(vi.seed)
    N, T = vi.n_sims, vi.horizon
    wacc = vi.wacc()

    # persistent realisation factors per benefit stream (model risk)
    real = {k: rng.triangular(vi.realisation_low, vi.realisation_mode, vi.realisation_high, N)
            for k in ("B1", "B2", "B3")}
    growth = rng.normal(vi.growth_mean, vi.growth_sd, (N, T))
    rev_index = np.cumprod(1 + growth, axis=1) / (1 + growth[:, :1])   # year-1 index = 1
    shocks = rng.random((N, T)) < vi.shock_prob
    severity = rng.uniform(0.5, 1.5, (N, T))
    yearly_noise = rng.normal(1.0, 0.08, (N, T))
    energy_price = vi.energy_price * np.cumprod(
        1 + rng.normal(vi.energy_escalation, 0.04, (N, T)), axis=1) / (1 + vi.energy_escalation)

    years = np.arange(1, T + 1)
    if vi.include_overheads:
        ramp = np.where(years == 1, vi.ramp_year1, np.where(years == 2, vi.ramp_year2, 1.0))
    else:
        ramp = np.ones(T)

    b1 = vi.b1_working_capital * c["B1"] * real["B1"][:, None] * rev_index
    b2 = vi.b2_credit_spread * c["B2"] * real["B2"][:, None] * np.ones((N, T))
    b3 = vi.b3_failure_avoidance * c["B3"] * real["B3"][:, None] * np.ones((N, T))
    b_shock = shocks * severity * vi.shock_benefit * c["shock"]
    gross = (b1 + b2 + b3) * ramp * yearly_noise + (b_shock if vi.include_overheads else 0)

    kwh = energy_kwh(t)
    if vi.include_overheads:
        energy = kwh * vi.pue * energy_price
        opex = t["fixed_opex"] * (1 + vi.opex_escalation) ** (years - 1)
        comp = np.full(T, vi.compliance_frac * t["capex"])
        refresh = np.array([t["hardware"] * t["refresh_frac"] if t["refresh_year"] == y else 0.0
                            for y in years])
    else:
        energy = np.full((N, T), kwh * vi.pue * vi.energy_price)
        opex = np.full(T, t["fixed_opex"])
        comp = np.zeros(T)
        refresh = np.zeros(T)
    cf = gross - energy - opex - comp - refresh                    # (N, T)

    # --- Survey Eq. 5: sigma = coefficient of variation of operating cash flow ---
    sigma = float(np.mean(cf.std(axis=0) / np.abs(cf.mean(axis=0)).clip(min=1)))
    r_sigma = ee.risk_adjusted_rate(vi.rf, vi.beta, vi.rm, vi.gamma, sigma)

    disc_w = (1 + wacc) ** -years
    disc_r = (1 + r_sigma) ** -years
    npv_w = cf @ disc_w - t["capex"]
    npv_r = cf @ disc_r - t["capex"]

    # vectorised IRR by bisection
    lo, hi = np.full(N, -0.95), np.full(N, 5.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        f = (cf * (1 + mid[:, None]) ** -years).sum(axis=1) - t["capex"]
        lo = np.where(f > 0, mid, lo)
        hi = np.where(f > 0, hi, mid)
    irr = (lo + hi) / 2
    irr = np.where(irr > 4.9, np.nan, irr)

    # discounted payback
    cum = np.cumsum(cf * disc_w, axis=1) - t["capex"]
    paid = cum >= 0
    first = np.where(paid.any(axis=1), paid.argmax(axis=1) + 1, np.nan)

    q5 = np.percentile(npv_r, 5)
    hist, edges = np.histogram(npv_r, bins=40)
    det_flows, parts = deterministic_flows(vi)
    static_vi = ValuationInput(**{**asdict(vi), "include_overheads": False})
    static_flows, _ = deterministic_flows(static_vi)

    pw = lambda arr: float(np.sum(np.asarray(arr) * disc_w))
    return {
        "tier": vi.tier, "tier_label": t["label"], "capex": t["capex"],
        "wacc": wacc, "sigma": sigma, "r_sigma": r_sigma,
        "static": {
            "flows": static_flows,
            "npv": ee.npv(wacc, static_flows),
            "irr": ee.irr(static_flows),
            "payback": ee.simple_payback(static_flows),
        },
        "expected_overhead_aware": {
            "flows": det_flows, "npv": ee.npv(wacc, det_flows),
            "npv_r_sigma": ee.npv(r_sigma, det_flows),
            "irr": ee.irr(det_flows),
            "annual_worth": ee.annual_worth(wacc, det_flows),
            "future_worth": ee.future_worth(wacc, det_flows),
            "discounted_payback": ee.discounted_payback(wacc, det_flows),
            "profitability_index": ee.profitability_index(wacc, det_flows),
        },
        "mc": {
            "n": N,
            "npv_mean": float(npv_r.mean()), "npv_median": float(np.median(npv_r)),
            "npv_std": float(npv_r.std()), "npv_p5": float(q5),
            "npv_p95": float(np.percentile(npv_r, 95)),
            "cvar5": float(npv_r[npv_r <= q5].mean()),
            "prob_loss": float((npv_r < 0).mean()),
            "npv_wacc_mean": float(npv_w.mean()),
            "irr_median": float(np.nanmedian(irr)),
            "irr_p5": float(np.nanpercentile(irr, 5)),
            "payback_median": float(np.nanmedian(first)) if np.isfinite(first).any() else None,
            "never_pays_back": float(np.isnan(first).mean()),
            "hist": {"counts": hist.tolist(), "edges": edges.tolist()},
            "cf_fan": {"p5": np.percentile(cf, 5, axis=0).tolist(),
                       "p50": np.percentile(cf, 50, axis=0).tolist(),
                       "p95": np.percentile(cf, 95, axis=0).tolist()},
        },
        "overheads_pw": {k: pw(v) for k, v in parts.items()},
        "energy": {"kwh_per_year": kwh * vi.pue, "co2_tonnes_per_year": kwh * vi.pue * vi.grid_co2 / 1000},
        "breakeven": breakeven(vi),
        "tornado": tornado(vi),
    }


def breakeven(vi: ValuationInput) -> dict:
    """Minimum benefit-realisation factor k with NPV(k * benefits) = 0."""
    wacc = vi.wacc()
    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ee.npv(wacc, deterministic_flows(vi, mid)[0]) > 0:
            hi = mid
        else:
            lo = mid
    k = (lo + hi) / 2
    t = _tier(vi)
    return {"benefit_realisation_needed": k,
            "breakeven_gross_benefit_static": ee.breakeven_annual_benefit(
                t["capex"], t["fixed_opex"] + energy_kwh(t) * vi.pue * vi.energy_price, wacc, vi.horizon)}


def tornado(vi: ValuationInput, swing: float = 0.20) -> list[dict]:
    base = ee.npv(vi.wacc(), deterministic_flows(vi)[0])
    params = ["b1_working_capital", "b2_credit_spread", "b3_failure_avoidance",
              "shock_benefit", "shock_prob", "energy_price", "rm", "ramp_year1"]
    out = []
    d = asdict(vi)
    for p in params:
        lo_vi = ValuationInput(**{**d, p: d[p] * (1 - swing)})
        hi_vi = ValuationInput(**{**d, p: d[p] * (1 + swing)})
        lo = ee.npv(lo_vi.wacc(), deterministic_flows(lo_vi)[0])
        hi = ee.npv(hi_vi.wacc(), deterministic_flows(hi_vi)[0])
        out.append({"param": p, "low": lo - base, "high": hi - base, "range": abs(hi - lo)})
    # tier capex swing
    t = _tier(vi)
    for label, f in (("capex", None),):
        lo_vi = ValuationInput(**{**d, "overrides": {"hardware": t["hardware"] * 0.8, "software": t["software"] * 0.8}})
        hi_vi = ValuationInput(**{**d, "overrides": {"hardware": t["hardware"] * 1.2, "software": t["software"] * 1.2}})
        lo = ee.npv(vi.wacc(), deterministic_flows(lo_vi)[0])
        hi = ee.npv(vi.wacc(), deterministic_flows(hi_vi)[0])
        out.append({"param": "capex", "low": lo - base, "high": hi - base, "range": abs(hi - lo)})
    return sorted(out, key=lambda r: -r["range"])


def compare_tiers(vi: ValuationInput, max_prob_loss: float = 0.10) -> dict:
    rows = []
    for name in TIERS:
        r = monte_carlo(ValuationInput(**{**asdict(vi), "tier": name, "overrides": {}}))
        rows.append({
            "tier": name, "label": r["tier_label"], "capex": r["capex"],
            "static_npv": r["static"]["npv"],
            "risk_adj_npv_mean": r["mc"]["npv_mean"],
            "npv_p5": r["mc"]["npv_p5"], "prob_loss": r["mc"]["prob_loss"],
            "irr_median": r["mc"]["irr_median"], "r_sigma": r["r_sigma"],
            "overhead_pw": sum(r["overheads_pw"][k] for k in ("energy", "refresh", "compliance")),
            "co2_t": r["energy"]["co2_tonnes_per_year"],
        })
    feasible = [r for r in rows if r["prob_loss"] <= max_prob_loss] or rows
    best = max(feasible, key=lambda r: r["risk_adj_npv_mean"])
    best_static = max(rows, key=lambda r: r["static_npv"])
    return {"rows": rows, "recommended": best["tier"], "static_choice": best_static["tier"],
            "max_prob_loss": max_prob_loss}
