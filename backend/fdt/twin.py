"""
FinTwin - the operational-to-financial Digital Twin simulator.

The twin mirrors a manufacturing SME day by day:
  physical layer  : CNC tool spindles (AI4I-style telemetry), inventory, shipments
  transaction layer: invoices issued / settled, supplier payables
  financial layer : DIO, DSO, DPO, CCC, working capital, real-time credit score,
                    loan rate, maintenance & failure cost

It runs two worlds with COMMON RANDOM NUMBERS so that the difference between
them is the measured economic benefit of the twin:
  * "baseline" - static ERP view, time-based maintenance, lender sees only history
  * "twin"     - live visibility, ML condition-based replacement (Survey Eq. 7),
                 smart-contract factoring on verified delivery, real-time credit
                 score (Survey Eq. 8-9)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import economics as ee
from . import ml


@dataclass
class BusinessProfile:
    name: str = "Precision Auto Components (demo SME)"
    currency: str = "$"
    annual_revenue: float = 35_000_000
    cogs_ratio: float = 0.70
    dio: float = 48.0            # Days Inventory Outstanding (baseline)
    dso: float = 42.0            # Days Sales Outstanding (baseline terms-mix)
    dpo: float = 35.0            # Days Payables Outstanding
    wc_carry_rate: float = 0.06  # cost of carrying working capital
    credit_line: float = 2_500_000
    credit_hist: float = 0.30    # historical credit score C_hist in [0,1]
    alpha: float = 0.40          # weight on history in Eq. 8
    r_base: float = 0.08
    dr_max: float = 0.15
    k_credit: float = 3.0
    n_machines: int = 6
    c_failure: float = 9_000.0   # unplanned failure: scrap + downtime + rush repair
    c_replace: float = 1_200.0   # planned tool/spindle replacement
    time_based_wear: float = 200 # baseline policy: OEM interval, replace at this wear [min]
    design_life_wear: float = 200 # challenger design life used for capital recovery
    wear_per_day: float = 7.0
    derate_output_loss: float = 0.20  # output lost when a machine is derated
    contribution_margin: float = 0.30
    invoices_per_day: int = 4
    # twin improvement targets (what full visibility can achieve, Survey Sec. 5.2)
    dio_twin: float = 34.0
    twin_verified_share: float = 0.90

    def to_dict(self):
        return asdict(self)


@dataclass
class Shock:
    kind: str          # supplier_delay | overload | customer_stress | demand_drop
    start: int
    length: int
    magnitude: float = 1.0


INDUSTRIES = ["automotive", "electronics", "pharma", "fmcg", "construction"]


def _machine_paths(rng, days, m, profile, shocks):
    """Exogenous telemetry draws for m machines over `days` (before wear resets)."""
    air = 300 + np.cumsum(rng.normal(0, 0.25, (days, m)), axis=0).clip(-4, 4)
    proc = air + 10 + rng.normal(0, 0.6, (days, m))
    rpm = rng.normal(1540, 150, (days, m)).clip(1200, 2800)
    torque = (61500 / rpm + rng.normal(0, 6, (days, m))).clip(5, 75)   # ~constant power
    rpm_nom, torque_nom = rpm.copy(), torque.copy()                     # derated = nominal load
    for s in shocks:
        if s.kind == "overload":
            sl = slice(s.start, min(days, s.start + s.length))
            torque[sl] += 14 * s.magnitude
            rpm[sl] -= 120 * s.magnitude
    types = rng.choice(["L", "M", "H"], m, p=[0.6, 0.3, 0.1])
    wear_inc = rng.uniform(0.7, 1.3, (days, m)) * profile.wear_per_day
    u_fail = rng.random((days, m))
    return dict(air=air, proc=proc, rpm=rpm, torque=torque, types=types,
                rpm_nom=rpm_nom, torque_nom=torque_nom,
                wear_inc=wear_inc, u_fail=u_fail)


WEAR_GRID = np.arange(0, 281, 10.0)


def _pfail_surface(paths, days, m, derated=False):
    """Batch ML inference: P_fail for every (day, machine, wear-grid point).
    One vectorised call instead of days*m calls; wear is interpolated later."""
    g = len(WEAR_GRID)
    t_idx, j_idx, w_idx = np.meshgrid(np.arange(days), np.arange(m), np.arange(g), indexing="ij")
    t_idx, j_idx, w_idx = t_idx.ravel(), j_idx.ravel(), w_idx.ravel()
    rows = {
        "type": paths["types"][j_idx], "air_temp": paths["air"][t_idx, j_idx],
        "proc_temp": paths["proc"][t_idx, j_idx],
        "rpm": paths["rpm_nom" if derated else "rpm"][t_idx, j_idx],
        "torque": paths["torque_nom" if derated else "torque"][t_idx, j_idx],
        "tool_wear": WEAR_GRID[w_idx],
    }
    surf = ml.predict_failure_proba(rows).reshape(days, m, g)
    # physics prior: hazard is non-decreasing in cumulative wear D(t)
    return np.maximum.accumulate(surf, axis=2)


def _p_at(surface, t, j, wear):
    return float(np.interp(wear, WEAR_GRID, surface[t, j]))


def _run_maintenance(paths, surface, surface_der, days, m, profile, policy,
                     derate_cost, rate_daily=0.10 / 365):
    """Simulate tool wear, ML failure probability and replacement decisions.

    policy = "time"  : replace when wear >= time_based_wear (baseline practice)
    policy = "twin"  : condition-based rule from Survey Eq. (7):
        replace when the marginal cost of running the defender one more day
            MC(t+1) = E[O&M(t+1)] + P_fail(D_t) * C_catastrophe
        exceeds the EUAC of a fresh challenger
            EUAC_ch = C_replace*(A/P, i, L) + E[O&M_new] + P_fail(new) * C_catastrophe
        (salvage of a worn consumable tool ~ 0, so dS and r*S vanish)
    """
    wear = np.zeros(m)
    out_p = np.zeros((days, m))
    failures = np.zeros(days)
    replacements = np.zeros(days)
    derates = np.zeros(days)
    maint_cost = np.zeros(days)
    life_days = max(int(profile.design_life_wear / profile.wear_per_day), 1)
    capital_recovery = profile.c_replace * ee.ap(rate_daily, life_days)
    om = lambda w: 15 + 0.25 * w            # energy / quality drift grows with wear
    decisions = []

    for t in range(days):
        for j in range(m):
            surf = surface
            p = _p_at(surface, t, j, wear[j])
            if policy == "twin":
                # control action a_t (Survey Eq. 4): derate an overloaded machine when the
                # avoided expected failure cost exceeds the lost contribution margin
                p_der = _p_at(surface_der, t, j, wear[j])
                if (p - p_der) * profile.c_failure > derate_cost:
                    p, surf = p_der, surface_der
                    derates[t] += 1
                    maint_cost[t] += derate_cost
            out_p[t, j] = p
            if paths["u_fail"][t, j] < p:                 # physical failure occurs
                failures[t] += 1
                maint_cost[t] += profile.c_failure
                wear[j] = 0
                continue
            if policy == "time":
                replace = wear[j] >= profile.time_based_wear
            else:
                # look ahead to period t+1 (Eq. 7 is a *next-period* marginal cost)
                p_next = _p_at(surf, t, j, wear[j] + profile.wear_per_day)
                mc_next = om(wear[j] + profile.wear_per_day) + p_next * profile.c_failure
                # challenger: average hazard of a fresh tool over its early life
                p_new = float(surf[t, j, WEAR_GRID <= 150].mean())
                euac_ch = capital_recovery + om(life_days * profile.wear_per_day / 2) + p_new * profile.c_failure
                replace = mc_next > euac_ch
                if replace and len(decisions) < 50:
                    decisions.append({"day": t, "machine": j, "wear": round(float(wear[j]), 1),
                                      "p_fail_next": round(p_next, 4), "mc_next": round(float(mc_next), 2),
                                      "euac_challenger": round(float(euac_ch), 2)})
            if replace:
                replacements[t] += 1
                maint_cost[t] += profile.c_replace
                wear[j] = 0
            else:
                wear[j] += paths["wear_inc"][t, j]
    return dict(p_fail=out_p, failures=failures, replacements=replacements, derates=derates,
                maint_cost=maint_cost, decisions=decisions)


def simulate(profile: BusinessProfile | None = None, days: int = 365,
             shocks: list[Shock] | None = None, seed: int = 11) -> dict:
    profile = profile or BusinessProfile()
    shocks = shocks or []
    rng = np.random.default_rng(seed)
    m = profile.n_machines
    rev_day = profile.annual_revenue / 365
    cogs_day = rev_day * profile.cogs_ratio

    # ---------- common random numbers ----------
    paths = _machine_paths(rng, days, m, profile, shocks)
    surface = _pfail_surface(paths, days, m)
    surface_der = _pfail_surface(paths, days, m, derated=True) if any(
        x.kind == "overload" for x in shocks) else surface
    # lost contribution margin of running one machine at reduced load for a day
    derate_cost = rev_day / m * profile.derate_output_loss * profile.contribution_margin
    demand_noise = rng.normal(0, 0.10, days)
    season = 1 + 0.08 * np.sin(np.arange(days) / 365 * 2 * np.pi)
    dio_noise = rng.normal(0, 1.2, days)
    dpo_noise = rng.normal(0, 0.8, days)
    n_inv = days * profile.invoices_per_day
    inv = {
        "customer_credit": rng.beta(5, 2, n_inv),
        "invoice_amount": rng.lognormal(10.5, 0.9, n_inv),
        "terms_days": rng.choice([30, 45, 60], n_inv, p=[0.6, 0.3, 0.1]),
        "industry": rng.choice(INDUSTRIES, n_inv),
        "past_avg_delay": np.clip(rng.gamma(2.0, 2.5, n_inv), 0, 60),
        "quarter_end": ((np.arange(n_inv) // profile.invoices_per_day) % 91 > 80).astype(int),
    }
    inv_day = np.arange(n_inv) // profile.invoices_per_day
    verify_draw = rng.random(n_inv)
    pay_noise = rng.normal(0, 3.0, n_inv)

    # shock modifiers
    demand_mult = np.ones(days)
    supplier_add = np.zeros(days)
    for s in shocks:
        sl = slice(s.start, min(days, s.start + s.length))
        if s.kind == "demand_drop":
            demand_mult[sl] *= 1 - 0.25 * s.magnitude
        elif s.kind == "supplier_delay":
            supplier_add[sl] += 12 * s.magnitude
        elif s.kind == "customer_stress":
            stressed = (inv_day >= s.start) & (inv_day < s.start + s.length)
            inv["customer_credit"] = np.where(stressed, inv["customer_credit"] * (1 - 0.35 * s.magnitude),
                                              inv["customer_credit"])
            inv["past_avg_delay"] = np.where(stressed, inv["past_avg_delay"] + 8 * s.magnitude,
                                             inv["past_avg_delay"])

    worlds = {}
    for world in ("baseline", "twin"):
        twin = world == "twin"
        # ----- receivables (ML DSO model, Survey Eq. 2 tau_j) -----
        verified = (verify_draw < profile.twin_verified_share).astype(int) if twin else np.zeros(n_inv, int)
        dtp = ml.predict_days_to_pay(inv | {"twin_verified_delivery": verified}) + pay_noise
        dtp = dtp.clip(5, 150)
        # baseline DSO is scaled so that its mean matches the firm's reported DSO
        if world == "baseline":
            dso_scale = profile.dso / dtp.mean()
        dtp_scaled = dtp * dso_scale
        dso_series = np.array([dtp_scaled[(inv_day > t - 30) & (inv_day <= t)].mean() for t in range(days)])
        within_terms = (dtp_scaled <= inv["terms_days"] * 1.05).astype(float)
        coll_eff = np.array([within_terms[(inv_day > t - 30) & (inv_day <= t)].mean() for t in range(days)])

        # ----- inventory: AR(1) toward target; twin reacts to supplier shocks faster ----
        target = profile.dio_twin if twin else profile.dio
        dio = np.zeros(days)
        x = profile.dio
        react = 0.35 if twin else 0.08             # speed of corrective action
        for t in range(days):
            x = x + react * (target - x) + dio_noise[t] + (0.6 if twin else 1.0) * supplier_add[t] * 0.15
            dio[t] = max(x, 5)
        dpo = profile.dpo + dpo_noise

        # ----- physical assets -----
        mt = _run_maintenance(paths, surface, surface_der, days, m, profile,
                              "twin" if twin else "time", derate_cost)
        fail_rate_30 = np.convolve(mt["failures"], np.ones(30), "full")[:days] / (30 * m)

        # ----- financial state -----
        rev = rev_day * season * (1 + demand_noise) * demand_mult
        on_time = 1 - np.clip(supplier_add / 40 + fail_rate_30 * 3, 0, 0.6)
        c_rt = 0.40 * on_time + 0.30 * (1 - np.clip(fail_rate_30 * 10, 0, 1)) + 0.30 * coll_eff
        if twin:
            c_comp = ee.composite_credit(profile.credit_hist, c_rt, profile.alpha)
        else:
            c_comp = np.full(days, profile.credit_hist)    # lender sees only history
        loan_r = profile.r_base + profile.dr_max * np.exp(-profile.k_credit * c_comp)
        ccc = dio + dso_series - dpo
        inv_val = dio * cogs_day
        ar = dso_series * rev_day
        ap_val = dpo * cogs_day
        wc = inv_val + ar - ap_val
        wc_cost = wc * profile.wc_carry_rate / 365
        interest = profile.credit_line * loan_r / 365

        worlds[world] = {
            "dio": dio, "dso": dso_series, "dpo": dpo, "ccc": ccc, "wc": wc,
            "wc_cost": wc_cost, "credit_rt": c_rt, "credit_comp": c_comp, "loan_rate": loan_r,
            "interest": interest, "revenue": rev, "maint_cost": mt["maint_cost"],
            "failures": mt["failures"], "replacements": mt["replacements"], "derates": mt["derates"],
            "p_fail_max": mt["p_fail"].max(axis=1), "p_fail_mean": mt["p_fail"].mean(axis=1),
            "coll_eff": coll_eff, "decisions": mt["decisions"],
        }

    b, w = worlds["baseline"], worlds["twin"]
    scale = 365 / days
    benefits = {
        # B1: working-capital carrying cost saving  (Survey Eq. 11-12)
        "B1_working_capital": float((b["wc_cost"].sum() - w["wc_cost"].sum()) * scale),
        # B2: borrowing spread compression           (Survey Eq. 8-9, 13)
        "B2_credit_spread": float((b["interest"].sum() - w["interest"].sum()) * scale),
        # B3: failure & defect avoidance             (Survey Eq. 7, 14)
        "B3_failure_avoidance": float((b["maint_cost"].sum() - w["maint_cost"].sum()) * scale),
    }
    benefits["total"] = sum(benefits.values())

    def summary(x):
        return {
            "avg_dio": float(x["dio"].mean()), "avg_dso": float(x["dso"].mean()),
            "avg_dpo": float(x["dpo"].mean()), "avg_ccc": float(x["ccc"].mean()),
            "avg_wc": float(x["wc"].mean()), "avg_loan_rate": float(x["loan_rate"].mean()),
            "failures": int(x["failures"].sum()), "replacements": int(x["replacements"].sum()),
            "derates": int(x["derates"].sum()),
            "maint_cost": float(x["maint_cost"].sum()),
            "avg_credit": float(x["credit_comp"].mean()),
        }

    def series(x):
        keys = ["dio", "dso", "ccc", "wc", "loan_rate", "credit_comp", "p_fail_max",
                "maint_cost", "failures"]
        return {k: np.round(x[k], 5).tolist() for k in keys}

    return {
        "profile": profile.to_dict(), "days": days,
        "shocks": [asdict(s) for s in shocks],
        "summary": {"baseline": summary(b), "twin": summary(w)},
        "benefits_annualised": benefits,
        "series": {"baseline": series(b), "twin": series(w)},
        "model_info": ml.failure_meta(),
        "replacement_log": w["decisions"][:20],
    }
