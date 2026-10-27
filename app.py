"""
FinTwin - Financial Digital Twin for Business
Flask web application (CIE Activity 2 mini project, Group 6).

Run locally:   python app.py          -> http://127.0.0.1:5000
Production:    gunicorn app:app
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, fields

import numpy as np
from flask import Flask, jsonify, render_template, request

from fdt import advisor, db
from fdt import economics as ee
from fdt import ml
from fdt.twin import BusinessProfile, Shock, simulate
from fdt.valuation import TIERS, ValuationInput, compare_tiers, monte_carlo

app = Flask(__name__)
db.init()
ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def clean(o):
    """Make numpy / NaN values JSON-safe."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def _build(cls, data: dict):
    names = {f.name: f for f in fields(cls)}
    kw = {}
    for k, v in (data or {}).items():
        if k in names and v is not None and v != "":
            t = names[k].type
            if t in ("float", float):
                v = float(v)
            elif t in ("int", int):
                v = int(float(v))
            elif t in ("bool", bool):
                v = v in (True, "true", "on", 1, "1")
            kw[k] = v
    return cls(**kw)


def _limits(vi: ValuationInput):
    vi.n_sims = max(200, min(vi.n_sims, 20000))
    vi.horizon = max(1, min(vi.horizon, 15))
    if vi.tier not in TIERS:
        vi.tier = "standard"
    return vi


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "llm": advisor.llm_available() or "rule-based"}


@app.get("/api/defaults")
def defaults():
    return jsonify(clean({
        "profile": BusinessProfile().to_dict(),
        "valuation": {k: v for k, v in asdict(ValuationInput()).items() if k != "overrides"},
        "tiers": TIERS,
        "model": ml.failure_meta(),
    }))


@app.get("/api/model-info")
def model_info():
    path = os.path.join(ROOT, "experiments", "results", "ml_results.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    return jsonify({"selected": ml.failure_meta(), "results": data})


# ---------------------------------------------------------------------------
# digital twin
# ---------------------------------------------------------------------------
@app.post("/api/twin/simulate")
def twin_simulate():
    body = request.get_json(force=True) or {}
    profile = _build(BusinessProfile, body.get("profile", {}))
    days = max(60, min(int(body.get("days", 365)), 730))
    shocks = [Shock(s["kind"], int(s["start"]), int(s["length"]), float(s.get("magnitude", 1)))
              for s in body.get("shocks", []) if s.get("kind")]
    res = simulate(profile, days, shocks, int(body.get("seed", 11)))
    return jsonify(clean(res))


@app.post("/api/twin/calibrate")
def twin_calibrate():
    """Run the twin with and without a disruption and convert the measured
    differences into benefit inputs for capital budgeting (closes the loop
    between the operational twin and the valuation model)."""
    body = request.get_json(force=True) or {}
    profile = _build(BusinessProfile, body.get("profile", {}))
    seeds = [11, 12]
    steady = [simulate(profile, 365, [], s)["benefits_annualised"] for s in seeds]
    shocks = [Shock("overload", 60, 30), Shock("supplier_delay", 150, 40), Shock("customer_stress", 240, 60)]
    shock = [simulate(profile, 365, shocks, s)["benefits_annualised"] for s in seeds]
    avg = lambda lst, k: float(np.mean([x[k] for x in lst]))
    out = {
        "b1_working_capital": avg(steady, "B1_working_capital"),
        "b2_credit_spread": avg(steady, "B2_credit_spread"),
        "b3_failure_avoidance": max(avg(steady, "B3_failure_avoidance"), 0.0),
        "shock_benefit": max(avg(shock, "total") - avg(steady, "total"), 0.0),
        "seeds": seeds,
    }
    return jsonify(clean(out))


# ---------------------------------------------------------------------------
# capital budgeting
# ---------------------------------------------------------------------------
@app.post("/api/valuation")
def valuation():
    vi = _limits(_build(ValuationInput, request.get_json(force=True) or {}))
    return jsonify(clean(monte_carlo(vi)))


@app.post("/api/tiers")
def tiers():
    body = request.get_json(force=True) or {}
    vi = _limits(_build(ValuationInput, body))
    vi.n_sims = min(vi.n_sims, 5000)
    return jsonify(clean(compare_tiers(vi, float(body.get("max_prob_loss", 0.10)))))


# ---------------------------------------------------------------------------
# replacement analysis (Survey Eq. 6 & 7)
# ---------------------------------------------------------------------------
@app.post("/api/replacement")
def replacement():
    b = request.get_json(force=True) or {}
    g = lambda k, d: float(b.get(k, d))
    rate = g("rate", 0.10)
    # challenger
    ch_cost, ch_dep, ch_om0, ch_omg = g("ch_cost", 120000), g("ch_dep", 0.20), g("ch_om0", 6000), g("ch_omg", 2500)
    n_star, euac_min, table = ee.economic_life(
        ch_cost, lambda n: ch_cost * (1 - ch_dep) ** n, lambda t: ch_om0 + ch_omg * (t - 1), rate, 15)
    # defender (current asset)
    d_cost, d_age, d_dep = g("d_cost", 100000), int(g("d_age", 3)), g("d_dep", 0.20)
    d_om0, d_omg, c_cat = g("d_om0", 7000), g("d_omg", 3500), g("c_cat", 60000)
    s_now = d_cost * (1 - d_dep) ** d_age
    s_next = d_cost * (1 - d_dep) ** (d_age + 1)
    om_next = d_om0 + d_omg * d_age
    tele = {"type": b.get("type", "M"), "air_temp": g("air_temp", 301.5), "proc_temp": g("proc_temp", 311.0),
            "rpm": g("rpm", 1200), "torque": g("torque", 55), "tool_wear": g("tool_wear", 200)}
    p_fail = float(ml.predict_failure_proba([tele])[0])
    mc = ee.marginal_cost_next_year(s_now, s_next, rate, om_next, p_fail, c_cat)
    mc_no_ai = ee.marginal_cost_next_year(s_now, s_next, rate, om_next, 0.0, c_cat)
    # defender marginal-cost curve for the next few years (deterministic + AI hazard)
    curve = []
    for k in range(0, 8):
        a = d_age + k
        curve.append({"year": k + 1,
                      "mc": ee.marginal_cost_next_year(d_cost * (1 - d_dep) ** a, d_cost * (1 - d_dep) ** (a + 1),
                                                       rate, d_om0 + d_omg * a, 0.0, c_cat)})
    keep_years = next((c["year"] - 1 for c in curve if c["mc"] > euac_min), len(curve))
    return jsonify(clean({
        "challenger": {"economic_life": n_star, "min_euac": euac_min, "table": table},
        "defender": {"salvage_now": s_now, "salvage_next": s_next, "om_next": om_next,
                     "p_fail": p_fail, "mc_next": mc, "mc_next_without_ai": mc_no_ai, "curve": curve},
        "decision": "REPLACE NOW" if mc > euac_min else "KEEP DEFENDER",
        "decision_without_ai": "REPLACE NOW" if mc_no_ai > euac_min else "KEEP DEFENDER",
        "keep_years_deterministic": keep_years,
        "telemetry": tele,
    }))


# ---------------------------------------------------------------------------
# Time-value-of-money toolkit
# ---------------------------------------------------------------------------
@app.post("/api/tvm")
def tvm():
    b = request.get_json(force=True) or {}
    rate = float(b.get("rate", 0.10))
    flows = [float(x) for x in b.get("cashflows", [-450000, 299360, 299360, 299360, 299360, 299360])]
    n = len(flows) - 1
    dep_cost, dep_salv = float(b.get("dep_cost", 280000)), float(b.get("dep_salvage", 28000))
    return jsonify(clean({
        "pw": ee.npv(rate, flows), "fw": ee.future_worth(rate, flows), "aw": ee.annual_worth(rate, flows),
        "irr": ee.irr(flows), "payback": ee.simple_payback(flows),
        "discounted_payback": ee.discounted_payback(rate, flows),
        "pi": ee.profitability_index(rate, flows) if flows[0] < 0 else None,
        "factors": {"P/F": ee.pf(rate, n), "F/P": ee.fp(rate, n), "P/A": ee.pa(rate, n),
                    "A/P": ee.ap(rate, n), "F/A": ee.fa(rate, n)},
        "npv_profile": [{"rate": r / 100, "npv": ee.npv(r / 100, flows)} for r in range(0, 101, 5)],
        "depreciation": {"straight_line": ee.straight_line(dep_cost, dep_salv, max(n, 1)),
                         "ddb": ee.declining_balance(dep_cost, dep_salv, max(n, 1))},
    }))


# ---------------------------------------------------------------------------
# AI advisor
# ---------------------------------------------------------------------------
@app.post("/api/advisor/memo")
def advisor_memo():
    b = request.get_json(force=True) or {}
    ctx = advisor.compact_context(b.get("valuation"), b.get("tiers"), b.get("twin"))
    return jsonify(advisor.memo(ctx, b.get("currency", "$")))


@app.post("/api/advisor/ask")
def advisor_ask():
    b = request.get_json(force=True) or {}
    ctx = advisor.compact_context(b.get("valuation"), b.get("tiers"), b.get("twin"))
    return jsonify(advisor.answer(str(b.get("question", ""))[:500], ctx, b.get("currency", "$")))


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
@app.get("/api/scenarios")
def scenarios_list():
    return jsonify(db.list_all())


@app.post("/api/scenarios")
def scenarios_save():
    b = request.get_json(force=True) or {}
    sid = db.save(b.get("name", "Untitled"), b.get("kind", "valuation"), b.get("inputs", {}), b.get("summary", {}))
    return {"id": sid}


@app.delete("/api/scenarios/<int:sid>")
def scenarios_delete(sid):
    db.delete(sid)
    return {"ok": True}


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))
