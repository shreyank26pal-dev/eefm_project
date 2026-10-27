"""
AI Advisor - generates an investment memo and answers questions about the twin.

Uses Google Gemini if GEMINI_API_KEY is set, OpenAI if OPENAI_API_KEY is set,
otherwise a deterministic rule-based expert system (so the app always works
offline / on a free cloud tier with no keys).
"""
from __future__ import annotations

import json
import os
import urllib.request

SYSTEM = (
    "You are FinTwin Advisor, an engineering-economics analyst. You explain the "
    "results of a Financial Digital Twin decision-support system to a CFO. "
    "Use ONLY the numbers in the JSON context; never invent figures. Refer to "
    "NPV, IRR, EUAC, CCC, risk-adjusted discount rate r(sigma), probability of "
    "loss and break-even. Be concise (<= 250 words), use short headed sections."
)


def _money(x, cur="$"):
    if x is None:
        return "n/a"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}{cur}{x/1e6:.2f}M"
    if x >= 1e3:
        return f"{sign}{cur}{x/1e3:.1f}k"
    return f"{sign}{cur}{x:.0f}"


def _pct(x):
    return "n/a" if x is None else f"{100*x:.1f}%"


def compact_context(val: dict | None, tiers: dict | None, twin: dict | None) -> dict:
    ctx = {}
    if val:
        ctx["valuation"] = {
            "tier": val["tier"], "capex": val["capex"], "wacc": val["wacc"],
            "sigma": val["sigma"], "r_sigma": val["r_sigma"],
            "static_npv": val["static"]["npv"], "static_irr": val["static"]["irr"],
            "overhead_aware_expected_npv": val["expected_overhead_aware"]["npv"],
            "mc": {k: v for k, v in val["mc"].items() if k not in ("hist", "cf_fan")},
            "overheads_pw": val["overheads_pw"], "breakeven": val["breakeven"],
            "top_sensitivities": val["tornado"][:3],
        }
    if tiers:
        ctx["tiers"] = tiers
    if twin:
        ctx["twin"] = {"summary": twin["summary"], "benefits": twin["benefits_annualised"],
                       "shocks": twin["shocks"]}
    return ctx


# ---------------------------------------------------------------------------
# LLM back-ends
# ---------------------------------------------------------------------------
def _post(url, payload, headers):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _gemini(prompt: str) -> str:
    key = os.environ["GEMINI_API_KEY"]
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    out = _post(url, {"system_instruction": {"parts": [{"text": SYSTEM}]},
                      "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": 0.3}},
                {"Content-Type": "application/json"})
    return out["candidates"][0]["content"]["parts"][0]["text"]


def _openai(prompt: str) -> str:
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    out = _post("https://api.openai.com/v1/chat/completions",
                {"model": model, "temperature": 0.3,
                 "messages": [{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}]},
                {"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    return out["choices"][0]["message"]["content"]


def llm_available() -> str | None:
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def _llm(prompt: str) -> str | None:
    engine = llm_available()
    try:
        if engine == "gemini":
            return _gemini(prompt)
        if engine == "openai":
            return _openai(prompt)
    except Exception as e:  # network/key problems -> fall back silently
        return None
    return None


# ---------------------------------------------------------------------------
# Rule-based expert system (offline fallback, also used for verification)
# ---------------------------------------------------------------------------
def rule_based_memo(ctx: dict, cur: str = "$") -> str:
    lines = []
    v = ctx.get("valuation")
    if v:
        mc = v["mc"]
        verdict = ("INVEST" if mc["npv_mean"] > 0 and mc["prob_loss"] <= 0.10
                   else "INVEST WITH STAGE-GATES" if mc["npv_mean"] > 0
                   else "DO NOT INVEST")
        lines += [
            f"## Recommendation: {verdict} ({v['tier']} twin)",
            "",
            "## Value",
            f"- Static survey-style NPV at WACC {_pct(v['wacc'])}: {_money(v['static_npv'], cur)} "
            f"(IRR {_pct(v['static_irr'])}).",
            f"- Overhead-aware, risk-adjusted NPV (r(sigma) = {_pct(v['r_sigma'])}, sigma = {v['sigma']:.2f}): "
            f"mean {_money(mc['npv_mean'], cur)}, 90% band {_money(mc['npv_p5'], cur)} to {_money(mc['npv_p95'], cur)}.",
            f"- Probability of loss: {_pct(mc['prob_loss'])}; CVaR(5%) {_money(mc['cvar5'], cur)}; "
            f"median IRR {_pct(mc['irr_median'])}; median discounted payback {mc['payback_median']} yr.",
            "",
            "## Why the static figure overstates value",
        ]
        oh = v["overheads_pw"]
        lines.append(f"- PW of hidden overheads: ramp-up loss {_money(oh['ramp_loss'], cur)}, "
                     f"compliance {_money(oh['compliance'], cur)}, hardware refresh {_money(oh['refresh'], cur)}, "
                     f"edge energy {_money(oh['energy'], cur)}.")
        drop = v["static_npv"] - mc["npv_mean"]
        if v["static_npv"]:
            lines.append(f"- Together with risk they remove {_money(drop, cur)} "
                         f"({100*drop/abs(v['static_npv']):.0f}%) of the static NPV.")
        be = v["breakeven"]["benefit_realisation_needed"]
        lines += ["", "## Break-even & sensitivity",
                  f"- The project breaks even if only {_pct(be)} of the projected benefits materialise.",
                  "- Most sensitive drivers: " + ", ".join(t["param"] for t in v["top_sensitivities"]) + "."]
    t = ctx.get("tiers")
    if t:
        rows = {r["tier"]: r for r in t["rows"]}
        rec = rows[t["recommended"]]
        lines += ["", "## Right-sizing the twin",
                  f"- Recommended fidelity tier: **{t['recommended']}** "
                  f"(risk-adjusted NPV {_money(rec['risk_adj_npv_mean'], cur)}, P(loss) {_pct(rec['prob_loss'])})."]
        for name, r in rows.items():
            if r["static_npv"] > 0 and r["risk_adj_npv_mean"] < 0:
                lines.append(f"- Decision reversal: the {name} tier looks acceptable statically "
                             f"({_money(r['static_npv'], cur)}) but destroys value once overheads and risk are priced "
                             f"({_money(r['risk_adj_npv_mean'], cur)}, P(loss) {_pct(r['prob_loss'])}).")
    tw = ctx.get("twin")
    if tw:
        b, w = tw["summary"]["baseline"], tw["summary"]["twin"]
        be = tw["benefits"]
        lines += ["", "## What the live twin measured",
                  f"- CCC {b['avg_ccc']:.1f} -> {w['avg_ccc']:.1f} days; loan rate {_pct(b['avg_loan_rate'])} -> {_pct(w['avg_loan_rate'])}.",
                  f"- Failures {b['failures']} -> {w['failures']}; derates {w.get('derates', 0)}; "
                  f"annualised benefit {_money(be['total'], cur)} "
                  f"(B1 {_money(be['B1_working_capital'], cur)}, B2 {_money(be['B2_credit_spread'], cur)}, "
                  f"B3 {_money(be['B3_failure_avoidance'], cur)})."]
    lines += ["", "## Next step",
              "- Deploy in Phase-1 shadow mode, verify benefits against the twin for one quarter, "
              "then release the smart-contract factoring stage (stage-gated real option)."]
    return "\n".join(lines)


def memo(ctx: dict, cur: str = "$") -> dict:
    base = rule_based_memo(ctx, cur)
    if llm_available():
        prompt = ("Write an investment memo for this Financial Digital Twin.\n"
                  "Context JSON:\n" + json.dumps(ctx, default=float)[:12000] +
                  "\n\nA rule-based draft for reference:\n" + base)
        txt = _llm(prompt)
        if txt:
            return {"engine": llm_available(), "text": txt}
    return {"engine": "rule-based", "text": base}


def answer(question: str, ctx: dict, cur: str = "$") -> dict:
    if llm_available():
        txt = _llm(f"Question: {question}\nContext JSON:\n" + json.dumps(ctx, default=float)[:12000])
        if txt:
            return {"engine": llm_available(), "text": txt}
    q = question.lower()
    v, tw = ctx.get("valuation"), ctx.get("twin")
    if v and any(k in q for k in ("npv", "worth", "invest", "value")):
        mc = v["mc"]
        txt = (f"Static NPV is {_money(v['static_npv'], cur)}; after hidden overheads and risk the mean NPV is "
               f"{_money(mc['npv_mean'], cur)} with a {_pct(mc['prob_loss'])} chance of loss.")
    elif v and ("irr" in q or "return" in q):
        txt = f"Median IRR is {_pct(v['mc']['irr_median'])} (5th percentile {_pct(v['mc']['irr_p5'])}) vs WACC {_pct(v['wacc'])}."
    elif v and ("risk" in q or "sigma" in q or "discount" in q):
        txt = (f"Cash-flow volatility sigma = {v['sigma']:.2f}, so Eq. 5 raises the discount rate from "
               f"{_pct(v['wacc'])} to {_pct(v['r_sigma'])}. CVaR(5%) = {_money(v['mc']['cvar5'], cur)}.")
    elif v and ("break" in q):
        txt = f"Break-even needs only {_pct(v['breakeven']['benefit_realisation_needed'])} of projected benefits."
    elif tw and any(k in q for k in ("ccc", "cash", "working")):
        b, w = tw["summary"]["baseline"], tw["summary"]["twin"]
        txt = f"The twin cut the cash conversion cycle from {b['avg_ccc']:.1f} to {w['avg_ccc']:.1f} days."
    elif tw and any(k in q for k in ("fail", "replace", "maint", "machine")):
        b, w = tw["summary"]["baseline"], tw["summary"]["twin"]
        txt = (f"Condition-based replacement (Eq. 7) gave {w['failures']} failures vs {b['failures']} under the "
               f"time-based policy, saving {_money(tw['benefits']['B3_failure_avoidance'], cur)} per year.")
    else:
        txt = ("I can explain NPV, IRR, risk-adjusted discount rate, break-even, CCC, credit-rate and replacement "
               "results. Run the Valuation and Live Twin tabs first, then ask e.g. 'Is the investment worth it?'. "
               "Set GEMINI_API_KEY for free-form answers.")
    return {"engine": "rule-based", "text": txt}
