"""
Engineering Economics engine for the FinTwin Financial Digital Twin.

Every function maps to an equation in our CIE-1 survey paper
("Engineering Economics & AI Digital Twins", Group 6) or to a standard
Engineering Economics formula (Time Value of Money, PW/FW/AW, capital
recovery, rate of return, break-even, replacement analysis).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# 1. Time Value of Money - interest factors
# ---------------------------------------------------------------------------
def pf(i: float, n: int) -> float:
    """(P/F, i, n) single-payment present worth factor."""
    return (1 + i) ** -n


def fp(i: float, n: int) -> float:
    """(F/P, i, n) single-payment compound amount factor."""
    return (1 + i) ** n


def pa(i: float, n: int) -> float:
    """(P/A, i, n) uniform-series present worth factor."""
    if i == 0:
        return float(n)
    return (1 - (1 + i) ** -n) / i


def ap(i: float, n: int) -> float:
    """(A/P, i, n) capital recovery factor."""
    return 1.0 / pa(i, n)


def fa(i: float, n: int) -> float:
    """(F/A, i, n) uniform-series compound amount factor."""
    if i == 0:
        return float(n)
    return ((1 + i) ** n - 1) / i


# ---------------------------------------------------------------------------
# 2. Present / Future / Annual Worth, NPV, IRR, Payback  (Survey Eq. 3)
# ---------------------------------------------------------------------------
def npv(rate: float, cashflows: Sequence[float]) -> float:
    """NPV where cashflows[0] is at t=0 (usually -I0). Survey Eq. (3)."""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))


def future_worth(rate: float, cashflows: Sequence[float]) -> float:
    n = len(cashflows) - 1
    return npv(rate, cashflows) * fp(rate, n)


def annual_worth(rate: float, cashflows: Sequence[float]) -> float:
    n = len(cashflows) - 1
    return npv(rate, cashflows) * ap(rate, n)


def irr(cashflows: Sequence[float], lo: float = -0.99, hi: float = 10.0,
        tol: float = 1e-7, max_iter: int = 500) -> float | None:
    """Internal Rate of Return by bisection (robust, no numpy_financial)."""
    f_lo, f_hi = npv(lo, cashflows), npv(hi, cashflows)
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = npv(mid, cashflows)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def simple_payback(cashflows: Sequence[float]) -> float | None:
    """Years until cumulative undiscounted cash flow turns non-negative."""
    cum = 0.0
    for t, cf in enumerate(cashflows):
        prev = cum
        cum += cf
        if t > 0 and cum >= 0:
            return t - 1 + (-prev / cf if cf else 0)
    return None


def discounted_payback(rate: float, cashflows: Sequence[float]) -> float | None:
    disc = [cf / (1 + rate) ** t for t, cf in enumerate(cashflows)]
    return simple_payback(disc)


def profitability_index(rate: float, cashflows: Sequence[float]) -> float:
    i0 = -cashflows[0]
    return (npv(rate, cashflows) + i0) / i0 if i0 else float("nan")


# ---------------------------------------------------------------------------
# 3. Volatility-adjusted discount rate  (Survey Eq. 5)
#    r(sigma) = rf + beta (rm - rf) + gamma * sigma
# ---------------------------------------------------------------------------
def risk_adjusted_rate(rf: float, beta: float, rm: float,
                       gamma: float, sigma: float) -> float:
    return rf + beta * (rm - rf) + gamma * sigma


# ---------------------------------------------------------------------------
# 4. Working capital & Cash Conversion Cycle  (Survey Eq. 1, 11, 12)
# ---------------------------------------------------------------------------
def ccc(dio: float, dso: float, dpo: float) -> float:
    """Cash Conversion Cycle = DIO + DSO - DPO.  Survey Eq. (1)."""
    return dio + dso - dpo


def working_capital_released(annual_revenue: float, delta_ccc_days: float) -> float:
    """Delta WC = Revenue * dCCC / 365.  Survey Eq. (11)."""
    return annual_revenue * delta_ccc_days / 365.0


# ---------------------------------------------------------------------------
# 5. Credit score -> loan rate  (Survey Eq. 8, 9)
# ---------------------------------------------------------------------------
def composite_credit(c_hist: float, c_rt: float, alpha: float = 0.6) -> float:
    """C = alpha*C_hist + (1-alpha)*C_realtime, scores in [0,1]. Eq. (8)."""
    return alpha * c_hist + (1 - alpha) * c_rt


def loan_rate(c_composite: float, r_base: float = 0.08,
              dr_max: float = 0.10, k: float = 3.0) -> float:
    """r_loan = r_base + dr_max * exp(-k*C).  Survey Eq. (9)."""
    return r_base + dr_max * math.exp(-k * c_composite)


# ---------------------------------------------------------------------------
# 6. Replacement analysis  (Survey Eq. 6, 7)
# ---------------------------------------------------------------------------
def euac(i0: float, salvage_n: float, om: Sequence[float], rate: float) -> float:
    """EUAC of keeping an asset n = len(om) years. Survey Eq. (6)."""
    n = len(om)
    pw = i0 - salvage_n * pf(rate, n) + sum(c * pf(rate, t + 1) for t, c in enumerate(om))
    return pw * ap(rate, n)


def economic_life(i0: float, salvage_fn, om_fn, rate: float, max_n: int = 20):
    """Return (best_n, min_euac, table) scanning n = 1..max_n."""
    table = []
    for n in range(1, max_n + 1):
        om = [om_fn(t) for t in range(1, n + 1)]
        table.append({"n": n, "euac": euac(i0, salvage_fn(n), om, rate)})
    best = min(table, key=lambda r: r["euac"])
    return best["n"], best["euac"], table


def marginal_cost_next_year(salvage_now: float, salvage_next: float, rate: float,
                            exp_om_next: float, p_fail: float,
                            c_catastrophe: float) -> float:
    """MC(t+1) = dS + r*S(t) + E[O&M] + P_fail(D)*C_cat.  Survey Eq. (7)."""
    return (salvage_now - salvage_next) + rate * salvage_now + exp_om_next + p_fail * c_catastrophe


# ---------------------------------------------------------------------------
# 7. Break-even analysis
# ---------------------------------------------------------------------------
def breakeven_annual_benefit(i0: float, opex: float, rate: float, n: int) -> float:
    """Gross annual benefit giving NPV = 0.  Survey Eq. (21)."""
    return i0 * ap(rate, n) + opex


def breakeven_units(fixed_cost: float, price: float, var_cost: float) -> float:
    return fixed_cost / (price - var_cost) if price > var_cost else float("inf")


# ---------------------------------------------------------------------------
# 8. Depreciation (used for tax shield / book value of twin hardware)
# ---------------------------------------------------------------------------
def straight_line(cost: float, salvage: float, life: int) -> list[float]:
    d = (cost - salvage) / life
    return [d] * life


def declining_balance(cost: float, salvage: float, life: int, factor: float = 2.0) -> list[float]:
    rate = factor / life
    book, out = cost, []
    for _ in range(life):
        d = min(book * rate, max(book - salvage, 0))
        out.append(d)
        book -= d
    return out


def build_cashflows(i0: float, annual: Iterable[float]) -> list[float]:
    return [-i0, *annual]
