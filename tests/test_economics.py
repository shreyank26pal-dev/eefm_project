"""Unit tests: the engine must reproduce textbook factors and the CIE-1 survey case study.
Run:  python -m pytest -q
"""
import math

import pytest

from fdt import economics as ee
from fdt.valuation import ValuationInput, compare_tiers, deterministic_flows, monte_carlo


def test_interest_factors_match_tables():
    assert ee.pa(0.10, 5) == pytest.approx(3.7908, abs=1e-4)     # (P/A,10%,5)
    assert ee.ap(0.10, 5) == pytest.approx(0.26380, abs=1e-5)    # (A/P,10%,5)
    assert ee.pf(0.10, 5) == pytest.approx(0.62092, abs=1e-5)
    assert ee.fa(0.10, 5) == pytest.approx(6.1051, abs=1e-4)


def test_npv_irr_payback_basic():
    flows = [-1000, 300, 300, 300, 300, 300]
    assert ee.npv(0.10, flows) == pytest.approx(137.24, abs=0.01)
    assert ee.irr(flows) == pytest.approx(0.15238, abs=1e-4)
    assert ee.simple_payback(flows) == pytest.approx(3.3333, abs=1e-3)


def test_survey_case_study_reproduced():
    """Survey Sec. 5: NPV = $684,814, IRR ~ 60.3%, payback 1.50 yr."""
    vi = ValuationInput(include_overheads=False)
    assert vi.wacc() == pytest.approx(0.10)
    flows, _ = deterministic_flows(vi)
    assert flows[1] == pytest.approx(299_360, rel=1e-3)
    assert ee.npv(0.10, flows) == pytest.approx(684_814, rel=1e-3)
    assert ee.irr(flows) == pytest.approx(0.603, abs=0.002)
    assert ee.simple_payback(flows) == pytest.approx(1.50, abs=0.01)


def test_survey_breakeven_eq21():
    assert ee.breakeven_annual_benefit(450_000, 45_000, 0.10, 5) == pytest.approx(163_708, rel=1e-3)


def test_ccc_and_working_capital_eq11():
    assert ee.ccc(48, 42, 35) == 55
    assert ee.working_capital_released(35_000_000, 32) == pytest.approx(3_068_493, rel=1e-6)


def test_loan_rate_decreases_with_credit_eq9():
    assert ee.loan_rate(0.9) < ee.loan_rate(0.3)
    assert ee.loan_rate(0.0, r_base=0.08, dr_max=0.10) == pytest.approx(0.18)


def test_euac_and_economic_life():
    n, val, table = ee.economic_life(10_000, lambda n: 10_000 * 0.8 ** n,
                                     lambda t: 500 + 400 * t, 0.10, 15)
    assert 1 <= n <= 15 and val == min(r["euac"] for r in table)


def test_depreciation_sums():
    assert sum(ee.straight_line(10_000, 1_000, 5)) == pytest.approx(9_000)
    assert sum(ee.declining_balance(10_000, 1_000, 5)) <= 9_000 + 1e-6


def test_overheads_lower_value_and_tier_reversal():
    static = monte_carlo(ValuationInput(include_overheads=False, n_sims=2000))
    aware = monte_carlo(ValuationInput(include_overheads=True, n_sims=2000))
    assert aware["mc"]["npv_mean"] < static["static"]["npv"]
    cmp_ = compare_tiers(ValuationInput(n_sims=2000))
    high = next(r for r in cmp_["rows"] if r["tier"] == "high")
    assert high["static_npv"] > 0 and high["risk_adj_npv_mean"] < 0   # decision reversal
    assert cmp_["recommended"] != "high"
