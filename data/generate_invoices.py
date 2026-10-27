"""
Generate a synthetic B2B invoice-settlement dataset used to train the
DSO (Days Sales Outstanding) predictor of the Financial Digital Twin.

Why synthetic?  Real receivables ledgers are confidential (this is exactly
Research Gap 4 of our survey).  The generator encodes published stylised
facts: late payment grows with weak credit, large invoice size and long
terms, and falls when delivery is digitally verified (Zhang et al. [A11],
Vaghani et al. [A9]).

Run:  python data/generate_invoices.py
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 6000

industries = ["automotive", "electronics", "pharma", "fmcg", "construction"]
ind_delay = {"automotive": 4, "electronics": 2, "pharma": 1, "fmcg": 0, "construction": 9}

df = pd.DataFrame({
    "customer_credit": RNG.beta(5, 2, N).round(3),            # 0..1
    "invoice_amount": RNG.lognormal(mean=10.5, sigma=0.9, size=N).round(0),
    "terms_days": RNG.choice([30, 45, 60], N, p=[0.5, 0.3, 0.2]),
    "industry": RNG.choice(industries, N),
    "past_avg_delay": np.clip(RNG.gamma(2.0, 4.0, N), 0, 60).round(1),
    "twin_verified_delivery": RNG.choice([0, 1], N, p=[0.55, 0.45]),
    "quarter_end": RNG.choice([0, 1], N, p=[0.75, 0.25]),
})

delay = (
    0.55 * df["past_avg_delay"]
    + 18 * (1 - df["customer_credit"]) ** 1.5
    + 2.2 * np.log(df["invoice_amount"] / 20000).clip(lower=0)
    + 0.08 * (df["terms_days"] - 30)
    + df["industry"].map(ind_delay)
    - 6.5 * df["twin_verified_delivery"]        # dispute-free, auto-factored
    - 2.0 * df["quarter_end"]
    + RNG.normal(0, 3.5, N)
)
df["days_to_pay"] = (df["terms_days"] + delay.clip(lower=-10)).round(0).astype(int)

if __name__ == "__main__":
    out = __file__.replace("generate_invoices.py", "invoices_synthetic.csv")
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows -> {out}")
    print(df.describe().T[["mean", "std", "min", "max"]])
