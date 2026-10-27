# FinTwin — Financial Digital Twin for Business

**CIE Activity 2 Mini Project · Engineering Economics and Financial Decision Making · Group 6**
Sharmad Kulkarni (21338) · Shreyank Pal (21369) · Parth Dharap (21321) · Kartik Agrawal (21333) · Deepak Dafne (21318)

FinTwin is a web-based, AI-enabled decision-support system. It mirrors a manufacturing SME day by day: its machines, inventory, invoices and credit. It turns that physical state into financial state, and then answers the question our CIE-1 survey left open:

> *Is a Financial Digital Twin worth building, and how much of one?*

## Continuity with the CIE-1 survey

| Survey element | What FinTwin does with it |
|---|---|
| Sec. 5 deterministic case study (NPV $684,814, IRR 60.3%) | Reproduced exactly (unit-tested), then re-evaluated under risk |
| Eq. 1, 11, 12 — CCC & working capital | Computed live from simulated inventory, receivables and payables |
| Eq. 2 — invoice settlement latency τ | ML regressor predicts days-to-pay per invoice |
| Eq. 4–5 — stochastic NPV, volatility-adjusted rate r(σ) | Monte-Carlo NPV/IRR discounted at r(σ) = r<sub>f</sub> + β(r<sub>m</sub>−r<sub>f</sub>) + γσ |
| Eq. 6–7 — EUAC & condition-based marginal cost | Challenger economic life + defender MC(t+1) with ML P<sub>fail</sub> |
| Eq. 8–9 — composite credit score → loan rate | Real-time credit score from operational telemetry |
| **Gap 2** — computational/energy overhead vs value | Hidden overheads priced; Lite / Standard / High-fidelity twin compared |
| **Gap 4** — validation only in simulation | Failure model trained on the real **AI4I 2020** dataset; benefits *measured* by the twin rather than assumed |

## Headline results (see `experiments/results/results.md`)

* The survey's static case reproduces to within $67 (NPV $684,881).
* With hidden overheads and risk priced in, the mean NPV falls to **$375k** (−45%). The discount rate rises from 10.0% to 13.5%.
* **Decision reversal:** the high-fidelity twin has static NPV **+$243k** but risk-adjusted NPV **−$273k**, with a 92.5% probability of loss.
* The twin-measured benefits ($183k/yr in steady state) are about half of the survey's assumed $344k/yr. Most of the twin's value appears in disruption years: failures drop from 68 to 39 under shocks. With measured benefits only the **Lite** twin keeps a positive expected NPV.
* Failure model (HistGradientBoosting, AI4I): ROC-AUC 0.963. Its cost-optimal alarm policy saves 80.8% vs run-to-failure.

## Modules

1. **Live Twin** — two-world simulation (baseline vs twin) with shock injection (overload, supplier delay, customer stress, demand drop).
2. **Stochastic NPV** — Monte Carlo, r(σ), NPV histogram, static-to-risk bridge, cash-flow fan, tornado, break-even.
3. **Right-size Twin** — tier selection with a loss-probability constraint.
4. **Replacement** — Eq. 6 EUAC curve and Eq. 7 decision, with and without the AI hazard term.
5. **TVM Toolkit** — PW / FW / AW, IRR, payback, interest factors, NPV profile, SL/DDB depreciation.
6. **AI Advisor** — investment memo and Q&A. Uses Gemini or OpenAI if a key is set, otherwise a rule-based expert system.
7. **Scenarios** — save and reload valuations (SQLite).

## Tech stack

Python 3.12 · Flask · NumPy · pandas · scikit-learn · SQLite · Chart.js (vendored) · Gunicorn · Render

## Run locally (Windows, VS Code terminal)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements-dev.txt
python data\generate_invoices.py       # builds the synthetic invoice ledger
python -m fdt.ml                        # trains both ML models (~10 s)
python -m pytest -q                     # 9 tests should pass
python app.py                           # open http://127.0.0.1:5000
```

Reproduce all report numbers and figures:

```powershell
python -m experiments.run_experiments   # ~90 s; writes experiments/results/
```

Optional LLM advisor:

```powershell
$env:GEMINI_API_KEY = "your-key"; python app.py
```

## Deploy (Render, free tier)

1. Push this folder to a GitHub repository.
2. On render.com: **New → Blueprint**, then pick the repo. `render.yaml` sets the build, which installs packages, builds the dataset and trains the models, and the start command (gunicorn).
3. Optionally add `GEMINI_API_KEY` under Environment.

## Project structure

```
app.py                  Flask routes / REST API
fdt/economics.py        Engineering-economics engine (TVM, NPV, IRR, EUAC, Eq. 1–9, 21)
fdt/twin.py             Operational-to-financial digital twin simulator
fdt/valuation.py        Overhead-aware Monte-Carlo capital budgeting + tiers
fdt/ml.py               Failure model (AI4I 2020) + DSO model
fdt/advisor.py          LLM / rule-based advisor
fdt/db.py               SQLite scenarios
data/                   ai4i2020.csv (UCI, CC BY 4.0), invoice generator
experiments/            Reproducible experiments, results, figures
tests/                  Unit tests (survey case study replication)
templates/, static/     Front end
docs/                   Project report, user manual
```

## Data credits

S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," *Proc. 3rd Int. Conf. AI for Industries*, 2020 — AI4I 2020 Predictive Maintenance Dataset, UCI ML Repository #601, CC BY 4.0.
