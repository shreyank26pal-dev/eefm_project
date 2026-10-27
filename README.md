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

## Headline results (see `backend/experiments/results/results.md`)

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
pip install -r backend\requirements-dev.txt
python dataset\generate_invoices.py    # builds the synthetic invoice ledger
cd backend
python -m fdt.ml                        # trains both ML models (~10 s)
python -m pytest -q                     # 9 tests should pass
python app.py                           # open http://127.0.0.1:5000
```

Reproduce all report numbers and figures (from `backend/`):

```powershell
python -m experiments.run_experiments   # ~90 s; writes backend\experiments\results\
```

Optional LLM advisor:

```powershell
$env:GEMINI_API_KEY = "your-key"; python app.py
```

## Deploy (Render, free tier)

1. Push this folder to a GitHub repository.
2. On render.com: **New → Blueprint**, then pick the repo. `render.yaml` installs `backend/requirements.txt`, builds the dataset, trains the models and starts gunicorn with `--chdir backend`.
3. Optionally add `GEMINI_API_KEY` under Environment.

## Project structure

```
eefm/
├── frontend/                   User interface (served by Flask)
│   ├── templates/index.html    Single-page dashboard (8 modules)
│   └── static/
│       ├── app.js              UI logic, charts, API calls
│       ├── style.css           Styling
│       └── vendor/chart.umd.min.js   Chart.js 4.4.1 (bundled)
├── backend/                    Python / Flask API
│   ├── app.py                  Flask routes / REST API
│   ├── fdt/
│   │   ├── economics.py        Engineering-economics engine (TVM, NPV, IRR, EUAC, Eq. 1–9, 21)
│   │   ├── twin.py             Operational-to-financial digital twin simulator
│   │   ├── valuation.py        Overhead-aware Monte-Carlo capital budgeting + tiers
│   │   ├── ml.py               Failure model (AI4I 2020) + DSO model
│   │   ├── advisor.py          LLM / rule-based advisor
│   │   └── db.py               SQLite scenarios
│   ├── models/                 Trained models (generated, not committed)
│   ├── experiments/            Reproducible experiments, results, figures
│   ├── tests/                  Unit tests (survey case-study replication)
│   ├── requirements.txt
│   └── requirements-dev.txt
├── dataset/
│   ├── ai4i2020.csv            AI4I 2020 Predictive Maintenance (UCI #601, CC BY 4.0)
│   ├── generate_invoices.py    Synthetic invoice-ledger generator
│   └── invoices_synthetic.csv  Generated ledger (6,000 invoices)
├── docs/                       Project report (PDF), user manual, screenshots
├── render.yaml                 Render Blueprint (cloud deployment)
└── Procfile
```

## Data credits

S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," *Proc. 3rd Int. Conf. AI for Industries*, 2020 — AI4I 2020 Predictive Maintenance Dataset, UCI ML Repository #601, CC BY 4.0.
