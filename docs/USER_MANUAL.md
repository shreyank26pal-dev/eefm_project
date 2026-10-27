# FinTwin — User Manual

## 1. Getting started

Open the deployed URL (https://fintwin-0ae4.onrender.com), or run `python app.py` from the `backend` folder and open http://127.0.0.1:5000. The left sidebar lists eight modules. The quickest tour is **Overview → Run full analysis**. That button runs the Live Twin under a stress preset, then a 5,000-run Monte Carlo valuation, then the tier comparison, and fills in the headline KPIs.

All money values are in $ to match the CIE-1 case study.

## 2. Live Twin

**Purpose:** see how operational events become financial outcomes, with and without a twin.

1. **Business profile:**
   - revenue, DIO / DSO / DPO
   - target DIO with the twin
   - historical credit score (0–1)
   - number of machines
   - time-based replacement interval (baseline policy)
   - failure cost and planned replacement cost
2. **Disruption shocks:** click *+ Add shock* or *Load stress preset*. Each shock has a type, a start day, a length and a severity (1 = nominal):
   - *Machine overload* raises torque and lowers RPM, which increases failure probability.
   - *Supplier delay* builds up inventory, which increases DIO.
   - *Customer credit stress* weakens payer credit, which slows collections.
   - *Demand drop* cuts revenue by 25% × severity.
3. Click **Run twin**. You get:
   - **KPI tiles:** CCC, loan rate, failures, and the annualised benefit split into B1 (working capital), B2 (credit spread) and B3 (failure avoidance).
   - **Charts:** shock periods are shaded red. The cumulative cost chart shows where the twin saves money.
   - **Replacement log:** each Eq. 7 decision, with P<sub>fail</sub>(t+1), MC(t+1) and the challenger EUAC.
4. **Use measured benefits in NPV →** copies B1–B3 into the valuation form.

## 3. Stochastic NPV

1. Choose a **twin tier** and a horizon.
2. Enter the benefits. *Survey assumptions* loads the CIE-1 values. *Twin-measured* runs the twin with and without disruption and loads the measured values, including the extra benefit in a shock year.
3. Set the discount-rate inputs. WACC = r<sub>f</sub> + β(r<sub>m</sub> − r<sub>f</sub>). γ is the premium per unit of cash-flow volatility σ (Eq. 5).
4. Set the uncertainty and overhead inputs:
   - a triangular distribution for benefit realisation
   - energy price
   - year-1 ramp-up
   - compliance cost
   - number of simulations
5. Untick *Include hidden twin overheads* to reproduce the survey's static method.
6. Click **Run Monte Carlo**.

How to read the results:

| Output | Meaning |
|---|---|
| Static NPV @ WACC | Survey method, deterministic |
| Risk-adj. mean NPV | Mean of the simulations discounted at r(σ) |
| P(NPV<0), CVaR 5% | Probability of loss; average NPV in the worst 5% of runs |
| Break-even realisation | Share of the projected benefits needed for NPV = 0 |
| Bridge chart | How each overhead and the risk adjustment reduce the static NPV |
| Tornado | NPV change for ±20% in each input |

**Save scenario** stores the inputs and results in the database (see Scenarios).

## 4. Right-size Twin

Set *Max P(loss)* (default 10%) and click **Compare tiers**. FinTwin recommends the tier with the highest risk-adjusted NPV whose loss probability is within the limit. If no tier meets the limit, it recommends the least-bad tier. The banner flags any **decision reversal**: a tier with a positive static NPV but a negative risk-adjusted NPV. The comparison uses the benefit values currently in the Stochastic NPV form.

## 5. Replacement

Enter the challenger's cost, salvage decline and O&M; the defender's age, O&M and catastrophe cost; and the current telemetry (machine type, temperatures, RPM, torque, tool wear). Click **Evaluate**. The two cards compare the AI decision (Eq. 7 including P<sub>fail</sub>·C<sub>cat</sub>) with the classical decision without the hazard term. Try raising tool wear from 150 to 220 to watch the AI decision flip.

## 6. TVM Toolkit

Type comma-separated cash flows starting at t = 0, plus an interest rate. You get:
- PW, FW, AW, IRR, payback, discounted payback and PI
- the interest factors (P/F, F/P, P/A, A/P, F/A)
- the NPV profile
- straight-line and double-declining-balance depreciation

## 7. AI Advisor

Run the analysis first, then click **Generate memo**. Ask follow-up questions in the chat. Without an API key, the rule-based engine answers questions about NPV, IRR, risk / r(σ), break-even, CCC and failures. To enable Gemini, set the environment variable `GEMINI_API_KEY` (or `OPENAI_API_KEY`). The footer of each answer shows which engine produced it.

## 8. Scenarios

Lists the saved valuations. **Load** copies a scenario's inputs back into the valuation form; **Delete** removes it. On Render's free tier the database resets on every redeploy.

## 9. REST API (for testing / viva)

| Method | Endpoint | Body |
|---|---|---|
| GET | `/api/health` | – |
| GET | `/api/defaults` | – |
| POST | `/api/twin/simulate` | `{profile:{…}, days, seed, shocks:[{kind,start,length,magnitude}]}` |
| POST | `/api/twin/calibrate` | `{profile:{…}}` |
| POST | `/api/valuation` | ValuationInput fields |
| POST | `/api/tiers` | ValuationInput fields + `max_prob_loss` |
| POST | `/api/replacement` | defender / challenger / telemetry fields |
| POST | `/api/tvm` | `{cashflows:[…], rate}` |
| POST | `/api/advisor/memo`, `/api/advisor/ask` | `{valuation, tiers, twin, question}` |
| GET / POST / DELETE | `/api/scenarios[/<id>]` | `{name, inputs, summary}` |

## 10. Troubleshooting

* **First request is slow:** the models are being trained. Run `python -m fdt.ml` once beforehand, from the `backend` folder.
* **"running scripts is disabled" in PowerShell:** run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
* **Render free instance sleeps:** the first visit after a period of inactivity takes about 30–50 s to wake.
