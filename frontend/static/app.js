/* FinTwin front-end */
const S = { defaults: null, twin: null, val: null, tiers: null, calib: null, charts: {} };
const C = { twin: "#0f7b6c", base: "#d9822b", neg: "#b4413c", ink: "#4a5663", grid: "#ece8e1", soft: "rgba(15,123,108,.15)" };
Chart.defaults.font.family = '"IBM Plex Sans", system-ui, sans-serif';
Chart.defaults.color = C.ink;
Chart.defaults.plugins.legend.display = false;
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.animation.duration = 400;

// ---------- utilities ----------
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const cur = () => (S.defaults?.profile?.currency) || "$";
function money(x, d = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  const s = x < 0 ? "−" : "", a = Math.abs(x);
  if (a >= 1e6) return `${s}${cur()}${(a / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${s}${cur()}${(a / 1e3).toFixed(d)}k`;
  return `${s}${cur()}${a.toFixed(0)}`;
}
const pct = (x, d = 1) => (x === null || x === undefined) ? "–" : `${(100 * x).toFixed(d)}%`;
const num = (x, d = 1) => (x === null || x === undefined) ? "–" : Number(x).toFixed(d);
const cls = (x) => (x < 0 ? "neg" : "pos");
async function api(path, body, method) {
  const r = await fetch(path, { method: method || (body ? "POST" : "GET"),
    headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(t._h); t._h = setTimeout(() => (t.hidden = true), 2600); }
function formData(form) {
  const o = {};
  $$("input, select, textarea", form).forEach((el) => {
    if (!el.name) return;
    if (el.type === "checkbox") o[el.name] = el.checked;
    else if (el.type === "number") o[el.name] = el.value === "" ? null : Number(el.value);
    else o[el.name] = el.value;
  });
  return o;
}
function fill(form, obj) {
  for (const [k, v] of Object.entries(obj)) {
    const el = form.elements[k];
    if (!el) continue;
    if (el.type === "checkbox") el.checked = !!v; else el.value = v;
  }
}
function chart(id, cfg) {
  if (S.charts[id]) S.charts[id].destroy();
  S.charts[id] = new Chart(document.getElementById(id), cfg);
  return S.charts[id];
}
const gridOpts = (yfmt) => ({
  x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } },
  y: { grid: { color: C.grid }, ticks: { callback: yfmt || ((v) => v) } },
});
function kpis(el, items) {
  el.innerHTML = items.map(([l, v, s, c]) =>
    `<div class="kpi"><div class="k-l">${l}</div><div class="k-v ${c || ""}">${v}</div>${s ? `<div class="k-s">${s}</div>` : ""}</div>`).join("");
}
function table(head, rows, hl) {
  return `<table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${
    rows.map((r, i) => `<tr class="${hl && hl(i) ? "hl" : ""}">${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
function md(text) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  let html = "", inList = false;
  for (const raw of text.split("\n")) {
    let l = esc(raw).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
    if (/^#{1,3}\s/.test(l)) { if (inList) { html += "</ul>"; inList = false; } html += `<h2>${l.replace(/^#+\s/, "")}</h2>`; }
    else if (/^\s*[-*]\s/.test(l)) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${l.replace(/^\s*[-*]\s/, "")}</li>`; }
    else if (l.trim()) { if (inList) { html += "</ul>"; inList = false; } html += `<p>${l}</p>`; }
  }
  return html + (inList ? "</ul>" : "");
}

// ---------- navigation ----------
function show(tab) {
  $$("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${tab}`));
  if (tab === "scenarios") loadScenarios();
  window.scrollTo(0, 0);
}
$$("#nav button").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));

// ---------- live twin ----------
const SHOCKS = { overload: "Machine overload", supplier_delay: "Supplier delay", customer_stress: "Customer credit stress", demand_drop: "Demand drop" };
function addShock(s = { kind: "overload", start: 60, length: 30, magnitude: 1 }) {
  const list = $("#shock-list");
  if (!list.children.length) list.insertAdjacentHTML("beforeend", `<div class="shock-h"><span>type</span><span>start day</span><span>days</span><span>severity</span><span></span></div>`);
  const row = document.createElement("div");
  row.className = "shock";
  row.innerHTML = `<select>${Object.entries(SHOCKS).map(([k, v]) => `<option value="${k}" ${k === s.kind ? "selected" : ""}>${v}</option>`).join("")}</select>
    <input type="number" value="${s.start}" min="0"><input type="number" value="${s.length}" min="1"><input type="number" value="${s.magnitude}" step="0.25" min="0"><button type="button" class="x" title="remove">×</button>`;
  row.querySelector(".x").onclick = () => { row.remove(); if (list.children.length === 1) list.innerHTML = ""; };
  list.appendChild(row);
}
function readShocks() {
  return $$("#shock-list .shock").map((r) => {
    const [k, a, b, c] = $$("select, input", r);
    return { kind: k.value, start: +a.value, length: +b.value, magnitude: +c.value };
  });
}
$("#add-shock").onclick = () => addShock();
$("#preset-shock").onclick = () => {
  $("#shock-list").innerHTML = "";
  addShock({ kind: "overload", start: 60, length: 30, magnitude: 1 });
  addShock({ kind: "supplier_delay", start: 150, length: 40, magnitude: 1 });
  addShock({ kind: "customer_stress", start: 240, length: 60, magnitude: 1 });
};

async function runTwin() {
  const f = formData($("#twin-form"));
  const { days, seed, ...profile } = f;
  $("#twin-status").textContent = "Simulating both worlds …";
  const t0 = performance.now();
  S.twin = await api("/api/twin/simulate", { profile, days, seed, shocks: readShocks() });
  $("#twin-status").textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s · failure model: ${S.twin.model_info.model}`;
  $("#send-benefits").disabled = false;
  renderTwin();
}
function renderTwin() {
  const r = S.twin, b = r.summary.baseline, w = r.summary.twin, ben = r.benefits_annualised;
  kpis($("#twin-kpis"), [
    ["CCC (days)", `${num(b.avg_ccc)} → ${num(w.avg_ccc)}`, `${num(b.avg_ccc - w.avg_ccc)} days released`],
    ["Loan rate", `${pct(b.avg_loan_rate)} → ${pct(w.avg_loan_rate)}`, "Eq. 8–9 credit link"],
    ["Failures", `${b.failures} → ${w.failures}`, `${w.derates} derate-days`],
    ["Annual benefit", money(ben.total), `B1 ${money(ben.B1_working_capital)} · B2 ${money(ben.B2_credit_spread)} · B3 ${money(ben.B3_failure_avoidance)}`, cls(ben.total)],
  ]);
  const days = [...Array(r.days).keys()].map((d) => d + 1);
  const shockBoxes = r.shocks;
  const shade = {
    id: "shade",
    beforeDraw(c) {
      const { ctx, chartArea: a, scales: { x } } = c;
      ctx.save(); ctx.fillStyle = "rgba(180,65,60,.07)";
      shockBoxes.forEach((s) => {
        const x0 = x.getPixelForValue(s.start), x1 = x.getPixelForValue(Math.min(r.days - 1, s.start + s.length));
        ctx.fillRect(x0, a.top, x1 - x0, a.bottom - a.top);
      });
      ctx.restore();
    },
  };
  const line = (label, data, color, dash) => ({ label, data, borderColor: color, borderWidth: 1.6, pointRadius: 0, tension: .25, borderDash: dash || [] });
  const opts = (fmt) => ({ interaction: { mode: "index", intersect: false }, scales: gridOpts(fmt), plugins: { tooltip: { callbacks: {} } } });
  chart("c-ccc", { type: "line", data: { labels: days, datasets: [line("baseline", r.series.baseline.ccc, C.base), line("twin", r.series.twin.ccc, C.twin)] }, options: opts(), plugins: [shade] });
  chart("c-loan", { type: "line", data: { labels: days, datasets: [line("baseline", r.series.baseline.loan_rate, C.base), line("twin", r.series.twin.loan_rate, C.twin)] }, options: opts((v) => pct(v, 1)), plugins: [shade] });
  const cum = (a) => a.reduce((acc, v) => (acc.push((acc.at(-1) || 0) + v), acc), []);
  chart("c-maint", { type: "line", data: { labels: days, datasets: [line("baseline", cum(r.series.baseline.maint_cost), C.base), line("twin", cum(r.series.twin.maint_cost), C.twin)] }, options: opts((v) => money(v, 0)), plugins: [shade] });
  chart("c-pfail", { type: "line", data: { labels: days, datasets: [line("baseline", r.series.baseline.p_fail_max, C.base), line("twin", r.series.twin.p_fail_max, C.twin)] }, options: opts((v) => pct(v, 0)), plugins: [shade] });
  const rows = [
    ["Avg DIO (days)", num(b.avg_dio), num(w.avg_dio)], ["Avg DSO (days, ML)", num(b.avg_dso), num(w.avg_dso)],
    ["Avg DPO (days)", num(b.avg_dpo), num(w.avg_dpo)], ["Avg CCC (days)", num(b.avg_ccc), num(w.avg_ccc)],
    ["Avg working capital", money(b.avg_wc), money(w.avg_wc)], ["Composite credit score", num(b.avg_credit, 3), num(w.avg_credit, 3)],
    ["Avg loan rate", pct(b.avg_loan_rate, 2), pct(w.avg_loan_rate, 2)], ["Failures", b.failures, w.failures],
    ["Planned replacements", b.replacements, w.replacements], ["Derate machine-days", b.derates, w.derates],
    ["Maintenance + failure + derate cost", money(b.maint_cost), money(w.maint_cost)],
  ];
  $("#twin-table").innerHTML = table(["Metric", "Baseline", "Twin"], rows);
  $("#twin-log").innerHTML = r.replacement_log.length ? table(["Day", "Machine", "Wear", "P_fail(t+1)", "MC(t+1)", "EUAC challenger"],
    r.replacement_log.map((d) => [d.day, d.machine, d.wear, pct(d.p_fail_next, 2), money(d.mc_next), money(d.euac_challenger)])) : '<p class="muted">No replacements.</p>';
}
$("#twin-form").addEventListener("submit", (e) => { e.preventDefault(); runTwin().catch((err) => toast(err.message)); });
$("#send-benefits").onclick = () => {
  const b = S.twin.benefits_annualised;
  fill($("#val-form"), { b1_working_capital: Math.round(b.B1_working_capital), b2_credit_spread: Math.round(b.B2_credit_spread), b3_failure_avoidance: Math.max(0, Math.round(b.B3_failure_avoidance)) });
  show("valuation"); toast("Twin-measured B1–B3 loaded. Run Monte Carlo.");
};

// ---------- valuation ----------
async function runValuation() {
  const f = formData($("#val-form"));
  $("#val-status").textContent = `Running ${f.n_sims} simulations …`;
  S.val = await api("/api/valuation", f);
  $("#val-status").textContent = `Done · σ = ${num(S.val.sigma, 3)} · r(σ) = ${pct(S.val.r_sigma, 2)}`;
  renderValuation();
}
function renderValuation() {
  const v = S.val, mc = v.mc, ex = v.expected_overhead_aware;
  kpis($("#val-kpis"), [
    ["Static NPV @ WACC", money(v.static.npv), `IRR ${pct(v.static.irr)} · payback ${num(v.static.payback, 2)} yr`, cls(v.static.npv)],
    ["Risk-adj. mean NPV", money(mc.npv_mean), `90% band ${money(mc.npv_p5)} … ${money(mc.npv_p95)}`, cls(mc.npv_mean)],
    ["P(NPV < 0)", pct(mc.prob_loss), `CVaR 5% ${money(mc.cvar5)}`, mc.prob_loss > 0.1 ? "neg" : ""],
    ["Median IRR", pct(mc.irr_median), `P5 ${pct(mc.irr_p5)}`],
    ["r(σ) vs WACC", `${pct(v.r_sigma)} / ${pct(v.wacc)}`, `σ = ${num(v.sigma, 3)}`],
    ["Break-even realisation", pct(v.breakeven.benefit_realisation_needed), "of projected benefits"],
  ]);
  // histogram
  const e = mc.hist.edges, mids = mc.hist.counts.map((_, i) => (e[i] + e[i + 1]) / 2);
  chart("c-hist", { type: "bar", data: { labels: mids.map((m) => money(m, 0)), datasets: [{ data: mc.hist.counts, backgroundColor: mids.map((m) => (m < 0 ? C.neg : C.twin)), barPercentage: 1, categoryPercentage: 1 }] },
    options: { scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 7 } }, y: { grid: { color: C.grid } } } } });
  // bridge (static -> risk adjusted)
  const oh = v.overheads_pw;
  const steps = [["Static NPV", v.static.npv, "total"], ["Ramp-up", -oh.ramp_loss], ["Refresh", -oh.refresh], ["Compliance", -oh.compliance],
    ["Energy esc.", -(oh.energy - (v.static.flows.length - 1 ? 0 : 0))], ["Growth & shocks", null], ["Risk (r(σ))", null], ["Risk-adj NPV", mc.npv_mean, "total"]];
  // reconcile: everything not explained by named overheads = residual
  const named = steps.slice(1, 5).reduce((s, x) => s + x[1], 0);
  const afterOverhead = ex.npv; // expected overhead-aware NPV @ WACC
  steps[5][1] = afterOverhead - (v.static.npv + named);
  steps[6][1] = mc.npv_mean - afterOverhead;
  let run = 0; const lo = [], hi = [], col = [];
  steps.forEach(([_, val, kind]) => {
    if (kind === "total") { lo.push(0); hi.push(val); run = val; col.push(val < 0 ? C.neg : "#1d3a4b"); }
    else { lo.push(run); hi.push(run + val); run += val; col.push(val < 0 ? C.neg : C.twin); }
  });
  chart("c-bridge", { type: "bar", data: { labels: steps.map((s) => s[0]), datasets: [{ data: lo.map((l, i) => [l, hi[i]]), backgroundColor: col, borderSkipped: false }] },
    options: { scales: gridOpts((x) => money(x, 0)), plugins: { tooltip: { callbacks: { label: (c) => money(c.raw[1] - c.raw[0]) } } } } });
  // fan
  const yrs = mc.cf_fan.p50.map((_, i) => `Y${i + 1}`);
  chart("c-fan", { type: "line", data: { labels: yrs, datasets: [
    { data: mc.cf_fan.p95, borderColor: "transparent", backgroundColor: C.soft, fill: "+1", pointRadius: 0 },
    { data: mc.cf_fan.p5, borderColor: "transparent", pointRadius: 0 },
    { data: mc.cf_fan.p50, borderColor: C.twin, borderWidth: 2, pointRadius: 3 },
    { data: v.static.flows.slice(1), borderColor: C.base, borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0 }] },
    options: { scales: gridOpts((x) => money(x, 0)), plugins: { legend: { display: true, labels: { filter: (i) => i.datasetIndex >= 2, generateLabels: () => [
      { text: "median (MC)", fillStyle: C.twin, strokeStyle: C.twin }, { text: "static survey flow", fillStyle: C.base, strokeStyle: C.base }] } } } } });
  // tornado
  const tor = v.tornado.slice(0, 8);
  chart("c-tornado", { type: "bar", data: { labels: tor.map((t) => t.param.replace(/_/g, " ")), datasets: [
    { data: tor.map((t) => t.low), backgroundColor: C.base }, { data: tor.map((t) => t.high), backgroundColor: C.twin }] },
    options: { indexAxis: "y", scales: { x: { stacked: true, grid: { color: C.grid }, ticks: { callback: (x) => money(x, 0) } }, y: { stacked: true, grid: { display: false } } } } });
  $("#val-table").innerHTML = table(["Measure", "Static (survey)", "Overhead-aware expected", "Monte Carlo"], [
    ["Net present worth", money(v.static.npv), money(ex.npv), `${money(mc.npv_mean)} @ r(σ)`],
    ["Annual worth (A/P)", "–", money(ex.annual_worth), "–"],
    ["Future worth (F/P)", "–", money(ex.future_worth), "–"],
    ["IRR", pct(v.static.irr), pct(ex.irr), `${pct(mc.irr_median)} median`],
    ["Payback (yr)", num(v.static.payback, 2), `${num(ex.discounted_payback, 2)} discounted`, `${num(mc.payback_median, 0)} median disc.`],
    ["Profitability index", "–", num(ex.profitability_index, 2), "–"],
    ["Edge energy", `${num(v.energy.kwh_per_year, 0)} kWh/yr`, `${num(v.energy.co2_tonnes_per_year, 1)} t CO₂/yr`, ""],
  ]);
}
$("#val-form").addEventListener("submit", (e) => { e.preventDefault(); runValuation().catch((err) => toast(err.message)); });
$("#reset-survey").onclick = () => fill($("#val-form"), { b1_working_capital: 184110, b2_credit_spread: 81750, b3_failure_avoidance: 78500, shock_benefit: 150000 });
$("#use-calib").onclick = async () => {
  $("#val-status").textContent = "Calibrating benefits from twin runs (steady + disruption) …";
  S.calib = await api("/api/twin/calibrate", { profile: formData($("#twin-form")) });
  fill($("#val-form"), { b1_working_capital: Math.round(S.calib.b1_working_capital), b2_credit_spread: Math.round(S.calib.b2_credit_spread),
    b3_failure_avoidance: Math.round(S.calib.b3_failure_avoidance), shock_benefit: Math.round(S.calib.shock_benefit) });
  $("#val-status").textContent = "Twin-measured benefits loaded.";
};
$("#save-val").onclick = async () => {
  if (!S.val) return toast("Run a valuation first");
  const name = prompt("Scenario name", `${S.val.tier} · ${new Date().toLocaleString()}`);
  if (!name) return;
  await api("/api/scenarios", { name, kind: "valuation", inputs: formData($("#val-form")),
    summary: { static_npv: S.val.static.npv, npv_mean: S.val.mc.npv_mean, prob_loss: S.val.mc.prob_loss, irr_median: S.val.mc.irr_median, r_sigma: S.val.r_sigma } });
  toast("Scenario saved");
};

// ---------- tiers ----------
async function runTiers() {
  const f = { ...formData($("#val-form")), max_prob_loss: Number($("#max-ploss").value) };
  $("#tier-status").textContent = "Simulating three tiers …";
  S.tiers = await api("/api/tiers", f);
  $("#tier-status").textContent = "";
  renderTiers();
}
function renderTiers() {
  const t = S.tiers, rows = t.rows;
  const rec = rows.find((r) => r.tier === t.recommended);
  const rev = rows.filter((r) => r.static_npv > 0 && r.risk_adj_npv_mean < 0);
  const b = $("#tier-banner"); b.hidden = false;
  b.className = "banner" + (rev.length ? " warn" : "");
  b.innerHTML = `<b>Recommended: ${t.recommended} twin</b> — risk-adjusted NPV ${money(rec.risk_adj_npv_mean)}, P(loss) ${pct(rec.prob_loss)}.` +
    (rev.length ? ` <br><b>Decision reversal:</b> ${rev.map((r) => `${r.tier} has static NPV ${money(r.static_npv)} but risk-adjusted ${money(r.risk_adj_npv_mean)} (P(loss) ${pct(r.prob_loss)})`).join("; ")}.` : "") +
    (t.static_choice !== t.recommended ? ` Static NPV alone would have chosen <b>${t.static_choice}</b>.` : "");
  chart("c-tiers", { type: "bar", data: { labels: rows.map((r) => r.tier), datasets: [
    { label: "static NPV", data: rows.map((r) => r.static_npv), backgroundColor: C.base },
    { label: "risk-adjusted mean NPV", data: rows.map((r) => r.risk_adj_npv_mean), backgroundColor: C.twin },
    { label: "P5 NPV", data: rows.map((r) => r.npv_p5), backgroundColor: "#9bb7c4" }] },
    options: { scales: gridOpts((x) => money(x, 0)), plugins: { legend: { display: true, position: "bottom" } } } });
  chart("c-tier-oh", { type: "bar", data: { labels: rows.map((r) => r.tier), datasets: [
    { label: "capex", data: rows.map((r) => r.capex), backgroundColor: "#1d3a4b" },
    { label: "PW hidden overheads", data: rows.map((r) => r.overhead_pw), backgroundColor: C.neg }] },
    options: { scales: gridOpts((x) => money(x, 0)), plugins: { legend: { display: true, position: "bottom" } } } });
  $("#tier-table").innerHTML = table(["Tier", "Capex", "Static NPV", "Risk-adj NPV", "P5 NPV", "P(loss)", "Median IRR", "r(σ)", "PW overheads", "CO₂ t/yr"],
    rows.map((r) => [`<b>${r.tier}</b><br><span class="muted small">${r.label}</span>`, money(r.capex), money(r.static_npv), money(r.risk_adj_npv_mean), money(r.npv_p5), pct(r.prob_loss), pct(r.irr_median), pct(r.r_sigma), money(r.overhead_pw), num(r.co2_t, 1)]),
    (i) => rows[i].tier === t.recommended);
}
$("#run-tiers").onclick = () => runTiers().catch((err) => toast(err.message));

// ---------- replacement ----------
async function runReplacement() {
  const r = await api("/api/replacement", formData($("#rep-form")));
  const d = r.defender, ch = r.challenger;
  const card = (title, dec, mc, sub) => `<div class="card decision ${dec.startsWith("REPLACE") ? "replace" : ""}"><h3>${title}</h3><div class="big">${dec}</div><div class="muted small">MC(t+1) ${money(mc)} vs min EUAC ${money(ch.min_euac)} · ${sub}</div></div>`;
  $("#rep-decisions").innerHTML = card("With AI twin (Eq. 7)", r.decision, d.mc_next, `P_fail = ${pct(d.p_fail, 2)}`) +
    card("Classical (no failure hazard)", r.decision_without_ai, d.mc_next_without_ai, `keep ≈ ${r.keep_years_deterministic} more yr`);
  chart("c-euac", { type: "line", data: { labels: ch.table.map((x) => x.n), datasets: [{ data: ch.table.map((x) => x.euac), borderColor: C.twin, pointRadius: ch.table.map((x) => (x.n === ch.economic_life ? 6 : 2)), pointBackgroundColor: C.twin }] },
    options: { scales: { x: { title: { display: true, text: "years kept" }, grid: { display: false } }, y: { grid: { color: C.grid }, ticks: { callback: (x) => money(x, 0) } } } } });
  chart("c-mc", { type: "bar", data: { labels: d.curve.map((c) => `+${c.year}`), datasets: [
    { type: "line", data: d.curve.map(() => ch.min_euac), borderColor: C.neg, borderDash: [6, 4], pointRadius: 0 },
    { data: d.curve.map((c, i) => (i === 0 ? d.mc_next : c.mc)), backgroundColor: d.curve.map((c, i) => (i === 0 ? C.base : "#9bb7c4")) }] },
    options: { scales: gridOpts((x) => money(x, 0)) } });
  $("#rep-table").innerHTML = `<p class="small muted">*Year +1 bar includes the AI hazard term P<sub>fail</sub>·C<sub>cat</sub> (orange); later years use the deterministic Eq. 7 terms only. Red dashed line = challenger minimum EUAC at its economic life of ${ch.economic_life} years.</p>` +
    table(["Eq. 7 term", "Value"], [["Salvage now S(t)", money(d.salvage_now)], ["Loss of salvage ΔS", money(d.salvage_now - d.salvage_next)],
      ["Capital cost r·S(t)", money(d.salvage_now * Number($("#rep-form").elements.rate.value))], ["E[O&M(t+1)]", money(d.om_next)],
      ["P_fail · C_cat (AI)", money(d.mc_next - d.mc_next_without_ai)], ["MC(t+1)", money(d.mc_next)], ["Challenger min EUAC", money(ch.min_euac)]]);
}
$("#rep-form").addEventListener("submit", (e) => { e.preventDefault(); runReplacement().catch((err) => toast(err.message)); });

// ---------- TVM ----------
async function runTVM() {
  const f = formData($("#tvm-form"));
  f.cashflows = f.cashflows.split(/[,\s]+/).filter(Boolean).map(Number);
  const r = await api("/api/tvm", f);
  kpis($("#tvm-kpis"), [["Present worth", money(r.pw), "", cls(r.pw)], ["Future worth", money(r.fw)], ["Annual worth", money(r.aw)],
    ["IRR", pct(r.irr)], ["Payback", `${num(r.payback, 2)} yr`, `discounted ${num(r.discounted_payback, 2)} yr`], ["PI", num(r.pi, 2)]]);
  chart("c-npvprof", { type: "line", data: { labels: r.npv_profile.map((p) => pct(p.rate, 0)), datasets: [{ data: r.npv_profile.map((p) => p.npv), borderColor: C.twin, pointRadius: 2 }] },
    options: { scales: gridOpts((x) => money(x, 0)) } });
  const n = f.cashflows.length - 1;
  $("#tvm-table").innerHTML = table(["Factor", `i = ${pct(f.rate, 1)}, n = ${n}`], Object.entries(r.factors).map(([k, v]) => [`(${k})`, num(v, 5)])) +
    "<br>" + table(["Year", "SL depreciation", "DDB depreciation"], r.depreciation.straight_line.map((x, i) => [i + 1, money(x), money(r.depreciation.ddb[i])]));
}
$("#tvm-form").addEventListener("submit", (e) => { e.preventDefault(); runTVM().catch((err) => toast(err.message)); });

// ---------- advisor ----------
$("#gen-memo").onclick = async () => {
  if (!S.val) return toast("Run the valuation (or full analysis) first");
  $("#memo-status").textContent = "Writing memo …";
  const r = await api("/api/advisor/memo", { valuation: S.val, tiers: S.tiers, twin: S.twin });
  $("#memo").innerHTML = md(r.text) + `<p class="muted small">engine: ${r.engine}</p>`;
  $("#memo-status").textContent = "";
};
$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = e.target.q.value.trim(); if (!q) return;
  const log = $("#chat-log");
  log.insertAdjacentHTML("beforeend", `<div class="msg u"></div>`); log.lastChild.textContent = q;
  e.target.q.value = "";
  const r = await api("/api/advisor/ask", { question: q, valuation: S.val, tiers: S.tiers, twin: S.twin });
  log.insertAdjacentHTML("beforeend", `<div class="msg a">${md(r.text)}<div class="eng">${r.engine}</div></div>`);
  log.scrollTop = log.scrollHeight;
});

// ---------- scenarios ----------
async function loadScenarios() {
  const rows = await api("/api/scenarios");
  $("#scen-table").innerHTML = rows.length ? table(["Name", "Saved (UTC)", "Tier", "Static NPV", "Risk-adj NPV", "P(loss)", "r(σ)", ""],
    rows.map((r) => [r.name, r.created_at.replace("T", " "), r.inputs.tier || "–", money(r.summary.static_npv), money(r.summary.npv_mean), pct(r.summary.prob_loss), pct(r.summary.r_sigma),
      `<button class="btn ghost" data-load="${r.id}">Load</button> <button class="btn ghost" data-del="${r.id}">Delete</button>`])) : '<p class="muted">No saved scenarios yet. Save one from the Stochastic NPV tab.</p>';
  $$("[data-del]").forEach((b) => (b.onclick = async () => { await api(`/api/scenarios/${b.dataset.del}`, null, "DELETE"); loadScenarios(); }));
  $$("[data-load]").forEach((b) => (b.onclick = () => { const r = rows.find((x) => x.id == b.dataset.load); fill($("#val-form"), r.inputs); show("valuation"); toast("Inputs loaded"); }));
}

// ---------- overview: full pipeline ----------
$("#run-all").onclick = async () => {
  const st = $("#run-all-status");
  try {
    st.textContent = "1/3 running live twin under stress preset …";
    if (!readShocks().length) $("#preset-shock").click();
    await runTwin();
    st.textContent = "2/3 Monte Carlo valuation …";
    await runValuation();
    st.textContent = "3/3 comparing twin tiers …";
    await runTiers();
    $("#ov-static").textContent = money(S.val.static.npv);
    $("#ov-risk").textContent = money(S.val.mc.npv_mean);
    $("#ov-risk").className = "k-v " + cls(S.val.mc.npv_mean);
    $("#ov-ploss").textContent = pct(S.val.mc.prob_loss);
    $("#ov-tier").textContent = S.tiers.recommended;
    st.textContent = "Done. Explore each module, or open the AI Advisor for a memo.";
  } catch (err) { st.textContent = err.message; }
};

// ---------- init ----------
(async function init() {
  S.defaults = await api("/api/defaults");
  fill($("#twin-form"), S.defaults.profile);
  fill($("#val-form"), S.defaults.valuation);
  const h = await api("/api/health");
  $("#llm-status").textContent = `advisor: ${h.llm}`;
  runReplacement(); runTVM();
})();
