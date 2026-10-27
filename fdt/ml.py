"""
AI / ML layer of FinTwin.

Model 1 - Failure-probability model  P_fail(D_t)  used in Survey Eq. (7)
          Trained on the REAL AI4I 2020 Predictive Maintenance dataset
          (UCI ML Repository, CC BY 4.0, 10,000 machine records).
          The alarm threshold is chosen by *expected economic cost*,
          not accuracy (cost-sensitive learning).

Model 2 - Invoice settlement-delay regressor  tau_{j,t}  used in Survey Eq. (2)
          Trained on a synthetic receivables ledger (data/generate_invoices.py)
          to forecast DSO and quantify the twin's DSO compression.

Run  `python -m fdt.ml`  to (re)train and write models/ + experiments/results/.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "experiments", "results")

FAIL_FEATURES = ["type", "air_temp", "proc_temp", "rpm", "torque", "tool_wear",
                 "power_kw", "temp_delta", "wear_torque"]


# ---------------------------------------------------------------------------
# Feature engineering (shared by training and live twin inference)
# ---------------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["power_kw"] = out["torque"] * out["rpm"] * 2 * np.pi / 60 / 1000
    out["temp_delta"] = out["proc_temp"] - out["air_temp"]
    out["wear_torque"] = out["tool_wear"] * out["torque"]
    return out


def load_ai4i() -> pd.DataFrame:
    raw = pd.read_csv(os.path.join(DATA, "ai4i2020.csv"), encoding="utf-8-sig")
    df = pd.DataFrame({
        "type": raw["Type"],
        "air_temp": raw["Air temperature [K]"],
        "proc_temp": raw["Process temperature [K]"],
        "rpm": raw["Rotational speed [rpm]"],
        "torque": raw["Torque [Nm]"],
        "tool_wear": raw["Tool wear [min]"],
        "failure": raw["Machine failure"],
    })
    return engineer(df)


def _fail_preprocessor():
    num = [c for c in FAIL_FEATURES if c != "type"]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["type"]),
    ])


def optimal_threshold(y_true, p, c_fn: float, c_fp: float):
    """Pick the alarm threshold that minimises total expected cost.
    c_fn = cost of an unpredicted failure (catastrophe + downtime)
    c_fp = cost of an unnecessary preventive intervention."""
    best = (0.5, float("inf"))
    for t in np.linspace(0.01, 0.95, 95):
        pred = p >= t
        fn = np.sum((~pred) & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        cost = fn * c_fn + fp * c_fp
        if cost < best[1]:
            best = (float(t), float(cost))
    return best


def train_failure_model(c_fn: float = 25000.0, c_fp: float = 1500.0, seed: int = 7):
    df = load_ai4i()
    X, y = df[FAIL_FEATURES], df["failure"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                               class_weight="balanced_subsample",
                                               random_state=seed, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                               random_state=seed),
    }
    report, fitted = {}, {}
    n_test_fail = int(yte.sum())
    for name, est in candidates.items():
        pipe = Pipeline([("prep", _fail_preprocessor()),
                         ("clf", CalibratedClassifierCV(est, method="isotonic", cv=3))])
        # threshold tuned on out-of-fold TRAIN predictions (no test leakage)
        p_oof = cross_val_predict(pipe, Xtr, ytr, cv=3, method="predict_proba")[:, 1]
        thr, _ = optimal_threshold(ytr, p_oof, c_fn, c_fp)
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        pred = p >= thr
        cost = float(np.sum((~pred) & (yte == 1)) * c_fn + np.sum(pred & (yte == 0)) * c_fp)
        # baseline policies for economic comparison on the same test set
        run_to_failure = n_test_fail * c_fn
        report[name] = {
            "roc_auc": round(roc_auc_score(yte, p), 4),
            "pr_auc": round(average_precision_score(yte, p), 4),
            "f1_at_0.5": round(f1_score(yte, p >= 0.5), 4),
            "cost_opt_threshold": round(thr, 3),
            "f1_at_cost_threshold": round(f1_score(yte, p >= thr), 4),
            "recall_at_cost_threshold": round(float(np.mean(pred[yte == 1])), 4),
            "alarms_raised": int(pred.sum()),
            "expected_cost": round(cost, 0),
            "run_to_failure_cost": round(run_to_failure, 0),
            "saving_vs_run_to_failure_pct": round(100 * (1 - cost / run_to_failure), 2),
        }
        fitted[name] = (pipe, thr)

    best = max(report, key=lambda k: report[k]["saving_vs_run_to_failure_pct"])
    pipe, thr = fitted[best]
    import sklearn
    meta = {"model": best, "threshold": thr, "sklearn_version": sklearn.__version__, "c_fn": c_fn, "c_fp": c_fp,
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "test_failures": n_test_fail, "dataset": "AI4I 2020 (UCI #601)"}
    os.makedirs(MODELS, exist_ok=True)
    joblib.dump({"pipeline": pipe, "meta": meta}, os.path.join(MODELS, "failure_model.joblib"))
    return report, meta


# ---------------------------------------------------------------------------
# Model 2: invoice delay / DSO
# ---------------------------------------------------------------------------
INV_NUM = ["customer_credit", "invoice_amount", "terms_days", "past_avg_delay",
           "twin_verified_delivery", "quarter_end"]
INV_CAT = ["industry"]


def train_dso_model(seed: int = 7):
    path = os.path.join(DATA, "invoices_synthetic.csv")
    if not os.path.exists(path):
        import runpy
        runpy.run_path(os.path.join(DATA, "generate_invoices.py"), run_name="__main__")
    df = pd.read_csv(path)
    X, y = df[INV_NUM + INV_CAT], df["days_to_pay"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    prep = ColumnTransformer([("num", StandardScaler(), INV_NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), INV_CAT)])

    naive = Xte["terms_days"] + Xte["past_avg_delay"]      # what a static ERP assumes
    lin = Pipeline([("prep", prep), ("reg", LinearRegression())]).fit(Xtr, ytr)
    gbr = Pipeline([("prep", prep), ("reg", GradientBoostingRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.05, random_state=seed))]).fit(Xtr, ytr)

    report = {
        "Naive (terms + past delay)": {"mae_days": round(mean_absolute_error(yte, naive), 3),
                                       "r2": round(r2_score(yte, naive), 4)},
        "LinearRegression": {"mae_days": round(mean_absolute_error(yte, lin.predict(Xte)), 3),
                             "r2": round(r2_score(yte, lin.predict(Xte)), 4)},
        "GradientBoosting": {"mae_days": round(mean_absolute_error(yte, gbr.predict(Xte)), 3),
                             "r2": round(r2_score(yte, gbr.predict(Xte)), 4)},
    }
    # counterfactual: effect of twin-verified delivery on predicted DSO
    cf0, cf1 = Xte.copy(), Xte.copy()
    cf0["twin_verified_delivery"], cf1["twin_verified_delivery"] = 0, 1
    report["twin_effect_days"] = round(float(np.mean(gbr.predict(cf0) - gbr.predict(cf1))), 2)
    import sklearn
    os.makedirs(MODELS, exist_ok=True)
    joblib.dump({"pipeline": gbr, "meta": {"model": "GradientBoostingRegressor",
                                           "sklearn_version": sklearn.__version__}},
                os.path.join(MODELS, "dso_model.joblib"))
    return report


# ---------------------------------------------------------------------------
# Inference helpers used by the live twin
# ---------------------------------------------------------------------------
def _load_or_train(name, trainer):
    """Load a saved model; retrain if missing or saved with another scikit-learn."""
    import sklearn
    path = os.path.join(MODELS, name)
    try:
        b = joblib.load(path)
        if b["meta"].get("sklearn_version", sklearn.__version__) == sklearn.__version__:
            return b
    except Exception:
        pass
    os.makedirs(MODELS, exist_ok=True)
    trainer()
    return joblib.load(path)


@lru_cache(maxsize=1)
def _failure_bundle():
    return _load_or_train("failure_model.joblib", train_failure_model)


@lru_cache(maxsize=1)
def _dso_bundle():
    return _load_or_train("dso_model.joblib", train_dso_model)


def predict_failure_proba(rows) -> np.ndarray:
    """rows: list of dicts or dict of arrays with raw telemetry columns."""
    b = _failure_bundle()
    df = engineer(pd.DataFrame(rows))
    return b["pipeline"].predict_proba(df[FAIL_FEATURES])[:, 1]


def failure_meta() -> dict:
    return _failure_bundle()["meta"]


def predict_days_to_pay(rows) -> np.ndarray:
    return _dso_bundle()["pipeline"].predict(pd.DataFrame(rows)[INV_NUM + INV_CAT])


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    fr, meta = train_failure_model()
    dr = train_dso_model()
    out = {"failure_model": fr, "failure_selected": meta, "dso_model": dr}
    with open(os.path.join(RESULTS, "ml_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
