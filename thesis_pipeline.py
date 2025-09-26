#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline (FINAL)
Multi-asset forecasting using: Yahoo Finance + Google Trends + Technical Indicators
Models: LSTM / GRU / Transformer (TF 2.10) + ARIMA baseline
Validation: fixed split + optional walk-forward
Evaluation: statistical (RMSE, MAE, R², directional accuracy) and financial (Sharpe, cumulative return, max drawdown, Buy&Hold)
Significance: paired t-test as a simplified Diebold–Mariano check
Reproducibility: run metadata + saved keywords + saved predictions + model artifacts (.h5)
Explainability: permutation importance on validation

Outputs
- Figures:            outputs/figs
- Per-ticker JSON:    outputs/results
- Master CSV:         outputs/ALL_MODELS_SUMMARY.csv
- Predictions CSVs:   outputs/preds
- Saved Keras models: outputs/models  (HDF5 .h5)
- Run metadata:       outputs/run_metadata.json
- Keywords table:     outputs/KEYWORDS_USED.csv
"""

# ======================= Imports & Setup =======================
import os, json, math, time, warnings, sys, platform
from itertools import product
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import yfinance as yf
from pytrends.request import TrendReq

from scipy.stats import ttest_rel, pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ======================= Configuration =========================
START_DATE = "2018-01-01"
END_DATE   = "2025-08-31"   # YYYY-MM-DD

OUTPUT_DIR = "outputs"
FIG_DIR    = os.path.join(OUTPUT_DIR, "figs")
RES_DIR    = os.path.join(OUTPUT_DIR, "results")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")
PRED_DIR   = os.path.join(OUTPUT_DIR, "preds")
for d in (FIG_DIR, RES_DIR, MODEL_DIR, PRED_DIR):
    os.makedirs(d, exist_ok=True)

# Assets (equities + commodity + crypto)
TICKERS = [
    "^GSPC",   # S&P 500
    "NVDA",
    "BLK",
    "SAP",     # or "SAP.DE" for XETRA
    "LMT",     # Lockheed Martin (defense)
    "GC=F",    # Gold futures
    "BTC-USD"  # Bitcoin
]

# Google Trends keywords
KEYWORDS_GLOBAL = {
    "^GSPC":   {"kw": ["stock market"],                                   "geo": ""},
    "NVDA":    {"kw": ["Nvidia", "NVDA stock"],                           "geo": ""},
    "BLK":     {"kw": ["BlackRock"],                                      "geo": ""},
    "SAP":     {"kw": ["SAP stock", "SAP SE"],                            "geo": ""},
    "LMT":     {"kw": ["Lockheed Martin", "F-35", "military jets", "war"],"geo": ""},
    "GC=F":    {"kw": ["gold", "gold price", "buy gold", "safe haven"],   "geo": ""},
    "BTC-USD": {"kw": ["Bitcoin", "BTC price", "buy Bitcoin", "crypto"],  "geo": ""}
}
KEYWORDS_LOCAL = {
    "^GSPC":   {"kw": ["stock market"],           "geo": "US"},
    "NVDA":    {"kw": ["Nvidia", "NVDA"],         "geo": "US"},
    "BLK":     {"kw": ["BlackRock"],              "geo": "US"},
    "SAP":     {"kw": ["SAP"],                    "geo": "DE"},
    "LMT":     {"kw": ["Lockheed Martin", "F-35"],"geo": "US"},
    "GC=F":    {"kw": ["gold price", "buy gold"], "geo": "US"},
    "BTC-USD": {"kw": ["Bitcoin", "BTC"],         "geo": "US"}
}
USE_LOCAL = False
KEYWORDS  = KEYWORDS_LOCAL if USE_LOCAL else KEYWORDS_GLOBAL

# Lags & training
GT_LAGS      = [0, 1, 3, 7, 14]
LOOKBACK     = 60
EPOCHS       = 20
BATCH_SIZE   = 32
VAL_SPLIT    = 0.1
USE_ROLLING  = True
ROLL_STEP    = 20

# Trading frictions
TRANSACTION_COST_BPS = 0.0005  # 5 bps per entry/exit

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Technical indicators
USE_TECH_FEATURES = True
TECH_FEATURES = ["ret_1", "ma_5", "ma_20", "vol_10", "rsi_14", "vol_z"]

# Hyperparameter grid (compact)
HP_GRID = {
    "model":  ["LSTM", "GRU", "TRANSFORMER"],
    "units1": [32, 64],
    "units2": [16, 32],
    "dropout":[0.1, 0.2],
}
HP_MAX_TRIALS = 6

# Explainability (permutation importance on validation)
DO_PERM_IMPORTANCE   = True
PERM_FEATURE_LIMIT   = 16

# ======================= Helpers ===============================
def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

def save_json(obj, name):
    path = os.path.join(RES_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2, ensure_ascii=False)
    print(f"[Saved] JSON -> {path}")

def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"[Saved] Figure -> {path}")

def write_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def normalize_minmax(df, cols):
    scalers = {}
    out = df.copy()
    for c in cols:
        sc = MinMaxScaler()
        out[c + "_norm"] = sc.fit_transform(out[[c]])
        scalers[c] = sc
    return out, scalers

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def compute_drawdowns(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_max - 1.0
    return float(np.min(drawdowns)) if len(drawdowns) else 0.0

def diebold_mariano_errors(e1, e2):
    e1, e2 = np.asarray(e1), np.asarray(e2)
    m = min(len(e1), len(e2))
    if m < 2:
        return None, None
    stat, p = ttest_rel(e1[:m], e2[:m])
    return float(stat), float(p)

def model_num_params(keras_model):
    try:
        return int(keras_model.count_params())
    except Exception:
        return None

# ======================= Data Layer ============================
def load_stock(ticker):
    print(f"  - Loading {ticker} ...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return df

def load_trends(keywords, geo=""):
    for attempt in range(2):
        try:
            print(f"  - Loading Google Trends {keywords} geo={geo} ...")
            pytrends = TrendReq(hl="en-US", tz=0)
            pytrends.build_payload(kw_list=keywords, timeframe=f"{START_DATE} {END_DATE}", geo=geo)
            gt = pytrends.interest_over_time()
            if gt is None or gt.empty:
                raise ValueError("Empty Google Trends response")
            if "isPartial" in gt.columns:
                gt = gt.drop(columns=["isPartial"])
            return gt
        except Exception:
            if attempt == 0:
                time.sleep(1.0)
            else:
                raise

def align_data(stock_df, gt_df):
    gt_b = gt_df.resample("B").ffill()
    df = pd.DataFrame(index=stock_df.index)
    df["Close"] = stock_df["Close"]
    if "Volume" in stock_df.columns:
        df["Volume"] = stock_df["Volume"]
    for c in gt_b.columns:
        df[c] = gt_b[c]
    df = df.dropna()
    return df

def add_lags(df, cols, lags):
    out = df.copy()
    for c in cols:
        for L in lags:
            out[f"{c}_lag{L}"] = out[c].shift(L)
    return out.dropna()

# ======================= Technical Indicators ==================
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def add_tech_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"]  = df["Close"].pct_change()
    df["ma_5"]   = df["Close"].rolling(5).mean()
    df["ma_20"]  = df["Close"].rolling(20).mean()
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["rsi_14"] = compute_rsi(df["Close"], period=14)
    if "Volume" in df.columns:
        mu = df["Volume"].rolling(20).mean()
        sd = df["Volume"].rolling(20).std()
        df["vol_z"] = (df["Volume"] - mu) / (sd + 1e-9)
    else:
        df["vol_z"] = np.nan
    return df.dropna()

# ======================= EDA (light) ===========================
def plot_price_vs_trends(ticker, df, gt_cols):
    tmp, _ = normalize_minmax(df.copy(), ["Close"] + gt_cols)
    plt.figure(figsize=(12,5))
    plt.plot(tmp.index, tmp["Close_norm"], label="Price")
    for c in gt_cols:
        plt.plot(tmp.index, tmp[c + "_norm"], label=c)
    plt.legend()
    plt.title(f"{ticker} — Normalized Price vs. Google Trends")
    save_fig(f"{ticker}_price_vs_trends.png")
    plt.close()

def correlations(df, gt_cols):
    res = {}
    tmp, _ = normalize_minmax(df.copy(), ["Close"] + gt_cols)
    for c in gt_cols:
        pear, _ = pearsonr(tmp["Close_norm"], tmp[c + "_norm"])
        spear,_ = spearmanr(tmp["Close_norm"], tmp[c + "_norm"])
        res[c] = {"pearson": float(pear), "spearman": float(spear)}
    return res

def granger(df, gt_cols, maxlag=3):
    out = {}
    tmp, _ = normalize_minmax(df.copy(), ["Close"] + gt_cols)
    for c in gt_cols:
        try:
            g = grangercausalitytests(tmp[["Close_norm", c + "_norm"]], maxlag=maxlag, verbose=False)
            pvals = {int(lag): float(stat[0]["ssr_ftest"][1]) for lag, stat in g.items()}
            out[c] = pvals
        except Exception as e:
            out[c] = {"error": str(e)}
    return out

# ======================= Features ==============================
def make_features_no_split(df, feat_cols, lookback):
    work = df[["Close"] + feat_cols].copy()
    scaler = MinMaxScaler()
    work[["Close"] + feat_cols] = scaler.fit_transform(work[["Close"] + feat_cols])
    y = work["Close"].values.reshape(-1, 1)
    X = work.values
    Xs, ys = create_sequences(X, y, lookback)
    return Xs, ys, scaler

def rolling_splits(n_samples, train_ratio=0.6, val_ratio=0.2, step=20):
    assert train_ratio + val_ratio < 1
    test_ratio = 1 - train_ratio - val_ratio
    window = int(n_samples * (train_ratio + val_ratio))
    test_size = int(n_samples * test_ratio)
    splits = []
    start = 0
    while start + window + test_size <= n_samples:
        tr_end  = int(start + train_ratio * window)
        val_end = start + window
        idx_tr  = np.arange(start, tr_end)
        idx_val = np.arange(tr_end, val_end)
        idx_te  = np.arange(val_end, val_end + test_size)
        splits.append((idx_tr, idx_val, idx_te))
        start += step
    return splits

# ======================= Transformer ===========================
def sinusoidal_position_encoding(length, d_model):
    positions = np.arange(length)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims//2)) / np.float32(d_model))
    angle_rads = positions * angle_rates
    sines = np.sin(angle_rads[:, 0::2])
    coses = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.zeros((length, d_model))
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = coses
    return tf.cast(pos_encoding, dtype=tf.float32)

def build_transformer(input_shape, d_model=64, num_heads=4, ff_dim=128, n_blocks=2, dropout=0.1):
    seq_len, _ = input_shape
    inp = Input(shape=input_shape)
    x = Dense(d_model)(inp)
    pe = sinusoidal_position_encoding(seq_len, d_model)
    x = x + pe
    for _ in range(n_blocks):
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_out)
        ff = Dense(ff_dim, activation="relu")(x)
        ff = Dropout(dropout)(ff)
        ff = Dense(d_model)(ff)
        x = LayerNormalization(epsilon=1e-6)(x + ff)
    x = x[:, -1, :]
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model

# ======================= Models & Training =====================
def build_model_from_hp(hp, input_shape):
    mdl = hp["model"]
    if mdl == "LSTM":
        m = Sequential([
            LSTM(hp["units1"], return_sequences=True, input_shape=input_shape),
            Dropout(hp["dropout"]),
            LSTM(hp["units2"]),
            Dense(1)
        ])
        m.compile(optimizer="adam", loss="mse")
        return m
    elif mdl == "GRU":
        m = Sequential([
            GRU(hp["units1"], return_sequences=True, input_shape=input_shape),
            Dropout(hp["dropout"]),
            GRU(hp["units2"]),
            Dense(1)
        ])
        m.compile(optimizer="adam", loss="mse")
        return m
    else:  # TRANSFORMER
        d_model = int(hp["units1"])
        ff_dim  = int(hp["units2"]) * 4
        return build_transformer(
            input_shape=input_shape,
            d_model=d_model,
            num_heads=4 if d_model >= 32 else 2,
            ff_dim=ff_dim,
            n_blocks=2,
            dropout=hp["dropout"]
        )

def apply_transaction_costs(signals, cost_bps=0.0):
    if cost_bps <= 0.0 or len(signals) < 2:
        return np.zeros_like(signals, dtype=float)
    flips = (np.diff(np.concatenate([[0], (signals > 0).astype(int)])) != 0).astype(int)
    costs = -cost_bps * flips[:len(signals)]
    return costs

def _keras_summary_str(model):
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)

def _save_keras_artifacts(model, ticker, tag):
    """Persist model and summary in HDF5 (.h5) for maximal compatibility."""
    try:
        mpath = os.path.join(MODEL_DIR, f"{ticker}_{tag}.h5")
        model.save(mpath, include_optimizer=False)  # HDF5 via .h5
        spath = os.path.join(MODEL_DIR, f"{ticker}_{tag}_summary.txt")
        write_text(_keras_summary_str(model), spath)
        return mpath, spath
    except Exception as e:
        print(f"[WARN] Model save failed: {e}")
        return None, None

def _save_predictions_csv(ticker, tag, dates, y_true, y_pred):
    try:
        dfp = pd.DataFrame({"date": pd.to_datetime(dates), "y_true": y_true, "y_pred": y_pred})
        outp = os.path.join(PRED_DIR, f"{ticker}_{tag}_preds.csv")
        dfp.to_csv(outp, index=False)
        print(f"[Saved] Predictions CSV -> {outp}")
    except Exception as e:
        print(f"[WARN] Could not save predictions CSV: {e}")

def _permute_feature_axis(X, feat_idx, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    order = rng.permutation(len(X))
    Xp[:, :, feat_idx] = X[order, :, feat_idx]
    return Xp

def permutation_importance_seq(model, X_val, y_val, base_rmse=None, max_features=None):
    """Permutation importance for sequence models on validation set."""
    if base_rmse is None:
        preds = model.predict(X_val, verbose=0).flatten()
        base_rmse = math.sqrt(mean_squared_error(y_val.flatten(), preds))
    n_feat = X_val.shape[2]
    feat_count = n_feat if (max_features is None) else min(n_feat, max_features)
    results = {}
    for j in range(feat_count):
        Xp = _permute_feature_axis(X_val, j)
        pred_p = model.predict(Xp, verbose=0).flatten()
        rmse_p = math.sqrt(mean_squared_error(y_val.flatten(), pred_p))
        results[int(j)] = float(rmse_p - base_rmse)
    return {"base_rmse": float(base_rmse), "rmse_increase_by_feature_index": results}

def train_eval_on_indices(model_builder, X, y, idx_tr, idx_val, idx_te, scaler, label,
                          ticker=None, save_artifacts=False, dates=None):
    # fit
    t0 = time.time()
    Xtr, ytr = X[idx_tr], y[idx_tr]
    Xva, yva = X[idx_val], y[idx_val]
    Xte, yte = X[idx_te], y[idx_te]

    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model = model_builder(Xtr.shape[1:])
    ckpt_path = os.path.join(MODEL_DIR, f"{ticker}_{label}_ckpt.h5") if (ticker and save_artifacts) else None
    cbs = [es]
    if ckpt_path:
        cbs.append(ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True))
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cbs, verbose=0)
    train_seconds = time.time() - t0

    preds = model.predict(Xte, verbose=0)

    # inverse scale (target only)
    tp = np.zeros((len(preds), scaler.n_features_in_))
    tt = np.zeros((len(yte),  scaler.n_features_in_))
    tp[:,0] = preds.flatten()
    tt[:,0] = yte.flatten()
    inv_pred = scaler.inverse_transform(tp)[:,0]
    inv_true = scaler.inverse_transform(tt)[:,0]

    # statistical metrics
    rmse = math.sqrt(mean_squared_error(inv_true, inv_pred))
    mae  = mean_absolute_error(inv_true, inv_pred)
    r2   = r2_score(inv_true, inv_pred)

    # directional accuracy
    act_dir  = np.sign(np.diff(inv_true))
    pred_dir = np.sign(np.diff(inv_pred))
    m = min(len(act_dir), len(pred_dir))
    act_dir, pred_dir = act_dir[:m], pred_dir[:m]
    dir_acc = accuracy_score(act_dir, pred_dir)
    cm = confusion_matrix(act_dir, pred_dir, labels=[-1,0,1]).tolist()

    # finance metrics (long/flat)
    ret = np.diff(inv_true) / inv_true[:-1]
    m2 = min(len(ret), len(pred_dir))
    ret, pred_dir2 = ret[:m2], pred_dir[:m2]

    strat = ret * (pred_dir2 > 0)
    cost_adj = apply_transaction_costs(pred_dir2, cost_bps=TRANSACTION_COST_BPS)
    strat_net = strat + cost_adj[:len(strat)]

    sharpe_gross = float(np.mean(strat) / (np.std(strat) + 1e-9)) if len(strat) else 0.0
    sharpe_net   = float(np.mean(strat_net) / (np.std(strat_net) + 1e-9)) if len(strat_net) else 0.0

    eq_gross = np.cumprod(1 + strat) if len(strat) else np.array([1.0])
    eq_net   = np.cumprod(1 + strat_net) if len(strat_net) else np.array([1.0])

    cumret_gross = float(eq_gross[-1] - 1) if len(eq_gross) else 0.0
    cumret_net   = float(eq_net[-1] - 1) if len(eq_net) else 0.0

    maxdd_gross = compute_drawdowns(eq_gross) if len(eq_gross) > 1 else 0.0
    maxdd_net   = compute_drawdowns(eq_net) if len(eq_net) > 1 else 0.0

    # buy & hold on same span
    eq_bh = np.cumprod(1 + ret) if len(ret) else np.array([1.0])
    bh_sharpe = float(np.mean(ret) / (np.std(ret) + 1e-9)) if len(ret) else 0.0
    bh_cumret = float(eq_bh[-1] - 1) if len(eq_bh) else 0.0
    bh_maxdd  = compute_drawdowns(eq_bh) if len(eq_bh) > 1 else 0.0

    # DM test errors (absolute)
    abs_err = np.abs(inv_true - inv_pred)

    # artifacts
    n_params = model_num_params(model)
    model_path, summary_path = (None, None)
    if save_artifacts and ticker:
        model_path, summary_path = _save_keras_artifacts(model, ticker, label)
        if dates is not None:
            _save_predictions_csv(ticker, label, dates[-len(inv_true):], inv_true, inv_pred)

    # explainability (validation)
    perm_imp = None
    if DO_PERM_IMPORTANCE:
        try:
            base_pred_val = model.predict(Xva, verbose=0).flatten()
            base_rmse_val = math.sqrt(mean_squared_error(yva.flatten(), base_pred_val))
            perm_imp = permutation_importance_seq(model, Xva, yva, base_rmse=base_rmse_val, max_features=PERM_FEATURE_LIMIT)
        except Exception as e:
            perm_imp = {"error": str(e)}

    res = {
        "label": label,
        "rmse": float(rmse), "mae": float(mae), "r2": float(r2),
        "directional_acc": float(dir_acc), "cm": cm,
        "sharpe_gross": float(sharpe_gross), "sharpe_net": float(sharpe_net),
        "cum_return_gross": float(cumret_gross), "cum_return_net": float(cumret_net),
        "maxdd_gross": float(maxdd_gross), "maxdd_net": float(maxdd_net),
        "bh_sharpe": float(bh_sharpe), "bh_cum_return": float(bh_cumret), "bh_maxdd": float(bh_maxdd),
        "n_test": int(len(inv_true)),
        "train_seconds": float(train_seconds),
        "n_params": n_params,
        "model_path": model_path,
        "summary_path": summary_path,
        "perm_importance_val": perm_imp
    }
    return res, inv_true, inv_pred, ret, pred_dir2, abs_err

def hp_search(X, y, scaler, base_label, ticker=None, dates=None, max_trials=HP_MAX_TRIALS):
    combos = list(product(HP_GRID["model"], HP_GRID["units1"], HP_GRID["units2"], HP_GRID["dropout"]))
    combos = combos[:max_trials]
    n = len(X)
    tr = int(0.6*n); vl = int(0.8*n)
    idx_tr = np.arange(0, tr); idx_val = np.arange(tr, vl); idx_te = np.arange(vl, n)

    trials, best = [], None
    for mdl, u1, u2, do in combos:
        hp = {"model": mdl, "units1": u1, "units2": u2, "dropout": do}
        label = f"{base_label}_{mdl}_u{u1}-{u2}_d{do}"
        def builder(shape): return build_model_from_hp(hp, shape)
        res, *_ = train_eval_on_indices(
            builder, X, y, idx_tr, idx_val, idx_te, scaler, label,
            ticker=ticker, save_artifacts=False, dates=dates
        )
        res["hp"] = hp
        trials.append(res)
        if (best is None) or (res["rmse"] < best["rmse"]):
            best = res
    return best, trials

# ======================= Plotting ==============================
def plot_truth_pred(ticker, label, y_true, y_pred):
    plt.figure(figsize=(10,4))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Pred")
    plt.title(f"{ticker} — {label}: Truth vs. Prediction")
    plt.legend()
    save_fig(f"{ticker}_{label}_truth_pred.png")
    plt.close()

def plot_equity_curve(ticker, label, returns, signals):
    strat = returns * (signals > 0)
    cost_adj = apply_transaction_costs(signals, cost_bps=TRANSACTION_COST_BPS)
    strat_net = strat + cost_adj[:len(strat)]
    eq_gross = np.cumprod(1 + strat)
    eq_net   = np.cumprod(1 + strat_net)
    eq_bh    = np.cumprod(1 + returns)
    plt.figure(figsize=(10,4))
    plt.plot(eq_bh, label="Buy&Hold")
    plt.plot(eq_gross, label="Strategy (gross)")
    plt.plot(eq_net, label="Strategy (net)")
    plt.title(f"{ticker} — {label}: Equity Curves")
    plt.legend()
    save_fig(f"{ticker}_{label}_equity.png")
    plt.close()

# ======================= ARIMA Baseline ========================
def arima_baseline_predict(train_series, test_len, order=(1,0,0)):
    try:
        model = ARIMA(train_series, order=order).fit()
        forecast = model.forecast(steps=test_len)
        return np.asarray(forecast)
    except Exception:
        return np.repeat(train_series[-1], test_len)

def arima_eval_on_split(series, idx_tr, idx_val, idx_te):
    train_series = series[:idx_te[0]]
    test_series  = series[idx_te]
    preds = arima_baseline_predict(train_series, len(test_series), order=(1,0,0))

    inv_true = np.asarray(test_series)
    inv_pred = np.asarray(preds)

    rmse = math.sqrt(mean_squared_error(inv_true, inv_pred))
    mae  = mean_absolute_error(inv_true, inv_pred)
    r2   = r2_score(inv_true, inv_pred)

    act_dir  = np.sign(np.diff(inv_true))
    pred_dir = np.sign(np.diff(inv_pred))
    m = min(len(act_dir), len(pred_dir))
    act_dir, pred_dir = act_dir[:m], pred_dir[:m]
    dir_acc = accuracy_score(act_dir, pred_dir)
    cm = confusion_matrix(act_dir, pred_dir, labels=[-1,0,1]).tolist()

    ret = np.diff(inv_true) / inv_true[:-1]
    m2 = min(len(ret), len(pred_dir))
    ret, pred_dir2 = ret[:m2], pred_dir[:m2]

    strat = ret * (pred_dir2 > 0)
    cost_adj = apply_transaction_costs(pred_dir2, cost_bps=TRANSACTION_COST_BPS)
    strat_net = strat + cost_adj[:len(strat)]

    sharpe_gross = float(np.mean(strat) / (np.std(strat) + 1e-9)) if len(strat) else 0.0
    sharpe_net   = float(np.mean(strat_net) / (np.std(strat_net) + 1e-9)) if len(strat_net) else 0.0

    eq_gross = np.cumprod(1 + strat) if len(strat) else np.array([1.0])
    eq_net   = np.cumprod(1 + strat_net) if len(strat_net) else np.array([1.0])

    cumret_gross = float(eq_gross[-1] - 1) if len(eq_gross) else 0.0
    cumret_net   = float(eq_net[-1] - 1) if len(eq_net) else 0.0

    maxdd_gross = compute_drawdowns(eq_gross) if len(eq_gross) > 1 else 0.0
    maxdd_net   = compute_drawdowns(eq_net) if len(eq_net) > 1 else 0.0

    eq_bh = np.cumprod(1 + ret) if len(ret) else np.array([1.0])
    bh_sharpe = float(np.mean(ret) / (np.std(ret) + 1e-9)) if len(ret) else 0.0
    bh_cumret = float(eq_bh[-1] - 1) if len(eq_bh) else 0.0
    bh_maxdd  = compute_drawdowns(eq_bh) if len(eq_bh) > 1 else 0.0

    res = {
        "label": "ARIMA_baseline",
        "rmse": float(rmse), "mae": float(mae), "r2": float(r2),
        "directional_acc": float(dir_acc), "cm": cm,
        "sharpe_gross": float(sharpe_gross), "sharpe_net": float(sharpe_net),
        "cum_return_gross": float(cumret_gross), "cum_return_net": float(cumret_net),
        "maxdd_gross": float(maxdd_gross), "maxdd_net": float(maxdd_net),
        "bh_sharpe": float(bh_sharpe), "bh_cum_return": float(bh_cumret), "bh_maxdd": float(bh_maxdd),
        "n_test": int(len(inv_true)),
        "train_seconds": 0.0,
        "n_params": None
    }
    return res, inv_true, inv_pred, ret, pred_dir2

# ======================= Orchestration =========================
def pretty_ticker(tk):
    mapping = {"^GSPC": "S&P 500", "GC=F": "Gold Futures", "BTC-USD": "Bitcoin"}
    return mapping.get(tk, tk)

def run_ticker(ticker, kw_list, geo):
    print(f"\n=== Processing {ticker} ({pretty_ticker(ticker)}) ===")
    stock = load_stock(ticker)
    gt = load_trends(kw_list, geo)
    df = align_data(stock, gt)
    if USE_TECH_FEATURES:
        df = add_tech_indicators(df)

    # EDA snapshots
    plot_price_vs_trends(ticker, df, kw_list)
    stats = {"corr": correlations(df, kw_list), "granger": granger(df, kw_list)}
    save_json(stats, f"{ticker}_stats.json")

    all_model_records, dm_tests_records = [], []
    baseline_errors_by_lag = {}
    all_dates = df.index.values  # for saving predictions

    for lag in GT_LAGS:
        print(f"  - Lag {lag}")
        tmp = add_lags(df, kw_list, [lag])
        gt_cols   = [f"{k}_lag{lag}" for k in kw_list]
        tech_cols = [c for c in TECH_FEATURES if c in tmp.columns]
        feature_sets = [
            ("prices",     []),
            ("withGT",     gt_cols),
            ("withTECH",   tech_cols),
            ("withGTTECH", gt_cols + tech_cols)
        ]

        # build sequences for each set
        built = {}
        for name, cols in feature_sets:
            X_all, y_all, scaler = make_features_no_split(tmp, cols, LOOKBACK)
            built[name] = (X_all, y_all, scaler)

        fixed_preds_cache = {}

        # ARIMA baseline on same split
        n_all = len(tmp["Close"].values)
        tr = int(0.6*n_all); vl = int(0.8*n_all)
        idx_tr = np.arange(0, tr); idx_val = np.arange(tr, vl); idx_te = np.arange(vl, n_all)
        ar_res, *_ = arima_eval_on_split(tmp["Close"].values, idx_tr, idx_val, idx_te)
        ar_res.update({"ticker": ticker, "feature_set": "ARIMA", "split": "fixed", "lag": lag})
        all_model_records.append(ar_res)

        # HP search (fixed split) then retrain best and save artifacts + preds
        for name, (X_all, y_all, scaler) in built.items():
            base_label = f"{name}_lag{lag}"

            best, trials = hp_search(X_all, y_all, scaler, base_label, ticker=ticker, dates=all_dates)
            save_json(trials, f"{ticker}_{name}_lag{lag}_hp_trials.json")

            hp = best["hp"]
            def builder(shape): return build_model_from_hp(hp, shape)
            n = len(X_all); tr = int(0.6*n); vl = int(0.8*n)
            idx_tr = np.arange(0,tr); idx_val = np.arange(tr,vl); idx_te = np.arange(vl,n)
            res, inv_true, inv_pred, ret, pred_dir, abs_err = train_eval_on_indices(
                builder, X_all, y_all, idx_tr, idx_val, idx_te, scaler,
                f"{base_label}_BEST", ticker=ticker, save_artifacts=True, dates=all_dates[-len(X_all):]
            )
            res.update({"ticker": ticker, "feature_set": name, "split": "fixed", "lag": lag, "hp": hp})
            all_model_records.append(res)
            fixed_preds_cache[name] = (inv_true, inv_pred, ret, pred_dir)

            if name == "prices":
                baseline_errors_by_lag[lag] = abs_err

        # DM tests: PX vs withGT / PX vs withGTTECH (fixed split)
        px_err = baseline_errors_by_lag.get(lag, None)
        for comp in ["withGT", "withGTTECH"]:
            if comp in fixed_preds_cache and px_err is not None:
                inv_true_c, inv_pred_c, _, _ = fixed_preds_cache[comp]
                abs_err_c = np.abs(inv_true_c - inv_pred_c)
                stat, p = diebold_mariano_errors(px_err, abs_err_c)
                dm_tests_records.append({
                    "ticker": ticker, "lag": lag, "compare": f"PX vs {comp}",
                    "dm_stat": stat, "p_value": p
                })

        # Walk-forward evaluation using best HP per feature set
        if USE_ROLLING:
            for name, (X_all, y_all, scaler) in built.items():
                bests = [r for r in all_model_records
                         if (r.get("ticker")==ticker and r.get("feature_set")==name
                             and r.get("split")=="fixed" and r.get("lag")==lag and "hp" in r)]
                if not bests:
                    continue
                hp = bests[-1]["hp"]
                def builder(shape): return build_model_from_hp(hp, shape)
                splits = rolling_splits(len(X_all), train_ratio=0.6, val_ratio=0.2, step=ROLL_STEP)
                roll_metrics = []
                for i, (idx_tr, idx_val, idx_te) in enumerate(splits):
                    lbl = f"{name}_lag{lag}_roll{i}"
                    res, inv_true, inv_pred, ret, pred_dir, _ = train_eval_on_indices(
                        builder, X_all, y_all, idx_tr, idx_val, idx_te, scaler, lbl,
                        ticker=ticker, save_artifacts=False
                    )
                    res.update({"ticker": ticker, "feature_set": name, "split": f"rolling_{i}", "lag": lag})
                    all_model_records.append(res)
                    roll_metrics.append(res)
                    if i == 0:
                        plot_truth_pred(ticker, lbl, inv_true, inv_pred)
                        plot_equity_curve(ticker, lbl, ret, pred_dir)

                if roll_metrics:
                    dfm = pd.DataFrame(roll_metrics)
                    agg = {
                        "rmse_mean": float(dfm["rmse"].mean()), "rmse_std": float(dfm["rmse"].std()),
                        "mae_mean": float(dfm["mae"].mean()),   "mae_std": float(dfm["mae"].std()),
                        "r2_mean": float(dfm["r2"].mean()),     "r2_std": float(dfm["r2"].std()),
                        "diracc_mean": float(dfm["directional_acc"].mean()),
                        "diracc_std":  float(dfm["directional_acc"].std()),
                        "sharpe_gross_mean": float(dfm["sharpe_gross"].mean()),
                        "sharpe_gross_std":  float(dfm["sharpe_gross"].std()),
                        "sharpe_net_mean": float(dfm["sharpe_net"].mean()),
                        "sharpe_net_std":  float(dfm["sharpe_net"].std()),
                        "cumret_gross_mean": float(dfm["cum_return_gross"].mean()),
                        "cumret_gross_std":  float(dfm["cum_return_gross"].std()),
                        "cumret_net_mean": float(dfm["cum_return_net"].mean()),
                        "cumret_net_std":  float(dfm["cum_return_net"].std()),
                        "maxdd_gross_mean": float(dfm["maxdd_gross"].mean()),
                        "maxdd_gross_std":  float(dfm["maxdd_gross"].std()),
                        "maxdd_net_mean": float(dfm["maxdd_net"].mean()),
                        "maxdd_net_std":  float(dfm["maxdd_net"].std()),
                    }
                    save_json(agg, f"{ticker}_{name}_lag{lag}_rolling_agg.json")

        # Figures for best fixed models
        for name in ["prices", "withGT", "withTECH", "withGTTECH"]:
            if name in fixed_preds_cache and fixed_preds_cache[name] is not None:
                inv_true, inv_pred, ret, pred_dir = fixed_preds_cache[name]
                lbl = f"{name}_lag{lag}_fixedBEST"
                plot_truth_pred(ticker, lbl, inv_true, inv_pred)
                plot_equity_curve(ticker, lbl, ret, pred_dir)

    # persist per-ticker outputs
    save_json(all_model_records, f"{ticker}_models_pro.json")
    if dm_tests_records:
        save_json(dm_tests_records, f"{ticker}_dmtests.json")

def _save_keywords_table():
    rows = []
    for tk, cfg in KEYWORDS.items():
        rows.append({"ticker": tk, "geo": cfg.get("geo",""), "keywords": ", ".join(cfg.get("kw", []))})
    dfk = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "KEYWORDS_USED.csv")
    dfk.to_csv(path, index=False)
    print(f"[Saved] Keywords table -> {path}")

def _save_run_metadata():
    meta = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "tensorflow": tf.__version__,
            "yfinance": getattr(yf, "__version__", "N/A"),
        },
        "config": {
            "START_DATE": START_DATE,
            "END_DATE": END_DATE,
            "TICKERS": TICKERS,
            "USE_LOCAL": USE_LOCAL,
            "GT_LAGS": GT_LAGS,
            "LOOKBACK": LOOKBACK,
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "VAL_SPLIT": VAL_SPLIT,
            "USE_ROLLING": USE_ROLLING,
            "ROLL_STEP": ROLL_STEP,
            "TRANSACTION_COST_BPS": TRANSACTION_COST_BPS,
            "RANDOM_SEED": RANDOM_SEED,
            "USE_TECH_FEATURES": USE_TECH_FEATURES,
            "TECH_FEATURES": TECH_FEATURES,
            "HP_GRID": HP_GRID,
            "HP_MAX_TRIALS": HP_MAX_TRIALS,
            "DO_PERM_IMPORTANCE": DO_PERM_IMPORTANCE,
            "PERM_FEATURE_LIMIT": PERM_FEATURE_LIMIT
        }
    }
    path = os.path.join(OUTPUT_DIR, "run_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(meta), f, indent=2, ensure_ascii=False)
    print(f"[Saved] Run metadata -> {path}")

def main():
    print("STEP 1 — Pipeline starting...")
    _save_keywords_table()
    _save_run_metadata()

    for tk in TICKERS:
        cfg = KEYWORDS.get(tk, {"kw": ["stock market"], "geo": ""})
        run_ticker(tk, cfg["kw"], cfg["geo"])

    # Master CSV (all per-ticker model records)
    rows = []
    for f in os.listdir(RES_DIR):
        if f.endswith("_models_pro.json"):
            with open(os.path.join(RES_DIR, f), "r", encoding="utf-8") as fh:
                data = json.load(fh)
                rows.extend(data)
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(OUTPUT_DIR, "ALL_MODELS_SUMMARY.csv")
        df.to_csv(csv_path, index=False)
        print("[Saved] CSV ->", csv_path)

    print("\nDone. Check:")
    print(" - Figures:", FIG_DIR)
    print(" - Results JSON:", RES_DIR)
    print(" - Models (.h5):", MODEL_DIR)
    print(" - Predictions CSV:", PRED_DIR)

if __name__ == "__main__":
    main()
