#!/usr/bin/env python3
# -- coding: utf-8 --

"""
thesis_visualization.py — Academic Visualization & Interactive Dashboard

Inputs (produced by thesis_pipeline.py / thesis_results.py):
  - outputs/ALL_MODELS_SUMMARY.csv
  - outputs/results/*_models_pro.json
  - outputs/results/*_dmtests.json
  - outputs/results/*_stats.json            (price vs Google Trends correlations)
  - outputs/preds/*_preds.csv               (test-set truth/pred for BEST models)

Outputs (static):
  - Overview heatmaps (Sharpe/RMSE/MAE) per ticker × feature_set
  - Best-per-ticker bar charts
  - Radar charts (normalized multi-metric comparison)
  - Correlation matrices (price vs GT)
  - Feature importance bar charts (from permutation importance if available)
  - Temporal analysis plots (lags; crisis windows)
  - Saved under outputs/figs/

Interactive (optional, --dash):
  - Plotly Dash app (overview, explainability, temporal, scenario sensitivity)
"""

import os, json, argparse
from glob import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional interactive
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from dash import Dash, dcc, html, Input, Output
    DASH_AVAILABLE = True
except Exception:
    DASH_AVAILABLE = False

# --------------------- Paths ---------------------
ROOT = os.path.abspath(".")
OUT_DIR   = os.path.join(ROOT, "outputs")
FIG_DIR   = os.path.join(OUT_DIR, "figs")
RES_DIR   = os.path.join(OUT_DIR, "results")
PRED_DIR  = os.path.join(OUT_DIR, "preds")
os.makedirs(FIG_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(OUT_DIR, "ALL_MODELS_SUMMARY.csv")

# Crisis windows (for temporal robustness)
CRISIS_WINDOWS = {
    "COVID_2020": ("2020-02-15", "2020-04-30"),
    "BEAR_2022":  ("2022-01-01", "2022-10-31")
}

# Trading frictions aligned with pipeline
TRANSACTION_COST_BPS = 0.0005


# ====================== IO ======================
def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _collect(pattern, folder=RES_DIR):
    return sorted(glob(os.path.join(folder, pattern)))

def _savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")

def _load_summary():
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError("ALL_MODELS_SUMMARY.csv not found. Run thesis_results.py first.")
    df = pd.read_csv(SUMMARY_CSV)
    # ensure numeric dtypes
    for c in df.columns:
        if c not in ("ticker","feature_set","split","label","model_path","summary_path"):
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


# ====================== Overview / Meta ======================
def pivot_metric_heatmap(df, metric, fname):
    """Heatmap: rows=ticker, cols=feature_set, values=metric (best by fixed split)."""
    if metric not in df.columns:
        print(f"[WARN] metric '{metric}' not found in summary — skipping heatmap.")
        return None
    dff = df[df.get("split","")=="fixed"].copy()
    if dff.empty:
        print("[WARN] no fixed-split rows — skipping heatmap.")
        return None
    # choose best per ticker×feature_set by metric direction
    asc = True if metric.lower() in ("rmse","mae") else False
    dff = dff.sort_values(["ticker","feature_set",metric], ascending=[True,True,asc])
    best = dff.groupby(["ticker","feature_set"], as_index=False).first()
    if best.empty:
        print("[WARN] no rows after best-per-group — skipping heatmap.")
        return None
    pt = best.pivot_table(index="ticker", columns="feature_set", values=metric, aggfunc="first")
    if pt.empty:
        print("[WARN] pivot is empty — skipping heatmap.")
        return None
    # plot
    plt.figure(figsize=(8, 5))
    plt.imshow(pt.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label=metric)
    plt.xticks(range(len(pt.columns)), pt.columns, rotation=45, ha="right")
    plt.yticks(range(len(pt.index)), pt.index)
    plt.title(f"{metric} — best fixed per (ticker × feature set)")
    _savefig(fname)
    return pt

def best_per_ticker_bar(df, score="rmse", fname="best_by_ticker_rmse.png"):
    """Bar: best fixed model per ticker (by score)."""
    if score not in df.columns:
        print(f"[WARN] score '{score}' not found — skipping bar.")
        return None
    dff = df[df.get("split","")=="fixed"].copy()
    if dff.empty:
        print("[WARN] no fixed-split rows — skipping bar.")
        return None
    asc = True if score.lower() in ("rmse","mae") else False
    dff = dff.sort_values(["ticker",score], ascending=[True,asc])
    best = dff.groupby("ticker", as_index=False).first()
    if best.empty:
        print("[WARN] no rows for best-per-ticker bar.")
        return None
    plt.figure(figsize=(8,4))
    plt.bar(best["ticker"], best[score])
    plt.title(f"Best fixed model per ticker — {score}")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(score)
    _savefig(fname)
    return best

def radar_by_featureset(df, metrics=("rmse","mae","directional_acc","sharpe_net"), fname="radar_featuresets.png"):
    """Radar: normalized mean metrics per feature_set (fixed best)."""
    for m in metrics:
        if m not in df.columns:
            print(f"[WARN] metric '{m}' missing — radar will ignore it.")
    dff = df[df.get("split","")=="fixed"].copy()
    if dff.empty:
        print("[WARN] no fixed-split rows — skipping radar.")
        return None
    # pick best per ticker×feature_set by RMSE to avoid multiple lines (fallback if rmse missing)
    sort_key = "rmse" if "rmse" in dff.columns else metrics[0]
    asc = True if sort_key in ("rmse","mae") else False
    dff = dff.sort_values(["ticker","feature_set",sort_key], ascending=[True,True,asc])
    best = dff.groupby(["ticker","feature_set"], as_index=False).first()
    keep = [m for m in metrics if m in best.columns]
    if not keep:
        print("[WARN] no usable metrics for radar.")
        return None
    agg = best.groupby("feature_set")[keep].mean(numeric_only=True)

    # normalize: lower-is-better metrics inverted
    norm = agg.copy()
    for m in keep:
        s = agg[m].astype(float)
        if s.nunique() <= 1:
            norm[m] = 1.0  # flat
            continue
        if m.lower() in ("rmse","mae"):
            norm[m] = (s.max() - s) / (s.max() - s.min() + 1e-9)
        else:
            norm[m] = (s - s.min()) / (s.max() - s.min() + 1e-9)

    metrics_used = keep
    theta = list(metrics_used) + [metrics_used[0]]
    labels = norm.index.tolist()
    plt.figure(figsize=(6.5,6.5))
    ax = plt.subplot(111, polar=True)
    angles = np.linspace(0, 2*np.pi, len(theta), endpoint=True)
    for fs in labels:
        vals = norm.loc[fs, list(metrics_used)].tolist() + [norm.loc[fs, metrics_used[0]]]
        ax.plot(angles, vals, linewidth=1)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_used)
    ax.set_yticklabels([])
    ax.set_title("Normalized multi-metric comparison by feature set (fixed best)")
    ax.legend(labels, bbox_to_anchor=(1.1, 1.05))
    _savefig(fname)
    return norm


# ====================== Explainability ======================
def load_stats_price_vs_gt():
    """Reads *_stats.json → {'corr': {'kw': {pearson,spearman}}, 'granger': ...}."""
    rows = []
    for p in _collect("*_stats.json"):
        data = _load_json(p)
        if not data: continue
        ticker = os.path.basename(p).split("_stats.json")[0]
        corr = data.get("corr", {})
        for kw, obj in corr.items():
            rows.append({"ticker": ticker, "keyword": kw,
                         "pearson": obj.get("pearson", np.nan),
                         "spearman": obj.get("spearman", np.nan)})
    return pd.DataFrame(rows)

def corr_matrix_price_vs_gt(df_stats, how="pearson", fname="corr_price_vs_gt.png"):
    """Aggregated correlation heatmap: rows=ticker, cols=keyword (avg across files)."""
    if df_stats is None or df_stats.empty:
        print("[INFO] price-vs-GT stats not found — skipping corr heatmap.")
        return None
    if how not in df_stats.columns:
        print(f"[WARN] correlation '{how}' missing — skipping.")
        return None
    pt = df_stats.pivot_table(index="ticker", columns="keyword", values=how, aggfunc="mean")
    if pt.empty:
        print("[WARN] corr pivot empty — skipping.")
        return None
    plt.figure(figsize=(8.5, 5.5))
    plt.imshow(pt.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label=f"{how} correlation")
    plt.xticks(range(len(pt.columns)), pt.columns, rotation=45, ha="right")
    plt.yticks(range(len(pt.index)), pt.index)
    plt.title(f"Price vs Google Trends — {how} correlation")
    _savefig(fname)
    return pt

def extract_perm_importance(best_fixed_df):
    """Extracts permutation importance dicts from *_models_pro.json for fixed best rows."""
    out = []
    if best_fixed_df is None or best_fixed_df.empty:
        return out
    # Index by (ticker, feature_set, lag, label)
    index_rows = {(r["ticker"], r["feature_set"], int(r["lag"]), r["label"]) for _, r in best_fixed_df.iterrows()}
    for p in _collect("*_models_pro.json"):
        data = _load_json(p)
        if not isinstance(data, list): continue
        for rec in data:
            key = (rec.get("ticker"), rec.get("feature_set"), int(rec.get("lag", -1)), rec.get("label"))
            if key in index_rows and isinstance(rec.get("perm_importance_val"), dict):
                out.append(rec)
    return out

def _feature_names(feature_set, lag):
    """Reconstructs feature order used in sequences."""
    names = ["Close"]
    TECH = ["ret_1","ma_5","ma_20","vol_10","rsi_14","vol_z"]
    if feature_set == "prices":
        return names
    if feature_set == "withGT":
        return names + [f"KW{j}_lag{lag}" for j in range(1, 9)]
    if feature_set == "withTECH":
        return names + TECH
    if feature_set == "withGTTECH":
        return names + [f"KW{j}_lag{lag}" for j in range(1, 9)] + TECH
    return names

def plot_perm_importance(best_fixed_df, top_k=10):
    """Bar plots of validation permutation importance (RMSE increase) for selected best models."""
    recs = extract_perm_importance(best_fixed_df)
    if not recs:
        print("[INFO] no permutation-importance dicts found — skipping.")
        return
    for rec in recs:
        fs = rec.get("feature_set"); lag = int(rec.get("lag", 0))
        pi = rec.get("perm_importance_val", {})
        inc = pi.get("rmse_increase_by_feature_index", {})
        if not isinstance(inc, dict) or len(inc)==0:
            continue
        fi = sorted(((int(k), float(v)) for k,v in inc.items()), key=lambda x: x[1], reverse=True)[:top_k]
        names = _feature_names(fs, lag)
        labels = [names[i] if i < len(names) else f"f{i}" for i,_ in fi]
        vals = [v for _,v in fi]
        plt.figure(figsize=(7,4))
        plt.barh(labels, vals)
        plt.gca().invert_yaxis()
        plt.xlabel("RMSE increase (validation)")
        plt.title(f"Permutation importance — {rec.get('ticker')} [{fs}, lag={lag}]")
        fname = f"permimp_{rec.get('ticker')}_{fs}_lag{lag}.png".replace("/","-")
        _savefig(fname)


# ====================== Temporal Analysis ======================
def _load_preds_files():
    files = _collect("*_preds.csv", folder=PRED_DIR)
    out = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["date"])
            df["__file"] = os.path.basename(f)
            out.append(df)
        except Exception:
            pass
    return out

def _equity_from_preds(df_series, cost_bps=TRANSACTION_COST_BPS):
    """Long/flat: sign(diff(pred)) as signal, apply costs, equity vs buy&hold."""
    y_true = df_series["y_true"].values
    y_pred = df_series["y_pred"].values
    ret = np.diff(y_true) / (y_true[:-1] + 1e-9)
    sig = np.sign(np.diff(y_pred))
    m = min(len(ret), len(sig))
    ret = ret[:m]; sig = sig[:m]
    strat = ret * (sig > 0)
    flips = (np.diff(np.concatenate([[0], (sig>0).astype(int)])) != 0).astype(int)
    costs = -cost_bps * flips[:len(strat)]
    strat_net = strat + costs
    eq_net = np.cumprod(1 + strat_net)
    eq_bh  = np.cumprod(1 + ret)
    return eq_net, eq_bh

def plot_lag_comparison(df_summary, prefer_fs="withGTTECH", ticker=None):
    """Compare lag 0/7/14 (fixed best) for a chosen feature_set (per-ticker)."""
    dff = df_summary[(df_summary.get("split","")=="fixed") & (df_summary.get("feature_set","")==prefer_fs)].copy()
    if ticker is not None:
        dff = dff[dff["ticker"]==ticker]
    keep_lags = [0,7,14]
    if "lag" not in dff.columns or "rmse" not in dff.columns or dff.empty:
        print("[WARN] missing lag/rmse or no rows — skipping lag comparison.")
        return
    dff = dff[dff["lag"].isin(keep_lags)]
    if dff.empty:
        print("[INFO] no requested lags — skipping.")
        return
    dff = dff.sort_values("lag")
    plt.figure(figsize=(7.5,4))
    for _, r in dff.iterrows():
        plt.plot([r["lag"]], [r["rmse"]], "o", label=f"lag {int(r['lag'])}")
    plt.title(f"RMSE vs lag — {ticker if ticker else 'all'} [{prefer_fs}]")
    plt.xlabel("lag"); plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"outputs/figs/lag_rmse{ticker if ticker else 'all'}_{prefer_fs}.png", dpi=300, bbox_inches="tight")
plt.close()


def plot_crisis_equity_examples():
    """Equity curves during crisis windows using available preds CSVs."""
    preds = _load_preds_files()
    for dfp in preds:
        if dfp is None or dfp.empty:
            continue
        if "date" not in dfp.columns or "y_true" not in dfp.columns or "y_pred" not in dfp.columns:
            continue
        # try slicing by crisis windows
        for cname, (s,e) in CRISIS_WINDOWS.items():
            dsub = dfp[(dfp["date"]>=pd.Timestamp(s)) & (dfp["date"]<=pd.Timestamp(e))].copy()
            if len(dsub) < 10:
                continue
            eq_net, eq_bh = _equity_from_preds(dsub)
            plt.figure(figsize=(7.5,4))
            plt.plot(eq_bh, label="Buy&Hold")
            plt.plot(eq_net, label="Strategy (net)")
            plt.title(f"Equity during {cname} — {dfp['__file'].iloc[0]}")
            plt.legend()
            base = dfp['__file'].iloc[0].replace('.csv','')
            fname = f"equity_{cname}_{base}.png"
            _savefig(fname)


# ====================== Scenario Sensitivity (stylized) ======================
def scenario_sensitivity_from_corr(df_stats, shock=2.0, how="pearson"):
    """
    Stylized scenario: proxy Δprice ≈ k * corr * shock, k=0.5%.
    Produces per-ticker aggregated bars by keyword (illustrative).
    """
    if df_stats is None or df_stats.empty or how not in df_stats.columns:
        print("[INFO] insufficient corr stats — skipping sensitivity.")
        return
    k = 0.005
    df = df_stats.copy()
    df["impact_proxy"] = k * df[how].astype(float) * float(shock)
    agg = df.groupby(["ticker","keyword"], as_index=False)["impact_proxy"].mean()
    # plot top impacts per ticker
    for tk, g in agg.groupby("ticker"):
        g = g.sort_values("impact_proxy", ascending=False).head(8)
        plt.figure(figsize=(7,4))
        plt.barh(g["keyword"], g["impact_proxy"])
        plt.gca().invert_yaxis()
        plt.xlabel("Impact proxy")
        plt.title(f"Sensitivity (shock ×{shock}) — {tk}")
        plt.savefig(f"outputs/figs/sensitivity{tk}.png", dpi=300, bbox_inches="tight")
plt.close()



# ====================== Interactive Dashboard (optional) ======================
def launch_dash(df_summary, df_stats):
    if not DASH_AVAILABLE:
        print("[INFO] Plotly/Dash not available. Install plotly & dash to enable the dashboard.")
        return

    app = Dash(__name__)
    app.title = "Thesis Visualization"

    # Precompute overview heatmap (Sharpe)
    dff = df_summary[df_summary.get("split","")=="fixed"].copy()
    if "sharpe_net" in dff.columns:
        dff = dff.sort_values(["ticker","feature_set","sharpe_net"], ascending=[True,True,False])
        best = dff.groupby(["ticker","feature_set"], as_index=False).first()
        pt = best.pivot_table(index="ticker", columns="feature_set", values="sharpe_net", aggfunc="first")
        heat = go.Figure(data=go.Heatmap(
            z=pt.values if not pt.empty else [],
            x=list(pt.columns) if not pt.empty else [],
            y=list(pt.index) if not pt.empty else [],
            colorbar_title="Sharpe(net)")
        )
        heat.update_layout(margin=dict(l=60,r=40,t=40,b=40), title="Sharpe (fixed best)")
    else:
        heat = go.Figure()

    # Layout
    app.layout = html.Div([
        html.H2("Results Overview / Explainability / Temporal Analysis"),
        dcc.Tabs([
            dcc.Tab(label="Overview", children=[
                dcc.Graph(figure=heat),
                html.Br(),
                html.Div([
                    "Score:",
                    dcc.Dropdown(
                        id="score-dd",
                        options=[{"label":x,"value":x} for x in ["rmse","mae","directional_acc","sharpe_net"] if x in df_summary.columns],
                        value="sharpe_net" if "sharpe_net" in df_summary.columns else ( "rmse" if "rmse" in df_summary.columns else None ),
                        clearable=False, style={"width":"220px"}
                    )
                ]),
                dcc.Graph(id="best-by-ticker")
            ]),
            dcc.Tab(label="Explainability", children=[
                html.Div([
                    "Correlation metric:",
                    dcc.Dropdown(
                        id="corr-metric",
                        options=[{"label":"pearson","value":"pearson"},{"label":"spearman","value":"spearman"}],
                        value="pearson", clearable=False, style={"width":"220px"}
                    )
                ]),
                dcc.Graph(id="corr-heatmap"),
                html.Div("Sensitivity shock multiplier"),
                dcc.Slider(id="shock", min=0.5, max=3.0, step=0.5, value=2.0),
                dcc.Graph(id="sensitivity-bars")
            ]),
            dcc.Tab(label="Temporal", children=[
                html.Div([
                    "Ticker:",
                    dcc.Dropdown(
                        id="ticker-dd",
                        options=[{"label":t,"value":t} for t in sorted(df_summary["ticker"].unique())] if "ticker" in df_summary.columns else [],
                        value=(sorted(df_summary["ticker"].unique())[0] if "ticker" in df_summary.columns and len(df_summary["ticker"].unique())>0 else None),
                        clearable=False, style={"width":"260px"}
                    )
                ]),
                dcc.Graph(id="lag-rmse-line")
            ]),
        ], style={"marginTop":"10px"})
    ])

    @app.callback(Output("best-by-ticker","figure"), Input("score-dd","value"))
    def _update_best(score):
        if score is None or score not in df_summary.columns:
            return go.Figure()
        asc = True if score in ("rmse","mae") else False
        dff = df_summary[df_summary.get("split","")=="fixed"].copy()
        if dff.empty:
            return go.Figure()
        dff = dff.sort_values(["ticker",score], ascending=[True,asc])
        best = dff.groupby("ticker", as_index=False).first()
        fig = px.bar(best, x="ticker", y=score, title=f"Best fixed per ticker: {score}")
        fig.update_layout(margin=dict(l=40,r=20,t=50,b=40))
        return fig

    @app.callback(Output("corr-heatmap","figure"), Input("corr-metric","value"))
    def _update_corr(how):
        if df_stats is None or df_stats.empty or how not in df_stats.columns:
            return go.Figure()
        pt = df_stats.pivot_table(index="ticker", columns="keyword", values=how, aggfunc="mean")
        fig = go.Figure(data=go.Heatmap(z=pt.values, x=list(pt.columns), y=list(pt.index),
                                        colorbar_title=how))
        fig.update_layout(title=f"Price vs GT — {how}", margin=dict(l=60,r=40,t=50,b=40))
        return fig

    @app.callback(Output("sensitivity-bars","figure"),
                  Input("corr-metric","value"), Input("shock","value"))
    def _update_sensitivity(how, shock):
        if df_stats is None or df_stats.empty or how not in df_stats.columns:
            return go.Figure()
        k = 0.005
        df = df_stats.copy()
        df["impact_proxy"] = k * df[how].astype(float) * float(shock)
        g = df.groupby(["ticker","keyword"], as_index=False)["impact_proxy"].mean()
        g = g.sort_values("impact_proxy", ascending=False).head(15)
        fig = px.bar(g, x="keyword", y="impact_proxy", color="ticker",
                     title=f"Sensitivity (shock ×{shock})")
        fig.update_layout(margin=dict(l=40,r=20,t=50,b=40), xaxis_tickangle=45)
        return fig

    @app.callback(Output("lag-rmse-line","figure"), Input("ticker-dd","value"))
    def _update_lag_line(tk):
        if tk is None or "lag" not in df_summary.columns or "rmse" not in df_summary.columns:
            return go.Figure()
        prefer_fs = "withGTTECH"
        dff = df_summary[(df_summary.get("split","")=="fixed") & (df_summary.get("feature_set","")==prefer_fs)].copy()
        dff = dff[dff["ticker"]==tk]
        if dff.empty:
            return go.Figure()
        dff = dff.sort_values("lag")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dff["lag"], y=dff["rmse"], mode="lines+markers", name="RMSE"))
        fig.update_layout(title=f"RMSE vs lag — {tk} [{prefer_fs}]",
                          xaxis_title="lag", yaxis_title="RMSE",
                          margin=dict(l=40,r=20,t=50,b=40))
        return fig


    app.run(debug=False, host="0.0.0.0", port=8050)


# ====================== Orchestration ======================
def main(run_dash=False):
    print("STEP — Visualization pipeline")
    df = _load_summary()

    # 1) Overview / Meta
    pivot_metric_heatmap(df, "sharpe_net", "meta_heatmap_sharpe.png")
    pivot_metric_heatmap(df, "rmse",       "meta_heatmap_rmse.png")
    pivot_metric_heatmap(df, "mae",        "meta_heatmap_mae.png")
    best_per_ticker_bar(df, score="rmse", fname="best_by_ticker_rmse.png")
    radar_by_featureset(df)

    # 2) Explainability
    stats_df = load_stats_price_vs_gt()
    if stats_df is not None and not stats_df.empty:
        corr_matrix_price_vs_gt(stats_df, how="pearson",  fname="corr_price_vs_gt_pearson.png")
        corr_matrix_price_vs_gt(stats_df, how="spearman", fname="corr_price_vs_gt_spearman.png")
        scenario_sensitivity_from_corr(stats_df, shock=2.0, how="pearson")

    dff = df[(df.get("split","")=="fixed")].copy()
    if not dff.empty:
        dff = dff.sort_values(["ticker","feature_set","rmse"] if "rmse" in dff.columns else ["ticker","feature_set"],
                              ascending=[True,True,True])
        best_fixed = dff.groupby(["ticker","feature_set"], as_index=False).first()
        plot_perm_importance(best_fixed)

    # 3) Temporal analysis
    plot_lag_comparison(df, prefer_fs="withGTTECH", ticker=None)
    plot_crisis_equity_examples()

    print(f"\nStatic figures saved to: {FIG_DIR}")

    # 4) Interactive dashboard (optional)
    if run_dash:
        launch_dash(df, stats_df if 'stats_df' in locals() else None)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization & Dashboard")
    parser.add_argument("--dash", action="store_true", help="Launch interactive dashboard")
    args = parser.parse_args()
    main(run_dash=args.dash)