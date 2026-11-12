"""
QuantumShield - detection console.

Three views over the deployed system.

  Live detection   classify one flow and see how each ensemble member voted,
                   with the dataset's own label alongside the verdict.
  Batch run        push a stream of real test flows through the API and watch
                   running accuracy, a decision timeline and a confusion
                   matrix build up as they land.
  Evaluation       the hypothesis outcomes, charts built from the recorded
                   metrics, and every figure written to results/.

Run from the project root so results/ and data/ are alongside:

    $env:QS_API = "http://3.254.138.3:5000"
    streamlit run deployment/dashboard.py

On the instance, where the repo is baked into the image:

    nohup streamlit run dashboard.py --server.port 8501 \
        --server.address 0.0.0.0 > streamlit.log 2>&1 &

QS_RESULTS and QS_DATA override the figure and test-set locations.
"""

import os
import re
import json
import glob
from datetime import datetime, timezone

import numpy as np
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

API = os.environ.get("QS_API", "http://localhost:5000")
RESULTS_DIR = os.environ.get("QS_RESULTS", "results")
DATA_DIR = os.environ.get("QS_DATA", os.path.join("data", "processed"))

MODEL_ORDER = ["DNN", "CNN-1D", "LSTM", "HybridQNN"]

# Okabe-Ito, ordered so no weak colour-vision pair sits adjacent.
# Validated: CVD separation dE 11.4, normal-vision floor 18.7.
SERIES = {
    "DNN":       "#0072B2",
    "CNN-1D":    "#009E73",
    "LSTM":      "#E69F00",
    "HybridQNN": "#CC79A7",
}
ATTACK = "#D55E00"
BENIGN = "#009E73"
INK = "#141a20"
MUTED = "#5a6472"
RULE = "#dce1e7"

st.set_page_config(page_title="QuantumShield", layout="wide",
                   initial_sidebar_state="collapsed")


# Streamlit added st.fragment in 1.33, and before that it was
# st.experimental_fragment (1.31 to 1.32). The live-refresh panels use it. On
# older Streamlit the shim below degrades to a one-shot render so the file
# still loads, and the LIVE badges fall back to a manual refresh button. Run
# `pip install --upgrade streamlit` to turn auto-refresh on.
if hasattr(st, "fragment"):
    live_fragment = st.fragment
    LIVE_ENABLED = True
elif hasattr(st, "experimental_fragment"):
    live_fragment = st.experimental_fragment
    LIVE_ENABLED = True
else:
    def live_fragment(run_every=None):
        def wrap(fn):
            return fn
        return wrap
    LIVE_ENABLED = False


# ---------------------------------------------------------------------------
# Streamlit compatibility
#
# st.image only accepted use_container_width from 1.41; before that the
# argument was use_column_width. The instance and the laptop are not always on
# the same version, so ask rather than assume.
# ---------------------------------------------------------------------------

def show_image(path):
    try:
        st.image(path, use_container_width=True)
    except TypeError:
        st.image(path, use_column_width=True)


def show_table(frame, **kwargs):
    try:
        st.dataframe(frame, use_container_width=True, hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(frame, **kwargs)


def show_chart(fig, key=None):
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False}, key=key)


# ---------------------------------------------------------------------------
# Presentation layer
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --ground:   #f4f6f8;
  --panel:    #ffffff;
  --ink:      #141a20;
  --muted:    #5a6472;
  --rule:     #dce1e7;
  --rule-firm:#b9c2cc;
  --attack:   #d55e00;
  --benign:   #009e73;
  --quantum:  #cc79a7;
}

#MainMenu, footer, header { visibility: hidden; }

html, body, [class*="st-"], .stMarkdown, button, input, textarea {
  font-family: 'Archivo', system-ui, sans-serif;
}

.stApp { background: var(--ground); }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1420px; }

/* ---- masthead ---------------------------------------------------------- */
.mast {
  display: flex; align-items: baseline; gap: 1.1rem; flex-wrap: wrap;
  border-bottom: 2px solid var(--ink);
  padding-bottom: .5rem; margin-bottom: 0;
}
.mast h1 {
  font-size: 1.5rem; font-weight: 700; letter-spacing: -.022em;
  color: var(--ink); margin: 0;
}
.mast .sub { font-size: .875rem; color: var(--muted); }
.mast .right {
  margin-left: auto; font-size: .76rem; color: var(--muted);
  display: flex; gap: 1.3rem;
}
.mast .right b {
  font-family: 'IBM Plex Mono', monospace; font-weight: 500;
  color: var(--ink); font-size: .78rem;
}
.mast .live::before {
  content: ''; display: inline-block; width: 7px; height: 7px;
  border-radius: 50%; background: var(--benign); margin-right: .4rem;
  vertical-align: middle;
}
.mast .down::before { background: var(--attack); }

/* ---- headline figures -------------------------------------------------- */
.kpiband {
  display: grid; grid-template-columns: repeat(5, 1fr);
  border: 1px solid var(--rule); border-top: none;
  background: var(--panel); margin-bottom: 1.1rem;
}
.kpi { padding: .95rem 1.15rem 1rem; border-left: 1px solid var(--rule); }
.kpi:first-child { border-left: none; }
.kpi .n {
  font-family: 'IBM Plex Mono', monospace; font-weight: 500;
  font-size: 1.75rem; line-height: 1; color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.kpi .n small { font-size: .9rem; font-weight: 400; color: var(--muted); }
.kpi .l { font-size: .75rem; color: var(--ink); margin-top: .42rem; }
.kpi .s { font-size: .69rem; color: var(--muted); margin-top: .1rem; }
.kpi.live { position: relative; }
.kpi.live::after {
  content: ''; position: absolute; top: .95rem; right: 1rem;
  width: 8px; height: 8px; border-radius: 50%; background: var(--benign);
  box-shadow: 0 0 0 0 rgba(0, 158, 115, .55);
  animation: qs-pulse 2.4s ease-out infinite;
}
@keyframes qs-pulse {
  0%   { box-shadow: 0 0 0 0 rgba(0, 158, 115, .55); }
  70%  { box-shadow: 0 0 0 10px rgba(0, 158, 115, 0); }
  100% { box-shadow: 0 0 0 0 rgba(0, 158, 115, 0); }
}

.livedot {
  display: inline-block; margin-left: .5rem; padding: 0 .4rem;
  font-family: 'IBM Plex Mono', monospace; font-size: .62rem; font-weight: 500;
  text-transform: uppercase; letter-spacing: .05em; color: var(--benign);
  border: 1px solid var(--benign); vertical-align: middle;
}
.livedot::before {
  content: ''; display: inline-block; width: 6px; height: 6px;
  border-radius: 50%; background: var(--benign); margin-right: .35rem;
  vertical-align: middle; animation: qs-pulse 2.4s ease-out infinite;
}

/* ---- tabs -------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {
  gap: 1.7rem; border-bottom: 1px solid var(--rule); margin-bottom: 1.2rem;
}
.stTabs [data-baseweb="tab"] {
  padding: .5rem 0 .6rem; font-size: .88rem; color: var(--muted);
  background: transparent;
}
.stTabs [aria-selected="true"] { color: var(--ink); font-weight: 600; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---- panels ------------------------------------------------------------ */
.panel {
  background: var(--panel); border: 1px solid var(--rule);
  padding: 1.05rem 1.25rem 1.15rem;
}
.panel.tight { padding-bottom: .6rem; }
.panel h2 {
  font-size: .8rem; font-weight: 600; color: var(--ink); margin: 0 0 .15rem;
}
.panel .note { font-size: .77rem; color: var(--muted); margin: 0; line-height: 1.45; }
.panel .note + .note { margin-top: .5rem; }

.caphead {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: .8rem; border-bottom: 1px solid var(--rule);
  padding: 0 0 .45rem; margin: 0 0 .7rem;
}
.caphead b { font-size: .8rem; font-weight: 600; color: var(--ink); }
.caphead span { font-size: .72rem; color: var(--muted); }

/* ---- the flow under test ----------------------------------------------- */
.specimen {
  border: 1px solid var(--rule-firm); background: #fbfcfd;
  padding: .7rem .85rem; margin: .55rem 0 .6rem;
}
.specimen .id {
  font-family: 'IBM Plex Mono', monospace; font-size: .79rem; color: var(--ink);
}
.specimen .truth { font-size: .73rem; color: var(--muted); margin-top: .2rem; }

/* ---- the verdict readout ----------------------------------------------- */
.readout {
  border: 1px solid var(--rule); background: var(--panel);
  padding: 1.15rem 1.3rem 1.3rem;
}
.verdict { display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem; }
.verdict .word {
  font-size: 2.6rem; font-weight: 700; letter-spacing: -.03em; line-height: 1;
}
.verdict .figure {
  font-family: 'IBM Plex Mono', monospace; font-size: 2.1rem; font-weight: 400;
  color: var(--ink); font-variant-numeric: tabular-nums; line-height: 1;
}
.verdict .cap { font-size: .73rem; color: var(--muted); margin-top: .3rem; }

.scored {
  margin-top: .85rem; padding: .45rem .75rem; font-size: .8rem;
  border-left: 3px solid var(--rule-firm); background: #fbfcfd;
}
.scored.hit  { border-left-color: var(--benign); }
.scored.miss { border-left-color: var(--attack); }

.vote { margin-top: 1.15rem; border-top: 1px solid var(--rule); padding-top: 1rem; }
.vote .hdr {
  display: flex; justify-content: space-between;
  font-size: .73rem; color: var(--muted); margin-bottom: .65rem;
}
.row { display: grid; grid-template-columns: 118px 1fr 70px; align-items: center;
       gap: .75rem; margin-bottom: .5rem; }
.row .name { font-size: .83rem; color: var(--ink); display: flex; align-items: center; gap: .4rem; }
.row .wt {
  font-family: 'IBM Plex Mono', monospace; font-size: .66rem; color: var(--muted);
  border: 1px solid var(--rule-firm); padding: 0 .3rem; line-height: 1.3;
}
.track { position: relative; height: 15px; background: #eef1f4; }
.fill  { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 0 3px 3px 0; }
.thr   { position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--ink); }
.row .val {
  font-family: 'IBM Plex Mono', monospace; font-size: .78rem; text-align: right;
  color: var(--ink); font-variant-numeric: tabular-nums;
}
.thrkey { font-size: .71rem; color: var(--muted); margin-top: .8rem;
          display: flex; align-items: center; gap: .4rem; }
.thrkey i { display: inline-block; width: 2px; height: 11px; background: var(--ink); }

.spread {
  margin-top: .85rem; font-size: .76rem; color: var(--muted);
  border-top: 1px dashed var(--rule); padding-top: .7rem;
}
.spread b { color: var(--ink); font-weight: 600; }

.empty { color: var(--muted); font-size: .84rem; padding: 2.6rem 1rem; text-align: center;
         border: 1px dashed var(--rule-firm); line-height: 1.6; }

/* ---- counters ---------------------------------------------------------- */
.counts { display: flex; gap: 2.1rem; flex-wrap: wrap; margin-bottom: .2rem; }
.counts div span {
  display: block; font-family: 'IBM Plex Mono', monospace; font-size: 1.45rem;
  color: var(--ink); font-variant-numeric: tabular-nums; line-height: 1.1;
}
.counts div small { font-size: .71rem; color: var(--muted); }

/* ---- section bands ----------------------------------------------------- */
.secthead {
  border-top: 2px solid var(--ink); padding-top: .5rem;
  margin: 1.8rem 0 .9rem; display: flex; align-items: baseline;
  justify-content: space-between; gap: 1rem;
}
.secthead h3 { font-size: .95rem; font-weight: 600; margin: 0; color: var(--ink); }
.secthead span { font-size: .75rem; color: var(--muted); }
.subhead {
  border-top: 1px solid var(--rule); padding-top: .45rem;
  margin: 1.1rem 0 .7rem; display: flex; align-items: baseline;
  justify-content: space-between; gap: 1rem;
}
.subhead h4 { font-size: .83rem; font-weight: 600; margin: 0; color: var(--ink); }
.subhead span { font-size: .72rem; color: var(--muted); }

.figcap {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: .8rem; border-bottom: 1px solid var(--rule);
  padding-bottom: .38rem; margin-bottom: .55rem;
}
.figcap b { font-size: .82rem; font-weight: 600; color: var(--ink); }
.figcap code {
  font-family: 'IBM Plex Mono', monospace; font-size: .67rem;
  color: var(--muted); background: none;
}

.verdictline { font-size: .83rem; color: var(--ink); padding: .45rem 0;
               border-bottom: 1px solid var(--rule); display: flex;
               justify-content: space-between; gap: 1rem; }
.verdictline:last-child { border-bottom: none; }
.verdictline em { font-style: normal; color: var(--muted); font-size: .77rem; }
.tag { font-family: 'IBM Plex Mono', monospace; font-size: .7rem;
       border: 1px solid var(--rule-firm); padding: .05rem .35rem; white-space: nowrap; }
.tag.yes { color: var(--benign); border-color: var(--benign); }
.tag.no  { color: var(--attack); border-color: var(--attack); }

/* ---- streamlit widget tidying ------------------------------------------ */
.stTextArea textarea {
  font-family: 'IBM Plex Mono', monospace !important; font-size: .76rem !important;
  background: #fbfcfd !important; border-color: var(--rule) !important;
  color: var(--ink) !important;
}
.stButton button {
  border-radius: 0; border: 1px solid var(--rule-firm); font-size: .8rem;
  font-weight: 500; color: var(--ink); background: var(--panel);
}
.stButton button:hover { border-color: var(--ink); background: #f8fafb; }
.stButton button[kind="primary"] {
  background: var(--ink); color: #fff; border-color: var(--ink);
}
.stButton button[kind="primary"]:hover { background: #2a333d; color: #fff; }
.stRadio [role="radiogroup"] { gap: 1.05rem; }
.stRadio label { font-size: .82rem !important; }
.stProgress > div > div > div { background-color: var(--ink); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers. None of these raise into the UI.
# ---------------------------------------------------------------------------

def api_get(path, **params):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=20)
        r.raise_for_status()
        return r.json(), None
    except Exception as exc:
        return None, str(exc)


def api_post(path, payload):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=60)
        if r.status_code == 400:
            return None, r.json().get("error", "the request was rejected")
        r.raise_for_status()
        return r.json(), None
    except Exception as exc:
        return None, str(exc)


def classify(values):
    return api_post("/predict", {"features": [float(v) for v in values]})


# ---------------------------------------------------------------------------
# Disk: the held-out test set, the recorded metrics, the figures
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_test_set(dataset, n_components):
    folder = os.path.join(DATA_DIR, dataset)
    x_path = os.path.join(folder, "X_test.npy")
    y_path = os.path.join(folder, "yb_test.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return None, None, folder
    try:
        return np.load(x_path)[:, :n_components], np.load(y_path), folder
    except Exception:
        return None, None, folder


@st.cache_data(show_spinner=False)
def load_hypotheses(root):
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "*hypotheses*.json"),
                                 recursive=True)):
        name = os.path.basename(path).replace("_hypotheses.json", "")
        try:
            with open(path) as f:
                out[name] = json.load(f)
        except Exception:
            continue
    return out


def metric_frame(payload):
    """per_model dict -> tidy frame of whatever numeric fields are present."""
    per_model = payload.get("per_model")
    if not isinstance(per_model, dict):
        return None
    rows = []
    for model, stats in per_model.items():
        if not isinstance(stats, dict):
            continue
        row = {"model": model}
        for k, v in stats.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                row[k.lower()] = float(v)
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def pick(frame, *candidates):
    for c in candidates:
        if c in frame.columns:
            return c
    return None


SECTIONS = [
    ("Model comparison",      ("comparison", "compare", "performance", "metric",
                               "benchmark", "summary", "params", "parameter")),
    ("ROC and precision-recall", ("roc", "auroc", "auc", "precision_recall",
                                  "pr_curve", "prcurve")),
    ("Confusion matrices",    ("confusion", "cm_", "_cm")),
    ("Training behaviour",    ("training", "history", "loss", "epoch",
                               "convergence", "learning")),
    ("Explainability",        ("shap", "explain", "importance", "feature",
                               "attribution")),
    ("Quantum circuit",       ("quantum", "qnn", "circuit", "qubit", "bloch",
                               "entangle", "ansatz")),
    ("Statistical testing",   ("mcnemar", "significance", "pvalue", "p_value",
                               "hypothes", "test")),
    ("Calibration",           ("calibration", "reliability", "brier", "ece")),
]

ACRONYMS = {
    "roc": "ROC", "auc": "AUC", "auroc": "AUROC", "pr": "PR", "shap": "SHAP",
    "pca": "PCA", "dnn": "DNN", "cnn": "CNN", "lstm": "LSTM", "qnn": "QNN",
    "cm": "confusion matrix", "ece": "ECE", "tpr": "TPR", "fpr": "FPR",
    "mcnemar": "McNemar", "ids": "IDS", "cicids2017": "CICIDS2017",
    "ciciot2023": "CICIoT2023", "hybridqnn": "HybridQNN", "1d": "1D",
}


def prettify(stem):
    words = re.split(r"[_\-\s]+", stem)
    out = [ACRONYMS.get(w.lower(), w if w.isupper() else w.lower()) for w in words]
    label = " ".join(x for x in out if x)
    return label[:1].upper() + label[1:]


@st.cache_data(show_spinner=False)
def discover_figures(root):
    paths = []
    for ext in ("png", "jpg", "jpeg"):
        paths += glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
    paths = sorted(set(paths))

    grouped = {title: [] for title, _ in SECTIONS}
    grouped["Other figures"] = []
    for path in paths:
        key = os.path.basename(path).lower()
        for title, needles in SECTIONS:
            if any(n in key for n in needles):
                grouped[title].append(path)
                break
        else:
            grouped["Other figures"].append(path)
    return {k: v for k, v in grouped.items() if v}, len(paths)


# ---------------------------------------------------------------------------
# Chart furniture, so every plot in the console reads as one system
# ---------------------------------------------------------------------------

def style(fig, height=300, right=12, top=10, left=0, bottom=0):
    fig.update_layout(
        height=height, margin=dict(l=left, r=right, t=top, b=bottom),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", showlegend=False,
        font=dict(family="Archivo", color=INK, size=11),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=RULE,
                        font=dict(family="Archivo", color=INK, size=11)),
    )
    fig.update_xaxes(gridcolor="#eef1f4", zeroline=False, color=MUTED,
                     linecolor=RULE)
    fig.update_yaxes(gridcolor="#eef1f4", zeroline=False, color=MUTED,
                     linecolor=RULE)
    return fig


def ranked_bars(labels, values, colour="#0072B2", fmt="{:.4f}", height=330):
    """Horizontal ranking, biggest at top, every bar labelled with its value."""
    fig = go.Figure(go.Bar(
        x=values[::-1], y=labels[::-1], orientation="h",
        marker=dict(color=colour, line=dict(color="#ffffff", width=2)),
        text=[fmt.format(v) for v in values[::-1]], textposition="outside",
        textfont=dict(color=INK, size=11, family="IBM Plex Mono"),
        hovertemplate="%{y}<br>%{x}<extra></extra>", width=0.62,
        cliponaxis=False,
    ))
    style(fig, height=height, right=72)
    fig.update_xaxes(showticklabels=False, range=[0, max(values) * 1.22])
    fig.update_yaxes(color=INK, gridcolor="#ffffff")
    fig.update_layout(bargap=0.3)
    return fig


def dot_ranking(labels, values, height=300, fmt="{:.4f}"):
    """
    For metrics bunched near 1.0, where a bar chart from zero renders every
    model the same length. A dot carries no length-from-zero reading, so the
    axis can be zoomed onto the spread without misleading anyone. The axis
    labels stay on, and the caption says the scale is zoomed.
    """
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-6)
    x0, x1 = lo - span * 0.4, hi + span * 0.5

    fig = go.Figure()
    for lab, v in zip(labels, values):
        fig.add_shape(type="line", x0=x0, x1=v, y0=lab, y1=lab,
                      line=dict(color="#e4e9ee", width=1.5))
    fig.add_trace(go.Scatter(
        x=values[::-1], y=labels[::-1], mode="markers+text",
        marker=dict(size=13, color="#0072B2",
                    line=dict(color="#ffffff", width=2)),
        # Leading space keeps the label clear of the marker's white halo.
        text=[" " + fmt.format(v) for v in values[::-1]],
        textposition="middle right",
        textfont=dict(family="IBM Plex Mono", size=11, color=INK),
        hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
    ))
    style(fig, height=height, right=26, left=4)
    fig.update_xaxes(range=[x0, x1], tickformat=".3f", showgrid=True)
    fig.update_yaxes(color=INK, gridcolor="#ffffff",
                     categoryorder="array", categoryarray=labels[::-1])
    return fig


def confusion_fig(tp, fp, fn, tn, height=260):
    """Predicted across, actual down. Counts drawn on the cells."""
    z = [[tn, fp], [fn, tp]]
    peak = max(1, max(max(r) for r in z))
    fig = go.Figure(go.Heatmap(
        z=z, x=["predicted benign", "predicted attack"],
        y=["actual benign", "actual attack"],
        # Not pure white at the floor, or an empty cell disappears into the
        # panel and the matrix stops reading as a grid.
        colorscale=[[0, "#eef2f6"], [1, "#0072B2"]], zmin=0, zmax=peak,
        showscale=False, xgap=3, ygap=3,
        hovertemplate="%{y}, %{x}<br>%{z} flows<extra></extra>",
    ))
    for r, row in enumerate(z):
        for c, v in enumerate(row):
            fig.add_annotation(
                x=c, y=r, text=f"<b>{v}</b>", showarrow=False,
                font=dict(family="IBM Plex Mono", size=17,
                          color="#ffffff" if v > peak * 0.55 else INK))
    style(fig, height=height, left=4)
    fig.update_xaxes(side="top", showgrid=False, color=MUTED)
    fig.update_yaxes(showgrid=False, color=MUTED, autorange="reversed")
    return fig


def timeline_fig(frame, threshold, height=300):
    """Each classified flow against the threshold, marked right or wrong."""
    fig = go.Figure()
    fig.add_hrect(y0=threshold, y1=1.02, fillcolor=ATTACK, opacity=0.045,
                  line_width=0)
    fig.add_hline(y=threshold, line=dict(color=INK, width=1.5, dash="dot"),
                  annotation_text=f"threshold {threshold:.3f}",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color=MUTED, family="Archivo"))

    has_truth = "actual" in frame.columns
    for correct, colour, name in (
            (True, "#0072B2", "correct"), (False, ATTACK, "wrong")):
        part = (frame[frame["correct"] == correct] if has_truth
                else (frame if correct else frame.iloc[0:0]))
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["n"], y=part["p"], mode="markers", name=name,
            marker=dict(size=9, color=colour, line=dict(color="#fff", width=1),
                        symbol=["circle" if a == "BENIGN" else "diamond"
                                for a in part.get("actual", part["verdict"])]),
            customdata=np.stack([part["verdict"],
                                 part.get("actual", part["verdict"])], axis=-1),
            hovertemplate=("flow %{x}<br>P(attack) %{y:.4f}"
                           "<br>called %{customdata[0]}"
                           "<br>actually %{customdata[1]}<extra></extra>"),
        ))
    style(fig, height=height, left=4, top=18)
    fig.update_xaxes(title_text="flow, in the order sent",
                     title_font=dict(size=10, color=MUTED))
    fig.update_yaxes(range=[-0.03, 1.05], title_text="P(attack)",
                     title_font=dict(size=10, color=MUTED))
    return fig


# ---------------------------------------------------------------------------
# Masthead and headline figures
# ---------------------------------------------------------------------------

health, health_err = api_get("/health")

if health:
    threshold = float(health["decision_threshold"])
    weights = health["ensemble_weights"]
    dataset = health["dataset"]
    n_components = int(health["expects"]["or_pca_components"])
    status = (f'<span class="live">serving</span>'
              f'<span>corpus <b>{dataset}</b></span>'
              f'<span>threshold <b>{threshold:.3f}</b></span>')
else:
    threshold, weights = 0.5, {m: 1 for m in MODEL_ORDER}
    dataset, n_components = "cicids2017", 16
    status = '<span class="live down">service unreachable</span>'

st.markdown(
    f'<div class="mast"><h1>QuantumShield</h1>'
    f'<span class="sub">hybrid quantum-classical intrusion detection</span>'
    f'<span class="right">{status}</span></div>',
    unsafe_allow_html=True,
)

hypotheses = load_hypotheses(RESULTS_DIR)
primary = hypotheses.get(dataset) or (next(iter(hypotheses.values()))
                                      if hypotheses else None)
metrics = metric_frame(primary) if primary else None

# The report-evidence tiles are fixed once the metrics are read.
_static_tiles = []
if metrics is not None:
    f1_col = pick(metrics, "f1", "f1_score", "f1score")
    auc_col = pick(metrics, "auroc", "auc", "roc_auc")
    par_col = pick(metrics, "params", "parameters", "n_params",
                   "trainable_params")

    if f1_col:
        best = metrics.loc[metrics[f1_col].idxmax()]
        _static_tiles.append((f"{best[f1_col]:.4f}", "best F1",
                              str(best["model"])))
    if auc_col:
        best = metrics.loc[metrics[auc_col].idxmax()]
        _static_tiles.append((f"{best[auc_col]:.4f}", "best AUROC",
                              str(best["model"])))
    if par_col:
        quantum = metrics[metrics["model"].str.contains("qnn|quantum",
                                                        case=False, na=False)]
        classical = metrics[~metrics["model"].str.contains(
            "qnn|quantum|ensemble", case=False, na=False)]
        if not quantum.empty:
            q = float(quantum.iloc[0][par_col])
            _static_tiles.append((f"{q:,.0f}", "quantum parameters",
                                  str(quantum.iloc[0]["model"])))
            if not classical.empty and q > 0:
                heaviest = classical.loc[classical[par_col].idxmax()]
                _static_tiles.append((
                    f"{float(heaviest[par_col]) / q:,.0f}<small> x</small>",
                    "smaller than the largest classical model",
                    f"{heaviest['model']} at {heaviest[par_col]:,.0f}"))


# This band and the detection-history panel below are Streamlit fragments:
# they re-run on their own schedule without touching the rest of the page,
# so flows landing in DynamoDB from any client show up here without anyone
# clicking refresh, and whatever the user was typing or picking elsewhere
# stays put.
@live_fragment(run_every="4s")
def kpi_band():
    today, _ = api_get(
        "/recent",
        day=datetime.now(timezone.utc).strftime("%Y-%m-%d"), n=500)
    live = (f"{today['count']:,}" if today else "0", "classified today",
            "written to DynamoDB  ·  live")
    tiles = _static_tiles + [live]
    if tiles:
        st.markdown(
            '<div class="kpiband" style="grid-template-columns:'
            'repeat(%d,1fr)">%s</div>' % (len(tiles), "".join(
                f'<div class="kpi{"" if i < len(tiles) - 1 else " live"}">'
                f'<div class="n">{n}</div>'
                f'<div class="l">{label}</div>'
                f'<div class="s">{sub}</div></div>'
                for i, (n, label, sub) in enumerate(tiles))),
            unsafe_allow_html=True)


kpi_band()

tab_live, tab_batch, tab_eval = st.tabs(
    ["Live detection", "Batch run", "Evaluation"])


# ---------------------------------------------------------------------------
# Shared: the verdict readout
# ---------------------------------------------------------------------------

def vote_readout(result, truth=None):
    p = float(result["attack_probability"])
    verdict = result["prediction"]
    colour = ATTACK if verdict == "ATTACK" else BENIGN
    per_model = result["per_model"]
    probs = [float(per_model[m]) for m in MODEL_ORDER]

    rows = "".join(
        f'<div class="row">'
        f'  <div class="name">{m}<span class="wt">w{weights[m]}</span></div>'
        f'  <div class="track">'
        f'    <div class="fill" style="width:{max(v, 0.004) * 100:.2f}%;'
        f'background:{SERIES[m]}"></div>'
        f'    <div class="thr" style="left:{threshold * 100:.2f}%"></div>'
        f'  </div>'
        f'  <div class="val">{v:.4f}</div>'
        f'</div>'
        for m, v in zip(MODEL_ORDER, probs)
    )

    scored = ""
    if truth in ("ATTACK", "BENIGN"):
        hit = truth == verdict
        scored = (f'<div class="scored {"hit" if hit else "miss"}">'
                  f'Ground truth is <b>{truth}</b>. The ensemble '
                  f'{"agrees" if hit else "is wrong on this flow"}.</div>')

    # How far apart the members are is the interesting signal: a verdict the
    # weights had to resolve is weaker evidence than one they all agreed on.
    voting_attack = sum(v >= threshold for v in probs)
    if voting_attack in (0, len(probs)):
        agreement = (f'All four models agree, with a spread of '
                     f'<b>{max(probs) - min(probs):.4f}</b>.')
    else:
        agreement = (f'<b>{voting_attack} of 4</b> models crossed the threshold, '
                     f'so the weights decided this verdict. Spread '
                     f'<b>{max(probs) - min(probs):.4f}</b>.')

    st.markdown(
        f'<div class="readout">'
        f'  <div class="verdict">'
        f'    <div><div class="word" style="color:{colour}">{verdict}</div>'
        f'      <div class="cap">weighted vote of four models</div></div>'
        f'    <div style="text-align:right">'
        f'      <div class="figure">{p:.6f}</div>'
        f'      <div class="cap">against a threshold of {threshold:.3f}</div>'
        f'    </div>'
        f'  </div>'
        f'  {scored}'
        f'  <div class="vote">'
        f'    <div class="hdr"><span>each model, voting independently</span>'
        f'<span>P(attack)</span></div>{rows}'
        f'    <div class="thrkey"><i></i>the fitted decision threshold, the '
        f'same one applied during evaluation</div>'
        f'    <div class="spread">{agreement}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


X_test, y_test, data_folder = load_test_set(dataset, n_components)


# ===========================================================================
# Live detection
# ===========================================================================

with tab_live:
    if not health:
        st.markdown(
            f'<div class="panel"><h2>The detection service is unreachable</h2>'
            f'<p class="note">Nothing answered at {API}. Check the container is '
            f'running and that port 5000 is open in the security group.</p>'
            f'<p class="note">{health_err}</p></div>',
            unsafe_allow_html=True)
    else:
        left, right = st.columns([0.92, 1.45], gap="medium")
        single, truth, batch, parse_error = None, None, None, None

        with left:
            st.markdown(
                '<div class="panel tight"><h2>Choose a flow to classify</h2>'
                '<p class="note">The models run on the server; this console '
                'only sends and displays.</p></div>',
                unsafe_allow_html=True)

            modes = ["Test set", "CSV file", "Type values"]
            if X_test is None:
                modes = ["CSV file", "Type values"]
            # Every widget on this tab is explicitly keyed. Without the keys,
            # Streamlit tracks them by their position in the script, and
            # buttons from the other branches of this if/elif/else leak into
            # the active branch when the mode switches.
            mode = st.radio("Source", modes, horizontal=True,
                            key="src_mode", label_visibility="collapsed")

            if mode == "Test set":
                attack_idx = np.where(y_test == 1)[0]
                benign_idx = np.where(y_test == 0)[0]
                if "row" not in st.session_state:
                    st.session_state.row = int(attack_idx[0])

                c1, c2 = st.columns(2)
                if c1.button("Random attack", key="btn_rand_atk",
                             use_container_width=True):
                    st.session_state.row = int(np.random.choice(attack_idx))
                if c2.button("Random benign", key="btn_rand_ben",
                             use_container_width=True):
                    st.session_state.row = int(np.random.choice(benign_idx))

                st.session_state.row = int(st.number_input(
                    "Row index", 0, len(X_test) - 1,
                    int(st.session_state.row), 1, key="num_row"))

                i = st.session_state.row
                truth = "ATTACK" if int(y_test[i]) == 1 else "BENIGN"
                st.markdown(
                    f'<div class="specimen"><div class="id">{dataset} · '
                    f'test row {i:,} of {len(X_test):,}</div>'
                    f'<div class="truth">labelled in the dataset as '
                    f'<b style="color:{ATTACK if truth == "ATTACK" else BENIGN}">'
                    f'{truth}</b></div></div>',
                    unsafe_allow_html=True)

                if st.button("Classify this flow", type="primary",
                             key="btn_classify_testset",
                             use_container_width=True):
                    single, parse_error = classify(X_test[i])
                    st.session_state.last = (single, truth)

            elif mode == "CSV file":
                st.markdown(
                    f'<p class="note">One flow per row, {n_components} numeric '
                    f'columns. An optional label column (1 for attack) is '
                    f'scored against the predictions.</p>',
                    unsafe_allow_html=True)
                upload = st.file_uploader("CSV", type=["csv"],
                                          key="csv_upload",
                                          label_visibility="collapsed")
                limit = st.number_input("Rows to classify", 1, 200, 20, 1,
                                        key="num_csv_limit")

                if upload is not None and st.button(
                        "Classify all rows", type="primary",
                        key="btn_classify_csv",
                        use_container_width=True):
                    try:
                        frame = pd.read_csv(upload)
                        labels = None
                        for col in ("label", "Label", "y", "target"):
                            if col in frame.columns:
                                labels = frame[col].values
                                frame = frame.drop(columns=[col])
                                break
                        values = frame.select_dtypes("number").values[
                            :int(limit), :n_components]
                        batch, bar = [], st.progress(0.0)
                        for n, vec in enumerate(values):
                            out, _ = classify(vec)
                            if out:
                                rec = {"row": n, "verdict": out["prediction"],
                                       "P(attack)": round(
                                           float(out["attack_probability"]), 6)}
                                for m in MODEL_ORDER:
                                    rec[m] = round(float(out["per_model"][m]), 4)
                                if labels is not None and n < len(labels):
                                    rec["actual"] = ("ATTACK" if int(labels[n]) == 1
                                                     else "BENIGN")
                                batch.append(rec)
                            bar.progress((n + 1) / len(values))
                        bar.empty()
                    except Exception as exc:
                        parse_error = f"That file could not be read. {exc}"

            else:
                st.markdown(
                    f'<p class="note">Paste {n_components} comma separated '
                    f'values, already scaled and projected, or the full raw '
                    f'feature vector. Click into the box below and overwrite '
                    f'the zeros with your own flow.</p>',
                    unsafe_allow_html=True)
                if "features" not in st.session_state:
                    st.session_state.features = ", ".join(
                        "0.0" for _ in range(n_components))
                raw = st.text_area("Feature values", key="features",
                                   height=140,
                                   label_visibility="collapsed",
                                   placeholder="0.5, -1.2, 0.3, ...")
                if st.button("Classify", type="primary",
                             key="btn_classify_typed",
                             use_container_width=True):
                    try:
                        vals = [float(v.strip()) for v in
                                raw.replace("\n", ",").split(",") if v.strip()]
                        single, parse_error = classify(vals)
                        st.session_state.last = (single, None)
                    except ValueError:
                        parse_error = ("Those values could not be read. Use "
                                       "comma separated numbers.")

            if parse_error:
                st.markdown(
                    f'<p class="note" style="color:var(--attack)">{parse_error}</p>',
                    unsafe_allow_html=True)
            if X_test is None and mode != "Type values":
                st.markdown(
                    f'<p class="note">No test set at <code>{data_folder}</code>, '
                    f'so that source is hidden. Set QS_DATA if it lives '
                    f'elsewhere.</p>', unsafe_allow_html=True)

        with right:
            if single is None and batch is None and "last" in st.session_state:
                single, truth = st.session_state.last

            if single is not None:
                vote_readout(single, truth)
            elif batch:
                frame = pd.DataFrame(batch)
                n_attack = int((frame["verdict"] == "ATTACK").sum())
                scored = ""
                if "actual" in frame.columns:
                    correct = int((frame["verdict"] == frame["actual"]).sum())
                    scored = (f'<div><span>{correct}/{len(frame)}</span>'
                              f'<small>correct</small></div>')
                st.markdown(
                    f'<div class="readout"><div class="counts">'
                    f'<div><span>{len(frame)}</span><small>classified</small></div>'
                    f'<div><span style="color:{ATTACK}">{n_attack}</span>'
                    f'<small>attack</small></div>'
                    f'<div><span style="color:{BENIGN}">'
                    f'{len(frame) - n_attack}</span><small>benign</small></div>'
                    f'{scored}</div></div>', unsafe_allow_html=True)
                show_table(frame)
            else:
                st.markdown(
                    '<div class="readout"><div class="empty">'
                    'Pick a flow on the left and classify it.<br>'
                    'The verdict, each model\'s independent vote and the '
                    'dataset\'s own label appear here.</div></div>',
                    unsafe_allow_html=True)

        st.write("")
        lower_l, lower_r = st.columns([1, 1.12], gap="medium")

        with lower_l:
            st.markdown(
                '<div class="panel tight"><h2>What the ensemble relies on</h2>'
                '<p class="note">Mean absolute SHAP value across the explained '
                'attack samples, projected from the PCA components back onto '
                'the original network features. This is a global ranking, not '
                'an explanation of one flow.</p></div>',
                unsafe_allow_html=True)

            shap_data, _ = api_get("/explain")
            feats = (shap_data or {}).get("top_features") or []
            if not feats:
                st.markdown('<div class="empty">No SHAP results available '
                            'from the service.</div>', unsafe_allow_html=True)
            else:
                show_chart(ranked_bars([f["feature"] for f in feats],
                                       [float(f["importance"]) for f in feats],
                                       height=345), key="shap_live")

        with lower_r:
            st.markdown(
                '<div class="panel tight"><h2>Detection history <span '
                'class="livedot">live</span></h2>'
                '<p class="note">Every classification is written to DynamoDB, '
                'partitioned by day and sorted by time, so a day reads back '
                'in one query. This panel polls the service every four '
                'seconds, so flows landing from any client appear here '
                'without anyone clicking refresh.</p></div>',
                unsafe_allow_html=True)

            # Widgets live outside the fragment so the user's choice of day
            # or row count survives each refresh.
            d1, d2 = st.columns([1.3, 1])
            d1.date_input("Day", value=datetime.now(timezone.utc).date(),
                          key="hist_day", label_visibility="collapsed")
            d2.number_input("Rows", 10, 500, 50, 10,
                            key="hist_n", label_visibility="collapsed")

            @live_fragment(run_every="4s")
            def history_live():
                day = st.session_state.get(
                    "hist_day", datetime.now(timezone.utc).date())
                n = int(st.session_state.get("hist_n", 50))
                hist, hist_err = api_get("/recent",
                                         day=day.strftime("%Y-%m-%d"), n=n)
                if hist_err:
                    st.markdown(
                        f'<div class="empty">The history could not be '
                        f'read.<br>{hist_err}</div>', unsafe_allow_html=True)
                    return
                if hist["count"] == 0:
                    st.markdown(
                        f'<div class="empty">Nothing has been classified on '
                        f'{day} yet.<br>Flows appear here as they land.'
                        f'</div>', unsafe_allow_html=True)
                    return

                stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
                st.markdown(
                    f'<div class="counts" style="margin-bottom:.8rem;'
                    f'align-items:baseline">'
                    f'<div><span>{hist["count"]}</span>'
                    f'<small>classified</small></div>'
                    f'<div><span style="color:{ATTACK}">{hist["attacks"]}'
                    f'</span><small>attack</small></div>'
                    f'<div><span style="color:{BENIGN}">{hist["benign"]}'
                    f'</span><small>benign</small></div>'
                    f'<div style="margin-left:auto;font-size:.7rem;'
                    f'color:var(--muted);align-self:center;font-family:'
                    f'IBM Plex Mono,monospace">pulled {stamp} UTC</div>'
                    f'</div>', unsafe_allow_html=True)

                items = hist["items"][::-1]
                frame = pd.DataFrame({
                    "n": range(len(items)),
                    "p": [float(i["attack_probability"]) for i in items],
                    "verdict": [i["prediction"] for i in items],
                })
                show_chart(timeline_fig(frame, threshold, height=240),
                           key="hist_timeline")

                table = [{"time": i["ts"][11:19],
                          "verdict": i["prediction"],
                          "P(attack)": round(
                              float(i["attack_probability"]), 6),
                          **{m: round(float(i["per_model"][m]), 4)
                             for m in MODEL_ORDER}}
                         for i in hist["items"]]
                show_table(pd.DataFrame(table),
                           height=min(230, 44 + 36 * len(table)))

            history_live()


# ===========================================================================
# Batch run
# ===========================================================================

with tab_batch:
    if not health or X_test is None:
        st.markdown(
            f'<div class="panel"><h2>Batch runs need the service and the test '
            f'set</h2><p class="note">The service at {API} must be answering, '
            f'and X_test.npy with yb_test.npy must be readable under '
            f'<code>{data_folder}</code>.</p></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="panel tight"><h2>Push real traffic through the '
            'deployed system</h2>'
            '<p class="note">Draws flows at random from the held-out test set '
            'and sends them to the API one at a time, exactly as a client '
            'would. Every one is scored against its dataset label and written '
            'to DynamoDB, so this doubles as a way to populate the detection '
            'history before a demonstration.</p></div>',
            unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
        count = c1.number_input("Flows", 5, 200, 40, 5)
        attack_share = c2.slider("Attack share", 0.0, 1.0, 0.35, 0.05)
        c3.write("")
        go_now = c3.button("Run the batch", type="primary",
                           use_container_width=True)
        c4.write("")
        if c4.button("Clear results", use_container_width=True):
            st.session_state.pop("stream", None)

        if go_now:
            attack_idx = np.where(y_test == 1)[0]
            benign_idx = np.where(y_test == 0)[0]
            n_att = min(int(round(count * attack_share)), len(attack_idx))
            n_ben = min(int(count) - n_att, len(benign_idx))
            chosen = np.concatenate([
                np.random.choice(attack_idx, n_att, replace=False),
                np.random.choice(benign_idx, n_ben, replace=False)])
            np.random.shuffle(chosen)

            bar = st.progress(0.0, text="sending flows")
            live = st.empty()
            rows = []

            for k, i in enumerate(chosen):
                out, _ = classify(X_test[i])
                if out:
                    actual = "ATTACK" if int(y_test[i]) == 1 else "BENIGN"
                    rows.append({
                        "n": len(rows), "row": int(i),
                        "p": float(out["attack_probability"]),
                        "verdict": out["prediction"], "actual": actual,
                        "correct": out["prediction"] == actual,
                        **{m: float(out["per_model"][m]) for m in MODEL_ORDER}})
                bar.progress((k + 1) / len(chosen),
                             text=f"sent {k + 1} of {len(chosen)}")
                # Redraw every few flows rather than every one, so the chart
                # keeps up without the run being dominated by rendering.
                if rows and (k % 4 == 0 or k == len(chosen) - 1):
                    with live.container():
                        hits = sum(r["correct"] for r in rows)
                        st.markdown(
                            f'<div class="counts" style="margin:.6rem 0 .3rem">'
                            f'<div><span>{len(rows)}</span><small>sent</small></div>'
                            f'<div><span>{hits / len(rows):.1%}</span>'
                            f'<small>accuracy so far</small></div></div>',
                            unsafe_allow_html=True)
                        show_chart(timeline_fig(pd.DataFrame(rows), threshold,
                                                height=250),
                                   key=f"stream_{k}")
            bar.empty()
            live.empty()
            st.session_state.stream = rows

        rows = st.session_state.get("stream")
        if not rows:
            st.markdown(
                '<div class="empty" style="margin-top:1rem">No batch has been '
                'run yet.<br>Choose how many flows and what share of them '
                'should be attacks, then run it.</div>',
                unsafe_allow_html=True)
        else:
            frame = pd.DataFrame(rows)
            tp = int(((frame.actual == "ATTACK") & (frame.verdict == "ATTACK")).sum())
            fn = int(((frame.actual == "ATTACK") & (frame.verdict == "BENIGN")).sum())
            fp = int(((frame.actual == "BENIGN") & (frame.verdict == "ATTACK")).sum())
            tn = int(((frame.actual == "BENIGN") & (frame.verdict == "BENIGN")).sum())

            recall = tp / (tp + fn) if tp + fn else 0.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0)
            accuracy = (tp + tn) / len(frame)

            st.markdown(
                '<div class="secthead"><h3>What came back</h3>'
                f'<span>{len(frame)} flows, scored against their dataset labels'
                '</span></div>', unsafe_allow_html=True)

            st.markdown(
                '<div class="kpiband" style="border-top:1px solid var(--rule);'
                'grid-template-columns:repeat(5,1fr)">'
                f'<div class="kpi"><div class="n">{accuracy:.1%}</div>'
                f'<div class="l">accuracy</div>'
                f'<div class="s">{tp + tn} of {len(frame)} correct</div></div>'
                f'<div class="kpi"><div class="n">{recall:.1%}</div>'
                f'<div class="l">attacks caught</div>'
                f'<div class="s">{tp} of {tp + fn}</div></div>'
                f'<div class="kpi"><div class="n">{precision:.1%}</div>'
                f'<div class="l">precision</div>'
                f'<div class="s">{fp} false alarms</div></div>'
                f'<div class="kpi"><div class="n">{f1:.4f}</div>'
                f'<div class="l">F1 on this sample</div>'
                f'<div class="s">harmonic mean of the two</div></div>'
                f'<div class="kpi"><div class="n">{fn}</div>'
                f'<div class="l">attacks missed</div>'
                f'<div class="s">the costly error</div></div></div>',
                unsafe_allow_html=True)

            a, b = st.columns([1.5, 1], gap="medium")
            with a:
                st.markdown('<div class="caphead"><b>Every flow against the '
                            'threshold</b><span>diamonds are real attacks, '
                            'orange is a mistake</span></div>',
                            unsafe_allow_html=True)
                show_chart(timeline_fig(frame, threshold, height=330),
                           key="batch_timeline")
            with b:
                st.markdown('<div class="caphead"><b>Confusion matrix</b>'
                            '<span>this sample only</span></div>',
                            unsafe_allow_html=True)
                show_chart(confusion_fig(tp, fp, fn, tn, height=330),
                           key="batch_cm")

            st.markdown('<div class="subhead"><h4>Where the members '
                        'disagreed</h4><span>mean P(attack) by model, split by '
                        'what the flow really was</span></div>',
                        unsafe_allow_html=True)

            d, e = st.columns([1, 1], gap="medium")
            with d:
                fig = go.Figure()
                for actual, dash in (("ATTACK", False), ("BENIGN", True)):
                    part = frame[frame.actual == actual]
                    if part.empty:
                        continue
                    fig.add_trace(go.Bar(
                        x=MODEL_ORDER,
                        y=[part[m].mean() for m in MODEL_ORDER],
                        marker=dict(
                            color=[SERIES[m] for m in MODEL_ORDER],
                            opacity=0.35 if dash else 1.0,
                            line=dict(color="#ffffff", width=1.5)),
                        text=[f"{part[m].mean():.3f}" for m in MODEL_ORDER],
                        textposition="outside",
                        textfont=dict(family="IBM Plex Mono", size=10, color=INK),
                        hovertemplate=(f"%{{x}} on real {actual.lower()} flows"
                                       "<br>mean %{y:.4f}<extra></extra>"),
                        cliponaxis=False))
                fig.add_hline(y=threshold,
                              line=dict(color=INK, width=1.5, dash="dot"))
                style(fig, height=290, top=22)
                fig.update_yaxes(range=[0, 1.12], title_text="mean P(attack)",
                                 title_font=dict(size=10, color=MUTED))
                fig.update_layout(barmode="group", bargap=0.35)
                show_chart(fig, key="batch_bymodel")
                st.markdown(
                    '<p class="note">Solid is real attacks, faded is real '
                    'benign traffic. A model whose two bars sit far apart is '
                    'separating the classes well on this sample.</p>',
                    unsafe_allow_html=True)

            with e:
                agree = frame[[*MODEL_ORDER]].ge(threshold).sum(axis=1)
                counts = [int((agree == k).sum()) for k in range(5)]
                fig = go.Figure(go.Bar(
                    x=[f"{k} of 4" for k in range(5)], y=counts,
                    marker=dict(color=["#b9c2cc", "#9aa6b4", "#7e8c9c",
                                       "#4a7fa5", "#0072B2"],
                                line=dict(color="#ffffff", width=1.5)),
                    text=counts, textposition="outside",
                    textfont=dict(family="IBM Plex Mono", size=11, color=INK),
                    hovertemplate="%{x} models voted attack<br>%{y} flows"
                                  "<extra></extra>",
                    width=0.6, cliponaxis=False))
                style(fig, height=290, top=22)
                fig.update_yaxes(title_text="flows",
                                 title_font=dict(size=10, color=MUTED))
                show_chart(fig, key="batch_agree")
                split = sum(counts[1:4])
                st.markdown(
                    f'<p class="note">How many of the four crossed the '
                    f'threshold on each flow. <b>{split}</b> of {len(frame)} '
                    f'were split decisions the weights had to resolve, which '
                    f'is where the ensemble earns its place.</p>',
                    unsafe_allow_html=True)

            st.markdown('<div class="subhead"><h4>Every flow</h4>'
                        '<span>sorted by how confident the ensemble was</span>'
                        '</div>', unsafe_allow_html=True)
            table = frame.sort_values("p", ascending=False)[
                ["row", "actual", "verdict", "p", *MODEL_ORDER]].round(4)
            table.columns = ["test row", "actual", "predicted", "P(attack)",
                             *MODEL_ORDER]
            show_table(table, height=330)


# ===========================================================================
# Evaluation
# ===========================================================================

with tab_eval:
    figures, n_figures = discover_figures(RESULTS_DIR)

    if not figures and not hypotheses:
        st.markdown(
            f'<div class="panel"><h2>No results found</h2>'
            f'<p class="note">Nothing was found in <code>'
            f'{os.path.abspath(RESULTS_DIR)}</code>. Start Streamlit from the '
            f'project root so the folder is alongside, or set QS_RESULTS to '
            f'point at it.</p></div>', unsafe_allow_html=True)
    else:
        for name, payload in hypotheses.items():
            st.markdown(
                f'<div class="secthead"><h3>{prettify(name)}</h3>'
                f'<span>from {name}_hypotheses.json</span></div>',
                unsafe_allow_html=True)

            claims = {k: v for k, v in payload.items()
                      if re.fullmatch(r"H\d+", k, flags=re.I)}
            if claims:
                lines = []
                for key in sorted(claims, key=lambda s: int(s[1:])):
                    body = claims[key]
                    if isinstance(body, dict):
                        text = (body.get("statement") or body.get("claim")
                                or body.get("description") or "")
                        outcome = str(body.get("supported",
                                               body.get("result",
                                                        body.get("outcome", ""))))
                        extra = " ".join(f"{k} {v}" for k, v in body.items()
                                         if k in ("p_value", "pvalue", "p",
                                                  "margin", "delta"))
                    else:
                        text, outcome, extra = str(body), "", ""

                    low = outcome.lower()
                    if low in ("true", "supported", "yes"):
                        cls, word = "yes", "supported"
                    elif low in ("false", "not supported", "no", "rejected"):
                        cls, word = "no", "not supported"
                    else:
                        cls, word = "", outcome
                    lines.append(
                        f'<div class="verdictline"><span><b>{key.upper()}</b> '
                        f'{text}{" <em>" + extra + "</em>" if extra else ""}'
                        f'</span>'
                        f'{f"<span class=tag {cls}>{word}</span>" if word else ""}'
                        f'</div>')
                st.markdown(f'<div class="panel">{"".join(lines)}</div>',
                            unsafe_allow_html=True)

            frame = metric_frame(payload)
            if frame is not None and len(frame):
                f1_col = pick(frame, "f1", "f1_score", "f1score")
                auc_col = pick(frame, "auroc", "auc", "roc_auc")
                par_col = pick(frame, "params", "parameters", "n_params",
                               "trainable_params")

                charts = [c for c in (f1_col, auc_col, par_col) if c]
                if charts:
                    cols = st.columns(len(charts), gap="medium")
                    for col, metric in zip(cols, charts):
                        part = frame.dropna(subset=[metric]).sort_values(
                            metric, ascending=False)
                        with col:
                            if metric == par_col:
                                st.markdown(
                                    '<div class="caphead"><b>Trainable '
                                    'parameters</b><span>log scale</span></div>',
                                    unsafe_allow_html=True)
                                part = part.sort_values(metric, ascending=False)
                                fig = go.Figure(go.Bar(
                                    x=part[metric][::-1],
                                    y=part["model"][::-1], orientation="h",
                                    marker=dict(
                                        color=[SERIES.get(m, "#8b97a5")
                                               for m in part["model"][::-1]],
                                        line=dict(color="#fff", width=2)),
                                    text=[f"{v:,.0f}"
                                          for v in part[metric][::-1]],
                                    textposition="outside",
                                    textfont=dict(family="IBM Plex Mono",
                                                  size=11, color=INK),
                                    hovertemplate="%{y}<br>%{x:,.0f} parameters"
                                                  "<extra></extra>",
                                    width=0.6, cliponaxis=False))
                                style(fig, height=300, right=78)
                                fig.update_xaxes(type="log",
                                                 showticklabels=False)
                                fig.update_yaxes(color=INK, gridcolor="#fff")
                                show_chart(fig, key=f"{name}_params")
                            else:
                                st.markdown(
                                    f'<div class="caphead"><b>{metric.upper()}'
                                    f'</b><span>higher is better, scale '
                                    f'zoomed</span></div>',
                                    unsafe_allow_html=True)
                                show_chart(dot_ranking(
                                    list(part["model"]), list(part[metric]),
                                    height=300), key=f"{name}_{metric}")

                show_table(frame.round(4))

        st.markdown(
            f'<div class="secthead" style="margin-bottom:.2rem">'
            f'<h3>Figures</h3><span>{n_figures} found in {RESULTS_DIR}/</span>'
            f'</div>', unsafe_allow_html=True)

        for title, paths in figures.items():
            st.markdown(
                f'<div class="subhead"><h4>{title}</h4>'
                f'<span>{len(paths)}</span></div>', unsafe_allow_html=True)
            for i in range(0, len(paths), 2):
                cols = st.columns(2, gap="medium")
                for col, path in zip(cols, paths[i:i + 2]):
                    stem = os.path.splitext(os.path.basename(path))[0]
                    with col:
                        st.markdown(
                            f'<div class="figcap"><b>{prettify(stem)}</b>'
                            f'<code>{os.path.relpath(path, RESULTS_DIR)}</code>'
                            f'</div>', unsafe_allow_html=True)
                        show_image(path)