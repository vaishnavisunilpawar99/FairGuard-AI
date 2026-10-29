import os, sys, joblib, tempfile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 72, "savefig.dpi": 72})

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import requests
from src import (data_loader, models, fairness_engine,
                 xai_engine, drift_engine, privacy_engine, model_card, governance)
from src.services import drift_service, privacy_service, risk_service
from src.database import crud


st.set_page_config(
    page_title="FairGuard AI",
    layout="wide",
    page_icon="⚖️",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "FairGuard AI — Responsible AI Assessment Workbench",
    },
)

# ─── hide Streamlit Deploy button & top-right toolbar ─────────────────────────
st.markdown(
    """
    <style>
    /* Hide Deploy button and running indicator */
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    #MainMenu {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important;}
    header[data-testid="stHeader"] {visibility: hidden !important; height: 0 !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stDeployButton"] {display: none !important;}
    button[kind="header"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

SC = {"Pass": "🟢", "Warning": "🟡", "Fail": "🔴", "N/A": "⚪"}
SAMPLE_N = 200
ATTRS    = ["sex", "race", "age_bin"]

# ─── helpers ──────────────────────────────────────────────────────────────────
def badge(metric, val):
    s = fairness_engine.status(metric, val)
    return f"{SC[s]} {s}"

def fairness_summary(y_true, preds, s_col):
    return fairness_engine.audit(y_true, preds, s_col)

def show_metrics_row(met, fa):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", met["accuracy"])
    c2.metric("AUC",      met["auc"])
    c3.metric("F1",       met["f1"])
    c4.metric(badge("demographic_parity_diff", fa["demographic_parity_diff"]) + " DPD",
              f"{fa['demographic_parity_diff']:.4f}")
    c5.metric(badge("equalized_odds_diff",     fa["equalized_odds_diff"])     + " EOD",
              f"{fa['equalized_odds_diff']:.4f}")
    c6.metric(badge("disparate_impact",        fa["disparate_impact"])        + " DI",
              f"{fa['disparate_impact']:.4f}")

def _color_delta(v):
    try:
        fv = float(v)
        if fv > 0.05:  return "color:red"
        if fv < -0.05: return "color:green"
    except Exception:
        pass
    return ""

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def check_backend_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.0)
        return r.status_code == 200, r.json()
    except Exception:
        return False, {}


# ─── sidebar: datasets & models ───────────────────────────────────────────────
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")

with st.sidebar:
    st.markdown("## ⚖️ FairGuard AI")
    st.caption("Responsible AI Assessment Workbench")

    api_online, health_data = check_backend_api()
    if api_online:
        st.success("🟢 FastAPI Backend Active")
    else:
        st.caption("⚪ Standalone Local Engine")

    st.markdown("---")
    st.subheader("📊 Evaluation Dataset")
    ds_options = {
        "adult": "UCI Adult Income (Primary)",
        "german": "German Credit (Secondary)",
        "synthetic_biased": "Synthetic — Imbalanced Bias",
        "synthetic_shifted": "Synthetic — Feature Drift",
        "synthetic_pii": "Synthetic — PII Injection",
    }
    sel_dataset = st.selectbox(
        "Select dataset to audit:",
        list(ds_options.keys()),
        format_func=lambda k: ds_options[k],
        key="sel_dataset"
    )

    st.markdown("---")
    st.subheader("📂 Pre-Built Input Models")
    input_files = (sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".joblib"))
                   if os.path.isdir(INPUT_DIR) else [])
    sel_input = st.selectbox("Select model to load:", ["— None —"] + input_files,
                             key="sel_input")

    st.markdown("---")
    st.subheader("📤 Upload Your Own Model")
    st.caption("`.joblib` / `.pkl` → `(model, preprocessor)` tuple")
    up_file = st.file_uploader("Choose file", type=["joblib", "pkl"], key="uploader")

    st.markdown("---")
    st.caption("**A** HistGradientBoosting (unmitigated)  \n"
               "**B** ExponentiatedGradient (DemographicParity)  \n"
               "**C** Your uploaded / selected model")


# ─── cached data & training ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading dataset & evaluating models…")
def load_and_train(dataset_name="adult"):
    X, y, sens = data_loader.load(dataset_name=dataset_name)
    Xtr, Xte, ytr, yte, str_, ste = train_test_split(
        X, y, sens, test_size=0.2, random_state=42, stratify=y)
    trained = models.train_all(Xtr, ytr, str_)
    res = {}
    for n, m in trained.items():
        p, pr = models.predict(m, Xte)
        res[n] = {"model": m, "preds": p, "proba": pr,
                  "metrics": models.metrics(yte, p, pr), "label": n}
    profile = data_loader.profile(X, y, sens)
    return Xtr, Xte, ytr, yte, str_, ste, res, profile


@st.cache_resource(show_spinner="Running drift & privacy…")
def run_drift_privacy(Xtr, Xte, ste, _model_A, ytr, yte):
    ref = Xtr.copy()
    cur = drift_engine.make_shifted(Xte.copy())
    dr  = drift_engine.drift(ref, cur)
    comb = pd.concat([Xte.reset_index(drop=True), ste.reset_index(drop=True)], axis=1)
    ka   = privacy_engine.k_anonymity(comb)
    mia  = privacy_engine.mia_proxy(_model_A, Xtr, ytr, Xte, yte)
    return dr, ka, mia


Xtr, Xte, ytr, yte, str_, ste, results, profile = load_and_train(dataset_name=sel_dataset)
ATTRS = list(ste.columns)


# ─── resolve Model C ─────────────────────────────────────────────────────────
model_c_tuple = None
model_c_name  = None

if up_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
            tmp.write(up_file.read())
            path = tmp.name
        obj = joblib.load(path)
        os.unlink(path)
        if isinstance(obj, tuple) and len(obj) == 2:
            model_c_tuple = obj
            model_c_name  = (up_file.name.replace(".joblib", "")
                                         .replace(".pkl", "")
                                         .replace("_", " ").title())
            st.sidebar.success(f"Uploaded: **{model_c_name}**")
        else:
            st.sidebar.error("File must be a (model, preprocessor) tuple.")
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")

elif sel_input != "— None —":
    try:
        obj = joblib.load(os.path.join(INPUT_DIR, sel_input))
        if isinstance(obj, tuple) and len(obj) == 2:
            model_c_tuple = obj
            model_c_name  = sel_input.replace(".joblib", "").replace("_", " ").title()
            st.sidebar.success(f"Loaded: **{model_c_name}**")
        else:
            st.sidebar.error("File must be a (model, preprocessor) tuple.")
    except Exception as e:
        st.sidebar.error(f"Load error: {e}")

# ─── inject Model C into results ─────────────────────────────────────────────
if model_c_tuple is not None:
    try:
        p_c, pr_c = models.predict(model_c_tuple, Xte)
        results["C"] = {
            "model":   model_c_tuple,
            "preds":   p_c,
            "proba":   pr_c,
            "metrics": models.metrics(yte, p_c, pr_c),
            "label":   model_c_name,
        }
    except Exception as e:
        st.sidebar.error(f"Model C evaluation failed: {e}")
        results.pop("C", None)
else:
    results.pop("C", None)

has_C = "C" in results
avail = list(results.keys())

def mdl_label(k):
    if k == "A":
        return "Model A — HistGradientBoosting"
    if k == "B":
        return "Model B — ExponentiatedGradient"
    lbl = results["C"]["label"] if "C" in results else "Unknown"
    return f"Model C — {lbl}"

def default_idx(keys):
    return keys.index("C") if has_C and "C" in keys else 0

# ─── page title & executive scorecard ───────────────────────────────────────
c_badge = f"  |  Model C: {results['C']['label']}" if has_C else ""
st.title(f"⚖️ FairGuard AI{c_badge}")

# Compute multi-pillar risk for active model
eval_model_key = "C" if has_C else "A"
eval_attr = ATTRS[0] if ATTRS else "sex"
fa_eval = fairness_engine.audit(yte, results[eval_model_key]["preds"], ste[eval_attr])
dr_eval, ka_eval, mia_eval = run_drift_privacy(Xtr, Xte, ste, results[eval_model_key]["model"], ytr, yte)
priv_eval = {
    "k_anonymity": ka_eval,
    "membership_inference": mia_eval,
    "privacy_risk_score": 0.7 if mia_eval.get("risk") == "High" else (0.35 if mia_eval.get("risk") == "Medium" else 0.1)
}
overall_risk = risk_service.calculate_overall_risk(
    fairness=fa_eval,
    privacy=priv_eval,
    drift=dr_eval,
    performance=results[eval_model_key]["metrics"],
    explainability={"proxy_warning_flag": False}
)
recommendations = risk_service.generate_recommendations(
    fairness=fa_eval,
    privacy=priv_eval,
    drift=dr_eval,
    performance=results[eval_model_key]["metrics"],
    explainability={"proxy_warning_flag": False}
)

# Executive Risk & Trustworthiness Banner
risk_lvl = overall_risk["overall_risk_level"]
risk_badge = {"Low": "🟢 Low Risk", "Medium": "🟡 Medium Risk", "High": "🔴 High Risk"}.get(risk_lvl, risk_lvl)
risk_color = {"Low": "#22c55e", "Medium": "#eab308", "High": "#ef4444"}.get(risk_lvl, "#3b82f6")

with st.container():
    st.markdown(
        f"""
        <div style="background: {risk_color}12; border-left: 5px solid {risk_color}; padding: 12px 18px; border-radius: 6px; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 17px; font-weight: 600; color: {risk_color};">
                    Trustworthiness Status: {risk_badge} (Score: {overall_risk['overall_risk_score']:.4f})
                </span>
                <span style="font-size: 13px; color: #6b7280;">
                    Model Under Audit: <strong>{mdl_label(eval_model_key)}</strong> | Dataset: <strong>{sel_dataset}</strong>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander(f"💡 Actionable Recommendations ({len(recommendations)} active items)", expanded=False):
        for rec in recommendations:
            sev_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(rec.get("severity"), "⚪")
            st.markdown(f"- **{sev_icon} [{rec.get('module')}]** {rec.get('text')}")

if has_C:
    st.info(f"Model C **({results['C']['label']})** is active — "
            f"its full assessment appears automatically in every tab below.")

tabs = st.tabs(["📊 Data Explorer", "📈 Model Comparison",
                "⚖️ Fairness Audit", "🔍 Explainability",
                "🌊 Drift & Privacy", "📋 Governance", "✅ Review"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.warning("Review measured model behaviour before making deployment decisions.")
    st.header("Dataset Profile — UCI Adult Income")
    cb  = profile["class_balance"]
    pos = cb.get(1, cb.get("1", cb.get(True, 0)))
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows",        f"{profile['n_rows']:,}")
    c2.metric("Features",    profile["n_features"])
    c3.metric("Income >50K", f"{pos:.1%}")
    st.bar_chart(pd.DataFrame({"Class": ["<=50K", ">50K"],
                               "Proportion": [1 - pos, pos]}).set_index("Class"))

    for attr in ATTRS:
        gc  = profile.get(f"group_counts_{attr}", {})
        gbr = profile.get(f"group_base_rates_{attr}", {})
        if not gc:
            continue
        st.subheader(f"Group Distribution — `{attr}`")
        gdf = pd.DataFrame({"Count": gc, "Positive Rate": gbr})
        st.dataframe(gdf.style.format({"Count": "{:,.0f}", "Positive Rate": "{:.2%}"}),
                     use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 3))
        gdf["Positive Rate"].sort_values().plot(kind="barh", ax=ax, color="steelblue")
        ax.axvline(pos, color="red", linestyle="--", label="Overall")
        ax.set_xlabel("Positive Rate")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.warning("Review measured model behaviour before making deployment decisions.")
    st.header("Model Comparison — Performance vs Fairness")
    if has_C:
        st.success(f"**{results['C']['label']}** included in comparison below.")

    rows = []
    for k in avail:
        m_res = results[k]
        fa    = fairness_engine.audit(yte, m_res["preds"], ste["sex"])
        rows.append({
            "Model":      mdl_label(k),
            "Accuracy":   m_res["metrics"]["accuracy"],
            "AUC":        m_res["metrics"]["auc"],
            "F1":         m_res["metrics"]["f1"],
            "DPD":        fa["demographic_parity_diff"],
            "EOD":        fa["equalized_odds_diff"],
            "DI":         fa["disparate_impact"],
            "ERD":        fa["error_rate_disparity"],
            "DPD Status": badge("demographic_parity_diff", fa["demographic_parity_diff"]),
            "EOD Status": badge("equalized_odds_diff",     fa["equalized_odds_diff"]),
            "DI Status":  badge("disparate_impact",        fa["disparate_impact"]),
        })
    df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        df.style.format({c: "{:.4f}" for c in ["Accuracy", "AUC", "F1", "DPD", "EOD", "DI", "ERD"]}),
        use_container_width=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    df[["Accuracy", "AUC", "F1"]].plot(kind="bar", ax=axes[0], ylim=(0, 1), colormap="Set2")
    axes[0].set_title("Performance")
    axes[0].tick_params(axis="x", rotation=15)

    df[["DPD", "EOD"]].plot(kind="bar", ax=axes[1], color=["tomato", "steelblue"])
    axes[1].set_title("Fairness Gap (lower = better)")
    axes[1].axhline(0.1, color="green",  linestyle="--", lw=1, label="Pass <=0.10")
    axes[1].axhline(0.2, color="orange", linestyle="--", lw=1, label="Warn <=0.20")
    axes[1].legend(fontsize=7)
    axes[1].tick_params(axis="x", rotation=15)

    df[["DI"]].plot(kind="bar", ax=axes[2], color="seagreen")
    axes[2].set_title("Disparate Impact (>=0.80 = Pass)")
    axes[2].axhline(0.8, color="green",  linestyle="--", lw=1)
    axes[2].axhline(0.7, color="orange", linestyle="--", lw=1)
    axes[2].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if has_C:
        st.markdown("---")
        st.subheader(f"Quick Assessment — {results['C']['label']}")
        fa_c = fairness_engine.audit(yte, results["C"]["preds"], ste["sex"])
        show_metrics_row(results["C"]["metrics"], fa_c)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FAIRNESS AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.warning("Review measured model behaviour before making deployment decisions.")
    st.header("Fairness Audit")
    if has_C:
        st.info(f"Model C **({results['C']['label']})** is selectable below. "
                f"A full cross-attribute comparison vs Model A is shown at the bottom.")

    col1, col2 = st.columns(2)
    fa_model = col1.selectbox("Model", avail, index=default_idx(avail),
                              format_func=mdl_label, key="fa_model")
    fa_attr  = col2.selectbox("Sensitive Attribute", ATTRS, key="fa_attr")

    m_res = results[fa_model]
    fa    = fairness_engine.audit(yte, m_res["preds"], ste[fa_attr])

    KEYS = [("Demographic Parity Diff", "demographic_parity_diff"),
            ("Equalized Odds Diff",     "equalized_odds_diff"),
            ("Disparate Impact",        "disparate_impact"),
            ("Error Rate Disparity",    "error_rate_disparity")]

    cols = st.columns(4)
    for (lbl_text, key), col in zip(KEYS, cols):
        val = fa[key]
        s   = fairness_engine.status(key, val)
        col.metric(f"{SC[s]} {lbl_text}", f"{val:.4f}")
        col.caption(f"Status: **{s}**")
    if fa.get("di_flag"):
        st.error("Disparate Impact < 0.80 — violates the Four-Fifths Rule!")

    st.subheader(f"Per-Group Metrics — `{fa_attr}`")
    st.caption("Group counts and small-group flags are shown beside the metrics.")
    st.dataframe(fa["by_group"].style.format("{:.4f}"), use_container_width=True)

    st.subheader("Confusion Matrices by Group")
    try:
        fig_cm = fairness_engine.confusion_by_group(yte, m_res["preds"], ste[fa_attr])
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    except Exception as e:
        st.warning(f"Confusion matrices: {e}")

    st.subheader("Intersectional — sex x race")
    try:
        inter = fairness_engine.intersectional(yte, m_res["preds"], ste[["sex", "race"]])
        st.dataframe(inter.style.format("{:.4f}"), use_container_width=True)
    except Exception as e:
        st.warning(f"Intersectional: {e}")

    if has_C:
        st.markdown("---")
        st.subheader(f"Model C Full Fairness Scorecard — {results['C']['label']}")
        cmp_rows = []
        for attr in ATTRS:
            fa_a = fairness_engine.audit(yte, results["A"]["preds"], ste[attr])
            fa_c = fairness_engine.audit(yte, results["C"]["preds"], ste[attr])
            for (mlbl, mkey) in KEYS:
                va = fa_a[mkey]; vc = fa_c[mkey]
                sa  = fairness_engine.status(mkey, va)
                sc2 = fairness_engine.status(mkey, vc)
                cmp_rows.append({
                    "Attribute":    attr,
                    "Metric":       mlbl,
                    "Model A":      va,
                    "Status A":     f"{SC[sa]} {sa}",
                    "Model C":      vc,
                    "Status C":     f"{SC[sc2]} {sc2}",
                    "Delta (C-A)":  round(vc - va, 4),
                })
        cmp_df = pd.DataFrame(cmp_rows)
        st.dataframe(
            cmp_df.style
                  .format({"Model A": "{:.4f}", "Model C": "{:.4f}", "Delta (C-A)": "{:.4f}"})
                  .map(_color_delta, subset=["Delta (C-A)"]),
            use_container_width=True)

        st.subheader(f"Model C — Per-Group (`{fa_attr}`)")
        fa_c2 = fairness_engine.audit(yte, results["C"]["preds"], ste[fa_attr])
        st.dataframe(fa_c2["by_group"].style.format("{:.4f}"), use_container_width=True)
        try:
            fig_cm_c = fairness_engine.confusion_by_group(
                yte, results["C"]["preds"], ste[fa_attr])
            st.pyplot(fig_cm_c)
            plt.close(fig_cm_c)
        except Exception as e:
            st.warning(f"Model C confusion: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.warning("Review measured model behaviour before making deployment decisions.")
    st.header("SHAP Explainability Studio")
    if has_C:
        st.info(f"Model C **({results['C']['label']})** is selectable. "
                f"SHAP plots for C appear at the bottom.")

    xai_key   = st.selectbox("Explain Model", avail, index=default_idx(avail),
                             format_func=mdl_label, key="xai_model")
    xai_tuple = results[xai_key]["model"]

    if xai_key == "B":
        st.warning("Model B: SHAP computed on its first base LogisticRegression predictor.")
        b_obj = xai_tuple[0]
        if hasattr(b_obj, "predictors_"):
            pl = list(b_obj.predictors_)
            if pl:
                xai_tuple = (pl[0], xai_tuple[1])

    xai_sample = Xte.head(SAMPLE_N).reset_index(drop=True)
    xai_sens   = ste["sex"].reset_index(drop=True).iloc[:SAMPLE_N]

    def render_shap(tup, label_key):
        safe_key = label_key.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

        try:
            with st.spinner("Computing beeswarm…"):
                fig = xai_engine.global_beeswarm(tup, xai_sample)
            st.subheader(f"Global Importance — Beeswarm ({label_key})")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Beeswarm failed: {e}")

        try:
            with st.spinner("Computing bar chart…"):
                fig = xai_engine.global_bar(tup, xai_sample)
            st.subheader(f"Global Importance — Bar ({label_key})")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Bar failed: {e}")

        idx = st.slider(f"Waterfall sample index",
                        0, SAMPLE_N - 1, 0, key=f"wf_{safe_key}")
        try:
            with st.spinner("Computing waterfall…"):
                fig = xai_engine.local_waterfall(tup, xai_sample, idx)
            st.subheader(f"Local Explanation — Waterfall ({label_key})")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Waterfall failed: {e}")

        try:
            with st.spinner("Computing group diff…"):
                gd = xai_engine.group_diff(tup, xai_sample, xai_sens)
            st.subheader(f"SHAP Difference by Sex ({label_key})")
            ranked = gd.drop(columns=["diff"], errors="ignore").mean(axis=1)
            ranked = ranked.sort_values(ascending=False).index.tolist()
            if xai_engine.proxy_warning(ranked):
                st.warning("Potential proxy feature detected among the top 10 model drivers; review its effect on model behaviour.")
            grad = [c for c in ["diff"] if c in gd.columns] or gd.columns.tolist()
            st.dataframe(
                gd.style.format("{:.4f}").background_gradient(subset=grad, cmap="RdYlGn"),
                use_container_width=True)
        except Exception as e:
            st.error(f"Group diff failed: {e}")

    render_shap(xai_tuple, mdl_label(xai_key))

    if has_C and xai_key != "C":
        st.markdown("---")
        st.subheader(f"SHAP — Model C: {results['C']['label']}")
        c_tuple = results["C"]["model"]
        m_obj   = c_tuple[0]
        info    = {"Type": type(m_obj).__name__}
        for attr in ["n_estimators", "max_depth", "C", "alpha", "kernel", "max_iter"]:
            if hasattr(m_obj, attr):
                info[attr] = str(getattr(m_obj, attr))
        st.json(info)
        render_shap(c_tuple, f"Model C {results['C']['label']}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DRIFT & PRIVACY
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.warning("Review measured model behaviour before making deployment decisions.")
    st.header("Data Drift & Privacy Risk")
    if has_C:
        st.info(f"MIA proxy for **{results['C']['label']}** shown automatically below.")

    dr, ka, mia_A = run_drift_privacy(Xtr, Xte, ste, results["A"]["model"], ytr, yte)

    st.subheader("Data Drift (Train vs Shifted Test)")
    d1, d2, d3 = st.columns(3)
    d1.metric("Drifted Features", f"{dr['share_drifted']:.1%}")
    d2.metric("Total Features",   len(dr["per_feature"]))
    d3.metric("Status", "Drift Detected!" if dr["share_drifted"] > 0.2 else "Low Drift")

    if dr["per_feature"]:
        feat_df = pd.DataFrame(dr["per_feature"]).T
        if "drifted" in feat_df.columns:
            feat_df["drifted"] = feat_df["drifted"].astype(str)
        st.dataframe(feat_df, use_container_width=True)

    drift_html = dr.get("html_path", "")
    if os.path.exists(drift_html):
        with open(drift_html, "r", encoding="utf-8") as f:
            drift_content = f.read()
        with st.expander("Full Evidently Drift Report", expanded=False):
            components.html(drift_content, height=700, scrolling=True)

    st.subheader("Privacy — Model A (Baseline)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Min k-Anonymity", ka["min_k"])
    p2.metric("Rows k < 5",      f"{ka['pct_rows_k_lt_5']}%")
    rc = {"Low": "Low", "Medium": "Medium", "High": "High"}.get(mia_A["risk"], "Unknown")
    p3.metric("MIA Risk",        rc)
    p4.metric("AUC Gap",         f"{mia_A['auc_gap']:.4f}")

    with st.expander("MIA Details — Model A"):
        mia_rows = {k: v for k, v in mia_A.items() if k != "risk"}
        st.dataframe(
            pd.DataFrame(list(mia_rows.items()), columns=["Metric", "Value"]),
            use_container_width=True)

    if ka["min_k"] == 1:
        st.error("k=1 — record is uniquely identifiable. Generalise quasi-identifiers.")
    elif ka["min_k"] < 5:
        st.warning(f"k={ka['min_k']} — some records re-identifiable.")
    else:
        st.success(f"k={ka['min_k']} — acceptable anonymity.")

    if has_C:
        st.markdown("---")
        st.subheader(f"Privacy — Model C: {results['C']['label']}")
        try:
            with st.spinner(f"Running MIA for {results['C']['label']}…"):
                mia_c = privacy_engine.mia_proxy(results["C"]["model"], Xtr, ytr, Xte, yte)
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Train AUC", mia_c["train_auc"])
            pc2.metric("Test AUC",  mia_c["test_auc"])
            rc2 = {"Low": "Low", "Medium": "Medium", "High": "High"}.get(mia_c["risk"], "Unknown")
            pc3.metric("MIA Risk",  rc2)
            pc4.metric("AUC Gap",   f"{mia_c['auc_gap']:.4f}")

            st.subheader("Model A vs Model C — Privacy Comparison")
            mia_a_clean = {k: v for k, v in mia_A.items() if k != "risk"}
            mia_c_clean = {k: v for k, v in mia_c.items() if k != "risk"}
            cmp = pd.DataFrame({
                "Model A": mia_a_clean,
                f"Model C ({results['C']['label']})": mia_c_clean,
            })
            st.dataframe(cmp.style.format("{:.4f}"), use_container_width=True)

            if mia_c["risk"] == "High":
                st.error("Model C has HIGH membership inference risk — overfitting suspected.")
            elif mia_c["risk"] == "Medium":
                st.warning("Model C has MEDIUM membership inference risk.")
            else:
                st.success("Model C membership inference risk is Low.")
        except Exception as e:
            st.error(f"Model C MIA failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — GOVERNANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.warning("Review measured model behaviour before making deployment decisions.")
    st.header("Governance & Model Card")
    if has_C:
        st.info(f"Full scorecard for **{results['C']['label']}** shown below, "
                f"including fairness across all attributes and MIA risk.")

    col_a, col_b = st.columns([1, 3])
    gen_btn = col_a.button("Generate Model Card", use_container_width=True)
    col_b.caption("Mitchell et al. compliant card — fairness, privacy & drift findings for Model A vs B.")

    if gen_btn:
        with st.spinner("Generating…"):
            dr2, ka2, mia2 = run_drift_privacy(Xtr, Xte, ste, results["A"]["model"], ytr, yte)
            fa_A = fairness_engine.audit(yte, results["A"]["preds"], ste["sex"])
            fa_B = fairness_engine.audit(yte, results["B"]["preds"], ste["sex"])
            try:
                model_card.generate(results["A"]["metrics"], results["B"]["metrics"],
                                    fa_A, fa_B, {"k_anon": ka2, "mia": mia2}, dr2)
                st.success("Model card generated!")
            except Exception as e:
                st.error(f"Error: {e}")

    mc_path = os.path.join("artifacts", "model_card.md")
    if os.path.exists(mc_path):
        with open(mc_path, "r", encoding="utf-8") as f:
            mc_txt = f.read()
        with st.expander("Preview Model Card", expanded=False):
            st.markdown(mc_txt)
        dl1, dl2 = st.columns(2)
        dl1.download_button("Download Markdown", mc_txt, "model_card.md",
                            "text/markdown", use_container_width=True)
        hp = os.path.join("artifacts", "model_card.html")
        if os.path.exists(hp):
            with open(hp, "r", encoding="utf-8") as f:
                html_txt = f.read()
            dl2.download_button("Download HTML", html_txt, "model_card.html",
                                "text/html", use_container_width=True)

    if has_C:
        st.markdown("---")
        st.subheader(f"Full Governance Scorecard — {results['C']['label']}")
        met_c = results["C"]["metrics"]

        st.markdown("#### Performance")
        pm1, pm2, pm3 = st.columns(3)
        pm1.metric("Accuracy", met_c["accuracy"])
        pm2.metric("AUC",      met_c["auc"])
        pm3.metric("F1",       met_c["f1"])

        st.markdown("#### Fairness vs Model A (by attribute)")
        FKEYS = [("DPD", "demographic_parity_diff"),
                 ("EOD", "equalized_odds_diff"),
                 ("DI",  "disparate_impact"),
                 ("ERD", "error_rate_disparity")]
        gov_rows = []
        for attr in ATTRS:
            fa_a = fairness_engine.audit(yte, results["A"]["preds"], ste[attr])
            fa_c = fairness_engine.audit(yte, results["C"]["preds"], ste[attr])
            for sh, fkey in FKEYS:
                va = fa_a[fkey]; vc = fa_c[fkey]
                sa  = fairness_engine.status(fkey, va)
                sc2 = fairness_engine.status(fkey, vc)
                gov_rows.append({
                    "Attr":     attr,
                    "Metric":   sh,
                    "Model A":  va,
                    "Status A": f"{SC[sa]} {sa}",
                    "Model C":  vc,
                    "Status C": f"{SC[sc2]} {sc2}",
                    "Delta C-A": round(vc - va, 4),
                })
        gov_df = pd.DataFrame(gov_rows)
        st.dataframe(
            gov_df.style
                  .format({"Model A": "{:.4f}", "Model C": "{:.4f}", "Delta C-A": "{:.4f}"})
                  .map(_color_delta, subset=["Delta C-A"]),
            use_container_width=True)

        st.markdown("#### Privacy (MIA Proxy)")
        try:
            with st.spinner("Running MIA for Model C…"):
                mia_c2 = privacy_engine.mia_proxy(results["C"]["model"], Xtr, ytr, Xte, yte)
            rc3 = {"Low": "Low", "Medium": "Medium", "High": "High"}.get(mia_c2["risk"], "Unknown")
            gm1, gm2, gm3 = st.columns(3)
            gm1.metric("MIA Risk",   rc3)
            gm2.metric("AUC Gap",    mia_c2["auc_gap"])
            gm3.metric("Conf. Gap",  mia_c2["confidence_gap"])
        except Exception as e:
            st.warning(f"MIA: {e}")

        st.markdown("#### Model Hyperparameters")
        m_obj = results["C"]["model"][0]
        info  = {"Type": type(m_obj).__name__}
        for a in ["n_estimators", "max_depth", "C", "alpha", "kernel", "max_iter", "learning_rate"]:
            if hasattr(m_obj, a):
                info[a] = str(getattr(m_obj, a))
        st.json(info)

    st.markdown("---")
    st.subheader("Deployment Sign-Off Checklist")
    items = [
        ("fair",   "Fairness audit reviewed by responsible AI lead"),
        ("thresh", "All metrics above threshold reviewed"),
        ("drift",  "Drift report reviewed — no significant shift"),
        ("priv",   "Privacy risk assessed — k-anonymity acceptable"),
        ("card",   "Model Card approved by governance team"),
        ("legal",  "Legal / regulatory review completed"),
        ("deploy", "Deployment approved"),
    ]
    checks = {k: st.checkbox(lbl, key=f"chk_{k}") for k, lbl in items}
    done   = sum(checks.values())
    st.progress(done / len(items), text=f"{done}/{len(items)} signed off")
    if done == len(items):
        st.success("All sign-offs complete — model cleared for deployment!")
    else:
        st.warning(f"{len(items) - done} item(s) pending.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — RESPONSIBLE AI REVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.warning("Review measured model behaviour before making deployment decisions.")
    st.header("Responsible AI Review")
    review_model = st.selectbox("Review model", avail, index=default_idx(avail),
                                format_func=mdl_label, key="review_model")
    review_fa = fairness_engine.audit(yte, results[review_model]["preds"], ste["sex"])
    reviewer = st.text_input("Reviewer name", key="reviewer_name")
    reason = st.text_area("Decision reason", key="decision_reason")
    gap = review_fa["equalized_odds_diff"]
    st.metric("Equalized odds gap", f"{gap:.4f}")
    approval_ok = governance.can_approve(gap, reviewer)
    if gap > 0.1:
        st.error("Approval is blocked because the equalized odds gap is above 0.1.")
    approve = st.button("Approve model", disabled=not approval_ok, key="approve_model")
    if approve:
        governance.save_decision(mdl_label(review_model), "Approved", reviewer, reason)
        if api_online:
            try:
                requests.post(
                    f"{API_URL}/assessments/default_assessment/review",
                    json={"reviewer": reviewer, "decision": "Approved", "comment": reason},
                    timeout=2.0
                )
            except Exception:
                pass
        st.success("Approval decision saved to relational database and audit logs.")

    st.caption(f"Current decision status: {governance.latest_status()}")

    # Display audit trail of reviews
    past_reviews = crud.get_reviews("default_assessment")
    if past_reviews:
        st.subheader("Auditor Sign-Off History")
        rev_df = pd.DataFrame(past_reviews)[["reviewer", "decision", "comment", "reviewed_at"]]
        st.dataframe(rev_df, use_container_width=True)

