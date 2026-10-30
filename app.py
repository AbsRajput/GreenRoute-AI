
from pathlib import Path
import html

import textwrap
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.model_profiler import benchmark_models
from src.custom_dataset_lab import evaluate_uploaded_dataset


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="GreenRoute-AI | Intelligence Console",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
(ROOT / "results").mkdir(exist_ok=True)


# ==========================================================
# 2. PREMIUM DARK UI
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #050b12;
    --panel: #0b1722;
    --mint: #62f5bd;
    --cyan: #38dff5;
    --violet: #a6a0ff;
    --muted: #91aab5;
    --line: rgba(107,237,220,.18);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
      radial-gradient(ellipse at 10% 0%, rgba(18,110,102,.22), transparent 33%),
      radial-gradient(ellipse at 92% 8%, rgba(30,102,165,.20), transparent 32%),
      linear-gradient(145deg,#050b12,#08131c 55%,#070c17);
    color: #effcfb;
}

[data-testid="stHeader"] {
    background: rgba(5,11,18,.9);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a1821,#071018);
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] * {
    color: #effcfb;
}

.block-container {
    max-width: 1550px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

h1,h2,h3,h4 {
    color: #effcfb !important;
    letter-spacing: -.04em;
}

p,li,label {
    color: #bfd1d8;
}

/* TOP BAR */

.topline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0 20px;
    color: #a7c0c9;
    font-size: .74rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: .09em;
}

.brandmark {
    color: var(--mint);
    font-weight: 800;
    font-size: 1rem;
}

.status-pill {
    display: inline-block;
    padding: 7px 11px;
    border-radius: 99px;
    border: 1px solid rgba(98,245,189,.27);
    background: rgba(98,245,189,.07);
    color: var(--mint);
    font-size: .67rem;
}

/* HERO */

.hero {
    position: relative;
    overflow: hidden;
    padding: 40px;
    margin-bottom: 26px;
    border-radius: 23px;
    border: 1px solid rgba(92,245,219,.36);
    background:
      radial-gradient(ellipse at 90% 20%,rgba(42,219,221,.18),transparent 32%),
      radial-gradient(ellipse at 65% 100%,rgba(64,109,229,.15),transparent 40%),
      linear-gradient(120deg,#0c242d,#07121e 65%,#091623);
    box-shadow:
      0 0 0 1px rgba(74,216,212,.04),
      0 18px 60px rgba(0,0,0,.30),
      inset 0 1px rgba(255,255,255,.04);
}

.hero:before {
    content: "";
    position: absolute;
    right: -85px;
    top: -160px;
    width: 390px;
    height: 390px;
    border: 1px solid rgba(56,223,245,.23);
    border-radius: 50%;
    box-shadow:
      0 0 0 28px rgba(56,223,245,.025),
      0 0 0 66px rgba(56,223,245,.02),
      0 0 0 108px rgba(56,223,245,.012);
    pointer-events: none;
}

.eyebrow {
    color: var(--mint);
    font-family: 'DM Mono', monospace;
    font-size: .69rem;
    letter-spacing: .16em;
    margin-bottom: 17px;
}

.hero-title {
    font-size: clamp(2.2rem,4.5vw,4.1rem);
    line-height: 1.04;
    font-weight: 800;
    letter-spacing: -.075em;
    margin-bottom: 16px;
    color: #f5ffff;
    position: relative;
    z-index: 1;
}

.gradient {
    background: linear-gradient(90deg,#62f5bd,#38dff5 56%,#a6a0ff);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-copy {
    color: #b0c9d2;
    font-size: .95rem;
    line-height: 1.8;
    max-width: 740px;
    margin-bottom: 22px;
    position: relative;
    z-index: 1;
}

.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    position: relative;
    z-index: 1;
}

.tag {
    padding: 8px 11px;
    border-radius: 8px;
    border: 1px solid rgba(123,235,227,.18);
    background: rgba(5,18,28,.62);
    color: #d4f8f1;
    font: .64rem 'DM Mono',monospace;
}

/* SECTION HEADINGS */

.section-title {
    font-size: 1.48rem;
    font-weight: 800;
    letter-spacing: -.05em;
    margin: 10px 0 3px;
}

.section-sub {
    color: var(--muted);
    font-size: .86rem;
    line-height: 1.7;
    margin-bottom: 18px;
}

/* KPI CARDS */

.metric-card {
    position: relative;
    overflow: hidden;
    min-height: 126px;
    padding: 19px;
    border-radius: 15px;
    border: 1px solid rgba(117,225,225,.19);
    background: linear-gradient(145deg,rgba(17,43,54,.94),rgba(9,24,35,.96));
    box-shadow: 0 12px 35px rgba(0,0,0,.16);
    transition: .2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(98,245,189,.5);
}

.metric-card:after {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg,var(--mint),var(--cyan),transparent);
}

.metric-label {
    color: #9bb8c2;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .08em;
    margin-bottom: 14px;
}

.metric-value {
    color: #f3fffd;
    font-size: 1.78rem;
    font-weight: 800;
    letter-spacing: -.06em;
    line-height: 1.1;
    overflow-wrap: anywhere;
}

.metric-foot {
    color: #7899a5;
    font-size: .69rem;
    margin-top: 10px;
}

/* GLASS PANELS */

.glass {
    padding: 20px;
    border-radius: 16px;
    border: 1px solid var(--line);
    background: linear-gradient(145deg,rgba(13,34,45,.87),rgba(8,21,31,.90));
    margin-bottom: 15px;
}

.mini-label {
    color: #84a6b2;
    font: .65rem 'DM Mono',monospace;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.note {
    padding: 14px 16px;
    border-left: 3px solid var(--mint);
    border-radius: 0 10px 10px 0;
    background: rgba(98,245,189,.055);
    color: #c2d8de;
    font-size: .84rem;
    line-height: 1.7;
    margin: 12px 0;
}

/* STREAMLIT CONTROLS */

.stButton button,
.stDownloadButton button {
    border: 1px solid rgba(98,245,189,.28);
    border-radius: 10px;
    background: linear-gradient(100deg,#62f5bd,#38dff5);
    color: #04131b;
    font-weight: 800;
}

.stButton button:hover,
.stDownloadButton button:hover {
    color: #04131b;
    filter: brightness(1.08);
    border-color: #9bfff0;
}

div[data-testid="stTabs"] button {
    color: #91adb8;
    padding: 10px 13px;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f2fffd;
    background: rgba(56,223,245,.08);
}

div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: #62f5bd;
}

[data-testid="stFileUploader"] {
    border: 1px dashed rgba(98,245,189,.35);
    border-radius: 13px;
    background: rgba(10,28,37,.45);
    padding: 10px;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #0e202b;
    border-color: rgba(117,225,225,.20);
    border-radius: 9px;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: rgba(117,225,225,.13);
}

#MainMenu,footer {
    visibility: hidden;
}

@media(max-width:768px) {
    .hero { padding: 25px 20px; }
    .hero-title { font-size: 2.5rem; }
    .metric-value { font-size: 1.45rem; }
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================

def section_heading(title, subtitle=""):
    st.markdown(
        f'<div class="section-title">{html.escape(title)}</div>'
        f'<div class="section-sub">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True
    )


def metric_card(label, value, foot=""):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{html.escape(str(label))}</div>'
        f'<div class="metric-value">{html.escape(str(value))}</div>'
        f'<div class="metric-foot">{html.escape(str(foot))}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def find_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def make_chart(df, x_col, y_col, title, y_label):
    if not x_col or not y_col:
        st.info(f"{title}: metric not available.")
        return

    plot_df = df[[x_col, y_col]].copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[y_col])

    if plot_df.empty:
        st.info(f"No numeric data available for {title}.")
        return

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#0b1722")
    ax.set_facecolor("#0b1722")

    bars = ax.bar(
        plot_df[x_col].astype(str),
        plot_df[y_col],
        edgecolor="#62f5bd",
        linewidth=.7
    )

    ax.set_title(title, color="#effcfb", pad=13, fontsize=12)
    ax.set_ylabel(y_label, color="#91aab5")
    ax.tick_params(axis="x", colors="#bfd1d8", rotation=12)
    ax.tick_params(axis="y", colors="#bfd1d8")
    ax.grid(axis="y", alpha=.13)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#243c49")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3g}",
            (bar.get_x() + bar.get_width()/2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color="#eafffa",
            fontsize=8
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def normalise_benchmark(df):
    df = df.copy()

    aliases = {
        "model": ["model", "model_name", "Model"],
        "accuracy": ["accuracy", "Accuracy"],
        "f1_score": ["f1_score", "weighted_f1", "f1"],
        "train_time_seconds": [
            "train_time_seconds", "training_time_seconds", "train_time"
        ],
        "prediction_latency_ms_per_sample": [
            "prediction_latency_ms_per_sample",
            "latency_ms_per_sample",
            "prediction_latency_ms",
            "latency_ms"
        ],
        "train_emissions_kg_co2eq": [
            "train_emissions_kg_co2eq",
            "training_emissions_kg_co2eq",
            "train_emissions"
        ],
        "prediction_emissions_kg_co2eq": [
            "prediction_emissions_kg_co2eq",
            "inference_emissions_kg_co2eq",
            "prediction_emissions"
        ],
    }

    for standard, options in aliases.items():
        found = find_col(df, options)
        if found and standard not in df.columns:
            df[standard] = df[found]

    return df


def route_candidates(df, min_accuracy, max_latency, max_emissions=None):
    model_col = find_col(df, ["model", "model_name", "Model"])
    acc_col = find_col(df, ["accuracy", "Accuracy"])
    latency_col = find_col(df, [
        "prediction_latency_ms_per_sample",
        "latency_ms_per_sample",
        "prediction_latency_ms",
        "latency_ms"
    ])

    emissions_col = find_col(df, [
        "prediction_emissions_kg_co2eq",
        "inference_emissions_kg_co2eq",
        "train_emissions_kg_co2eq",
        "training_emissions_kg_co2eq",
        "emissions_kg_co2eq"
    ])

    if not model_col or not acc_col or not latency_col:
        return None, pd.DataFrame(), "Benchmark is missing required columns."

    work = df.copy()
    work[acc_col] = pd.to_numeric(work[acc_col], errors="coerce")
    work[latency_col] = pd.to_numeric(work[latency_col], errors="coerce")
    work = work.dropna(subset=[acc_col, latency_col])

    eligible = work[
        (work[acc_col] >= min_accuracy) &
        (work[latency_col] <= max_latency)
    ].copy()

    if max_emissions is not None:
        if not emissions_col:
            return None, eligible, "No emissions metric is available."

        eligible[emissions_col] = pd.to_numeric(
            eligible[emissions_col], errors="coerce"
        )
        eligible = eligible.dropna(subset=[emissions_col])
        eligible = eligible[eligible[emissions_col] <= max_emissions]

    if eligible.empty:
        return None, eligible, "No model meets all active constraints."

    eligible = eligible.sort_values(
        [latency_col, acc_col],
        ascending=[True, False]
    )

    return (
        eligible.iloc[0],
        eligible,
        "Fastest model among candidates meeting the active constraints."
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_benchmark(dataset_name):
    result = benchmark_models(dataset_name=dataset_name)

    if not isinstance(result, pd.DataFrame):
        result = pd.DataFrame(result)

    return normalise_benchmark(result)


# ==========================================================
# 4. SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        '<div style="padding:8px 0 18px">'
        '<div style="font-size:1.8rem">🌿</div>'
        '<div style="font-size:1.12rem;font-weight:800;color:#effcfb">'
        'GREENROUTE<span style="color:#62f5bd">-AI</span></div>'
        '<div class="mini-label" style="margin-top:6px">'
        'INTELLIGENCE CONSOLE</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("#### EXPERIMENT CONTROL")

    dataset_choice = st.selectbox(
        "Built-in dataset",
        ["breast_cancer", "wine"]
    )

    min_accuracy = st.slider(
        "Minimum accuracy",
        0.50, 1.00, 0.90, 0.01
    )

    max_latency = st.number_input(
        "Maximum latency (ms/sample)",
        min_value=0.001,
        max_value=10000.0,
        value=1.0,
        step=0.1,
        format="%.3f"
    )

    carbon_enabled = st.checkbox(
        "Enable carbon-emissions constraint",
        value=False
    )

    max_emissions = None

    if carbon_enabled:
        max_emissions = st.number_input(
            "Maximum CO₂e (kg)",
            min_value=0.0,
            value=0.00001,
            step=0.000001,
            format="%.8f"
        )

    st.markdown("---")
    st.markdown(
        '<div class="mini-label">SYSTEM STATUS</div>'
        '<div style="margin-top:9px">'
        '<span class="status-pill">● LOCAL RESEARCH MODE</span></div>',
        unsafe_allow_html=True
    )

    st.caption("Local benchmarking • Adaptive routing • CSV export")


# ==========================================================
# 5. TOP BAR + HERO
# ==========================================================

st.html("""
<div class="topline">
    <div>
        <span class="brandmark">◈ GREENROUTE-AI</span>
        &nbsp; / &nbsp; RESEARCH CONTROL CENTER
    </div>
    <div class="status-pill">● SYSTEM READY</div>
</div>
<div class="hero">
    <div class="eyebrow">
        SUSTAINABLE AI / MODEL INTELLIGENCE PLATFORM
    </div>
    <div class="hero-title">
        Make every inference<br>
        <span class="gradient">count for more.</span>
    </div>
    <div class="hero-copy">
        A carbon-aware model evaluation and adaptive routing workspace.
        Benchmark candidate models, explore performance trade-offs, and
        inspect routing decisions through a clean experimental interface.
    </div>
    <div class="hero-tags">
        <span class="tag">↗ MODEL BENCHMARKING</span>
        <span class="tag">◉ CUSTOM DATASETS</span>
        <span class="tag">⌁ LATENCY ANALYTICS</span>
        <span class="tag">✳ CARBON-AWARE ROUTING</span>
    </div>
</div>
""")

# ==========================================================
# 6. LOAD BUILT-IN BENCHMARK
# ==========================================================

with st.spinner("Preparing model benchmark..."):
    try:
        benchmark_df = load_benchmark(dataset_choice)
        benchmark_error = None
    except Exception as exc:
        benchmark_df = pd.DataFrame()
        benchmark_error = str(exc)


# ==========================================================
# 7. NAVIGATION TABS
# ==========================================================

overview, routing, custom, analytics, methodology = st.tabs([
    "◈ Overview",
    "⌁ Routing Lab",
    "＋ Custom Dataset Lab",
    "▥ Analytics",
    "◎ Methodology"
])


# ==========================================================
# 8. OVERVIEW
# ==========================================================

with overview:

    section_heading(
        "Dashboard overview",
        "Live view of the selected benchmark and its key performance metrics."
    )

    if benchmark_error:
        st.error("The benchmark could not be loaded.")
        st.code(benchmark_error)

    elif benchmark_df.empty:
        st.warning("The benchmark returned no rows.")

    else:
        model_col = find_col(
            benchmark_df, ["model", "model_name", "Model"]
        )
        acc_col = find_col(
            benchmark_df, ["accuracy", "Accuracy"]
        )
        lat_col = find_col(
            benchmark_df,
            ["prediction_latency_ms_per_sample",
             "latency_ms_per_sample",
             "prediction_latency_ms",
             "latency_ms"]
        )
        f1_col = find_col(
            benchmark_df, ["f1_score", "weighted_f1", "f1"]
        )
        train_col = find_col(
            benchmark_df,
            ["train_time_seconds", "training_time_seconds", "train_time"]
        )
        emissions_col = find_col(
            benchmark_df,
            ["prediction_emissions_kg_co2eq",
             "inference_emissions_kg_co2eq",
             "train_emissions_kg_co2eq",
             "training_emissions_kg_co2eq",
             "emissions_kg_co2eq"]
        )

        best_acc = (
            pd.to_numeric(benchmark_df[acc_col], errors="coerce").max()
            if acc_col else None
        )
        best_lat = (
            pd.to_numeric(benchmark_df[lat_col], errors="coerce").min()
            if lat_col else None
        )
        best_f1 = (
            pd.to_numeric(benchmark_df[f1_col], errors="coerce").max()
            if f1_col else None
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "CANDIDATE MODELS",
                str(benchmark_df[model_col].nunique() if model_col else len(benchmark_df)),
                "Compared in this benchmark"
            )

        with c2:
            metric_card(
                "BEST TEST ACCURACY",
                f"{best_acc:.1%}" if best_acc is not None and pd.notna(best_acc) else "N/A",
                "Higher is better"
            )

        with c3:
            metric_card(
                "LOWEST PREDICTION LATENCY",
                f"{best_lat:.3f} ms" if best_lat is not None and pd.notna(best_lat) else "N/A",
                "Per sample where available"
            )

        with c4:
            metric_card(
                "BEST WEIGHTED F1",
                f"{best_f1:.3f}" if best_f1 is not None and pd.notna(best_f1) else "N/A",
                "Classification metric"
            )

        st.markdown("")

        left, right = st.columns([1.2, 1], gap="large")

        with left:

            st.markdown(
                f'<div class="glass">'
                f'<div class="mini-label">ACTIVE EXPERIMENT</div>'
                f'<h3 style="margin:8px 0">{html.escape(dataset_choice.replace("_", " ").title())}</h3>'
                f'<p style="color:#91aab5">Candidate models are compared on this '
                f'built-in dataset. Change the sidebar settings to adjust routing.</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            make_chart(
                benchmark_df,
                model_col,
                acc_col,
                "Model accuracy comparison",
                "Accuracy"
            )

            st.download_button(
                "↓ Export benchmark CSV",
                benchmark_df.to_csv(index=False).encode("utf-8"),
                file_name=f"greenroute_{dataset_choice}_benchmark.csv",
                mime="text/csv",
                key="overview_export"
            )

        with right:

            st.markdown(
                '<div class="glass">'
                '<div class="mini-label">BENCHMARK SNAPSHOT</div>'
                '<h3 style="margin:8px 0">Model leaderboard</h3></div>',
                unsafe_allow_html=True
            )

            visible_cols = [
                c for c in [
                    model_col, acc_col, f1_col, train_col, lat_col
                ] if c
            ]

            st.dataframe(
                benchmark_df[visible_cols] if visible_cols else benchmark_df,
                use_container_width=True,
                hide_index=True
            )

            if emissions_col:
                best_emissions = pd.to_numeric(
                    benchmark_df[emissions_col], errors="coerce"
                ).min()

                metric_card(
                    "LOWEST REPORTED CO₂e",
                    f"{best_emissions:.2e} kg" if pd.notna(best_emissions) else "N/A",
                    f"Source: {emissions_col}"
                )
            else:
                st.markdown(
                    '<div class="note">Carbon estimate is not available in '
                    'this benchmark. No value is inferred from latency.</div>',
                    unsafe_allow_html=True
                )

    st.markdown(
        '<div class="note"><b>Research note:</b> results depend on the dataset, '
        'split, hardware load, and measurement procedure. Estimated CO₂e is '
        'not a direct electricity-meter reading.</div>',
        unsafe_allow_html=True
    )


# ==========================================================
# 9. ADAPTIVE ROUTING LAB
# ==========================================================

with routing:

    section_heading(
        "Adaptive routing lab",
        "Tune the policy and inspect which model satisfies the active constraints."
    )

    if benchmark_error or benchmark_df.empty:
        st.warning("A successful benchmark is required before routing.")

    else:
        selected, eligible, message = route_candidates(
            benchmark_df,
            min_accuracy,
            max_latency,
            max_emissions if carbon_enabled else None
        )

        left, right = st.columns([.9, 1.1], gap="large")

        with left:

            st.markdown(
                '<div class="glass"><div class="mini-label">POLICY CONFIGURATION</div>'
                '<h3 style="margin:8px 0">Active constraints</h3></div>',
                unsafe_allow_html=True
            )

            metric_card(
                "MINIMUM ACCURACY",
                f"{min_accuracy:.0%}",
                "Candidates below this are excluded"
            )

            metric_card(
                "MAXIMUM LATENCY",
                f"{max_latency:.3f} ms",
                "Candidates above this are excluded"
            )

            metric_card(
                "CARBON FILTER",
                "ON" if carbon_enabled else "OFF",
                "Requires compatible emissions data"
            )

        with right:

            st.markdown(
                '<div class="glass"><div class="mini-label">ROUTER OUTPUT</div>'
                '<h3 style="margin:8px 0">Decision result</h3></div>',
                unsafe_allow_html=True
            )

            if selected is None:
                st.warning(message)

            else:
                model_col = find_col(
                    benchmark_df, ["model", "model_name", "Model"]
                )
                acc_col = find_col(
                    benchmark_df, ["accuracy", "Accuracy"]
                )
                lat_col = find_col(
                    benchmark_df,
                    ["prediction_latency_ms_per_sample",
                     "latency_ms_per_sample",
                     "prediction_latency_ms",
                     "latency_ms"]
                )

                st.markdown(
                    f'<div class="hero" style="padding:25px;margin-bottom:12px">'
                    f'<div class="eyebrow">SELECTED CANDIDATE</div>'
                    f'<div style="font-size:2rem;font-weight:800;color:#62f5bd">'
                    f'{html.escape(str(selected[model_col]))}</div>'
                    f'<div style="display:flex;gap:32px;flex-wrap:wrap;margin-top:20px">'
                    f'<div><div class="mini-label">ACCURACY</div>'
                    f'<div style="font-size:1.4rem;font-weight:800">{float(selected[acc_col]):.2%}</div></div>'
                    f'<div><div class="mini-label">LATENCY</div>'
                    f'<div style="font-size:1.4rem;font-weight:800">{float(selected[lat_col]):.3f} ms</div></div>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

                st.success(message)

            st.caption(f"Eligible candidates: {len(eligible)}")

            if not eligible.empty:
                st.dataframe(
                    eligible,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No eligible candidates under current thresholds.")

        st.markdown("#### How the routing decision works")

        st.markdown(
            "1. Exclude models below the accuracy threshold.\n"
            "2. Exclude models above the latency threshold.\n"
            "3. If carbon filtering is enabled, require a compatible emissions metric.\n"
            "4. Select the lowest-latency candidate from the remaining pool."
        )

        st.caption(
            "This threshold-based policy does not guarantee globally minimum carbon emissions."
        )


# ==========================================================
# 10. CUSTOM DATASET LAB
# ==========================================================

with custom:

    section_heading(
        "Custom dataset lab",
        "Upload a labelled CSV and benchmark supported classifiers."
    )

    st.markdown(
        '<div class="glass">'
        '<div class="mini-label">YOUR DATA / YOUR EXPERIMENT</div>'
        '<h3 style="margin:8px 0">Bring your own dataset</h3>'
        '<p style="color:#91aab5">Current evaluator supports numeric feature '
        'columns and classification targets. Select the column containing '
        'the label you want to predict.</p></div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"],
        key="dataset_upload"
    )

    if uploaded_file is not None:

        try:
            uploaded_df = pd.read_csv(uploaded_file)

            if uploaded_df.empty:
                st.error("This CSV has no data rows.")

            else:
                c1, c2, c3 = st.columns(3)

                with c1:
                    metric_card("SAMPLES", f"{len(uploaded_df):,}", "CSV rows")

                with c2:
                    metric_card("COLUMNS", str(uploaded_df.shape[1]), "Features + target")

                with c3:
                    metric_card(
                        "MISSING CELLS",
                        f"{int(uploaded_df.isna().sum().sum()):,}",
                        "Before evaluation"
                    )

                st.markdown("#### Dataset preview")

                st.dataframe(
                    uploaded_df.head(10),
                    use_container_width=True,
                    hide_index=True
                )

                target_column = st.selectbox(
                    "Target / label column",
                    list(uploaded_df.columns),
                    index=len(uploaded_df.columns) - 1,
                    key="dataset_target"
                )

                test_fraction = st.slider(
                    "Test-set proportion",
                    .10, .40, .20, .05,
                    key="dataset_test_size"
                )

                st.caption(
                    f"Train: {1-test_fraction:.0%} · "
                    f"Test: {test_fraction:.0%} · Random state: 42"
                )

                if st.button(
                    "▶ Run custom evaluation",
                    type="primary",
                    key="run_dataset_eval"
                ):

                    try:
                        with st.spinner("Training candidate models..."):

                            result_df = evaluate_uploaded_dataset(
                                dataframe=uploaded_df,
                                target_column=target_column,
                                test_size=test_fraction,
                                random_state=42
                            )

                        st.session_state["gr_custom_result"] = result_df
                        st.session_state["gr_custom_filename"] = uploaded_file.name

                        st.success("Evaluation completed.")

                    except Exception as exc:
                        st.error("Evaluation failed.")
                        st.code(str(exc))

                custom_result = st.session_state.get("gr_custom_result")

                if (
                    custom_result is not None
                    and st.session_state.get("gr_custom_filename") == uploaded_file.name
                ):

                    st.markdown("---")
                    st.markdown("#### Model results")

                    best_accuracy = pd.to_numeric(
                        custom_result["accuracy"], errors="coerce"
                    ).max()

                    best_f1 = pd.to_numeric(
                        custom_result["f1_score"], errors="coerce"
                    ).max()

                    lowest_latency = pd.to_numeric(
                        custom_result["prediction_latency_ms_per_sample"],
                        errors="coerce"
                    ).min()

                    a1, a2, a3 = st.columns(3)

                    with a1:
                        metric_card(
                            "BEST ACCURACY",
                            f"{best_accuracy:.2%}" if pd.notna(best_accuracy) else "N/A",
                            "Across evaluated models"
                        )

                    with a2:
                        metric_card(
                            "BEST WEIGHTED F1",
                            f"{best_f1:.4f}" if pd.notna(best_f1) else "N/A",
                            "Across evaluated models"
                        )

                    with a3:
                        metric_card(
                            "LOWEST LATENCY",
                            f"{lowest_latency:.3f} ms" if pd.notna(lowest_latency) else "N/A",
                            "Per sample"
                        )

                    st.dataframe(
                        custom_result,
                        use_container_width=True,
                        hide_index=True
                    )

                    ca, cb = st.columns(2, gap="large")

                    with ca:
                        make_chart(
                            custom_result,
                            "model",
                            "accuracy",
                            "Accuracy by model",
                            "Accuracy"
                        )

                    with cb:
                        make_chart(
                            custom_result,
                            "model",
                            "prediction_latency_ms_per_sample",
                            "Prediction latency",
                            "ms/sample"
                        )

                    st.download_button(
                        "↓ Download custom results",
                        custom_result.to_csv(index=False).encode("utf-8"),
                        file_name="greenroute_custom_results.csv",
                        mime="text/csv",
                        key="custom_export"
                    )

                    st.markdown(
                        '<div class="note"><b>Measurement note:</b> custom evaluation '
                        'measures predictive metrics, training duration, and latency. '
                        'It does not measure carbon emissions for this upload.</div>',
                        unsafe_allow_html=True
                    )

        except Exception as exc:
            st.error("Could not read the uploaded CSV.")
            st.code(str(exc))

    else:

        st.markdown(
            '<div class="glass" style="text-align:center;padding:38px 20px">'
            '<div style="font-size:2.3rem">↥</div>'
            '<h3>Upload your CSV to begin</h3>'
            '<p style="color:#91aab5">Dataset preview, evaluation results, '
            'and charts will appear here.</p></div>',
            unsafe_allow_html=True
        )


# ==========================================================
# 11. MODEL ANALYTICS
# ==========================================================

with analytics:

    section_heading(
        "Performance analytics",
        "Visual comparison of the metrics available in the selected benchmark."
    )

    if benchmark_error or benchmark_df.empty:
        st.warning("Benchmark data is not available.")

    else:

        model_col = find_col(
            benchmark_df, ["model", "model_name", "Model"]
        )
        acc_col = find_col(
            benchmark_df, ["accuracy", "Accuracy"]
        )
        f1_col = find_col(
            benchmark_df, ["f1_score", "weighted_f1", "f1"]
        )
        lat_col = find_col(
            benchmark_df,
            ["prediction_latency_ms_per_sample",
             "latency_ms_per_sample",
             "prediction_latency_ms",
             "latency_ms"]
        )
        train_col = find_col(
            benchmark_df,
            ["train_time_seconds", "training_time_seconds", "train_time"]
        )
        emissions_col = find_col(
            benchmark_df,
            ["prediction_emissions_kg_co2eq",
             "inference_emissions_kg_co2eq",
             "train_emissions_kg_co2eq",
             "training_emissions_kg_co2eq",
             "emissions_kg_co2eq"]
        )

        c1, c2 = st.columns(2, gap="large")

        with c1:
            make_chart(
                benchmark_df, model_col, acc_col,
                "Test accuracy", "Accuracy"
            )

        with c2:
            make_chart(
                benchmark_df, model_col, lat_col,
                "Prediction latency", "ms/sample"
            )

        c3, c4 = st.columns(2, gap="large")

        with c3:
            make_chart(
                benchmark_df, model_col, train_col,
                "Training time", "Seconds"
            )

        with c4:
            make_chart(
                benchmark_df, model_col, f1_col,
                "Weighted F1 score", "F1"
            )

        st.markdown("#### Carbon / emissions")

        if emissions_col:

            make_chart(
                benchmark_df,
                model_col,
                emissions_col,
                "Reported CO₂e values",
                "kg CO₂e"
            )

            st.caption(
                f"Displayed column: {emissions_col}. "
                "Confirm whether it represents training or prediction."
            )

        else:
            st.info(
                "No compatible emissions column exists. "
                "Carbon is not inferred from latency."
            )

        st.markdown("#### Full benchmark table")

        st.dataframe(
            benchmark_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "↓ Export analytics data",
            benchmark_df.to_csv(index=False).encode("utf-8"),
            file_name=f"greenroute_{dataset_choice}_analytics.csv",
            mime="text/csv",
            key="analytics_export"
        )


# ==========================================================
# 12. METHODOLOGY
# ==========================================================

with methodology:

    section_heading(
        "Methodology & research notes",
        "Transparent benchmarking, constraint filtering, and careful interpretation."
    )

    st.markdown("""
    <div class="glass">
      <div class="mini-label">01 / DATA PIPELINE</div>
      <h3>Load and validate</h3>
      <p>Use a built-in scikit-learn dataset or labelled CSV.
      The custom evaluator uses a stratified hold-out split.</p>
    </div>

    <div class="glass">
      <div class="mini-label">02 / BENCHMARKING</div>
      <h3>Compare candidate models</h3>
      <p>Report accuracy, weighted F1, training duration,
      and prediction latency where available.</p>
    </div>

    <div class="glass">
      <div class="mini-label">03 / ADAPTIVE ROUTING</div>
      <h3>Filter, then select</h3>
      <p>Exclude candidates that fail configured thresholds,
      then select the lowest-latency candidate among those remaining.</p>
    </div>

    <div class="glass">
      <div class="mini-label">04 / CARBON ACCOUNTING</div>
      <h3>Keep estimates transparent</h3>
      <p>Emissions depend on instrumentation and the measured task.
      Training and inference should be reported separately;
      latency alone is not a carbon measurement.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Current limitations")

    limitations = [
        "The available model pool is defined by src/model_profiler.py.",
        "Routing thresholds are manually configured; the policy is not learned automatically.",
        "The lowest-latency eligible model is not necessarily the lowest-carbon model.",
        "Latency depends on hardware, system load, warm-up, and batch size.",
        "Custom CSV evaluation supports numeric features and classification labels only.",
        "A single train/test split does not establish statistical significance.",
        "Carbon savings should only be claimed after comparable baseline experiments."
    ]

    for item in limitations:
        st.markdown(f"- {item}")


# ==========================================================
# 13. FOOTER
# ==========================================================

st.markdown("---")

footer_left, footer_right = st.columns([2, 1])

with footer_left:

    st.markdown(
        '<div style="font-weight:800;color:#e9fffa">'
        'GREENROUTE<span style="color:#62f5bd">-AI</span></div>'
        '<div style="font-size:.73rem;color:#7899a5;margin-top:5px">'
        'Adaptive Carbon-Aware AI Model Routing and Scheduling Framework</div>',
        unsafe_allow_html=True
    )

with footer_right:

    st.markdown(
        '<div style="text-align:right;font-size:.7rem;color:#7899a5">'
        'LOCAL RESEARCH CONSOLE<br>PYTHON · STREAMLIT</div>',
        unsafe_allow_html=True
    )














































































