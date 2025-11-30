# app.py
import os
import json
import base64

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import numbers
from datetime import datetime


# 0. Setup: page config + API key

st.set_page_config(
    page_title="Databricks Cluster Stats Analyser",
    layout="wide",
)

# Small UI polish (hide anchor links, set a subtle background card)
st.markdown(
    """
    <style>
    a.anchor-link { display: none !important; }
    .stApp { font-family: Inter, Arial, sans-serif; }
    .card { padding: 12px; border-radius: 8px; background: #ffffff; box-shadow: 0 1px 6px rgba(20,20,20,0.04); }
    .muted { color: #6c757d; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load env
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Header (centered)
st.markdown("<h1 style='text-align: center; margin-bottom: 2px;'>🚀 Databricks Cluster Stats Analyser</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; margin-top:0; color:#6c757d'>Turn cluster metrics and screenshots into clear, business-friendly insights.</p>", unsafe_allow_html=True)
col_head_1, col_head_2, col_head_3 = st.columns([3, 1, 0.7])
with col_head_1:
    st.markdown("<h5>Use this tool to inspect cluster health, costs, and identify risks — both from structured metrics and screenshots.</h5>", unsafe_allow_html=True)
with col_head_2:
    st.metric("App Status", "Online", "beta")
with col_head_3:
    st.caption("AI ANALYSIS: " + ("✅ Yes" if OPENROUTER_API_KEY else "❌ No"))
st.markdown("---")


# 1. Helpers functions

def make_jsonable(obj):
    """
    Recursively convert pandas/np types and datetimes to JSON-serializable Python types.
    Use on dicts/lists produced from DataFrame.to_dict(...) or row.to_dict().
    """
    # primitives
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, numbers.Number):
        # convert numpy numbers to native Python numbers
        try:
            return obj.item()
        except Exception:
            return obj
    # datetimes
    if isinstance(obj, (pd.Timestamp, datetime)):
        # ISO 8601 string
        return obj.isoformat()
    # numpy datetime64
    try:
        if isinstance(obj, np.datetime64):
            # convert to python datetime then isoformat
            return pd.to_datetime(obj).isoformat()
    except Exception:
        pass
    # dict
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
    # fallback for numpy scalars
    try:
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    # last resort: string conversion
    return str(obj)



@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """Load local sample JSON as a fallback and sanitize values."""
    path = "Data/cluster_sample_data.json"
    if not os.path.exists(path):
        st.error(f"Error loading sample JSON data: {path} does not exist")
        return pd.DataFrame()
    try:
        df = pd.read_json(path)
        # parse timestamp if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = sanitize_df(df)
        return df
    except Exception as e:
        st.error(f"Error loading sample JSON data: {e}")
        return pd.DataFrame()


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive sanitization: clip CPU% to [0,100], ensure memory not negative, fill missing columns if needed."""
    df = df.copy()
    # CPU columns
    cpu_cols = [c for c in df.columns if "cpu_usage_percent" in c]
    for c in cpu_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)

    # Memory columns
    mem_cols = [c for c in df.columns if "memory" in c or "mem" in c]
    for c in mem_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    # worker memory per node presence
    if "worker_mem_gb_per_node" in df.columns:
        df["worker_mem_gb_per_node"] = pd.to_numeric(df["worker_mem_gb_per_node"], errors="coerce").fillna(1.0)
    return df


def compute_derived_metrics(cluster_df: pd.DataFrame) -> dict:
    """Create a small summary dict with derived metrics used for the LLM and UI highlights."""
    out = {}
    if cluster_df.empty:
        return out

    out["samples"] = len(cluster_df)
    out["avg_worker_cpu_pct"] = float(cluster_df["cpu_usage_percent_workers_avg"].mean())
    out["p95_worker_cpu_pct"] = float(cluster_df["cpu_usage_percent_workers_avg"].quantile(0.95))
    out["peak_worker_cpu_pct"] = float(cluster_df["cpu_usage_percent_workers_avg"].max())
    out["avg_driver_cpu_pct"] = float(cluster_df["cpu_usage_percent_driver"].mean())
    out["peak_driver_cpu_pct"] = float(cluster_df["cpu_usage_percent_driver"].max())

    out["avg_worker_mem_gb"] = float(cluster_df["memory_used_gb_workers_avg"].mean())
    out["peak_worker_mem_gb"] = float(cluster_df["memory_used_gb_workers_avg"].max())

    # memory % relative to node (if node mem present)
    if "worker_mem_gb_per_node" in cluster_df.columns:
        node_mem = float(cluster_df["worker_mem_gb_per_node"].iloc[0])
        out["avg_worker_mem_pct_of_node"] = float((out["avg_worker_mem_gb"] / node_mem) * 100)
    else:
        out["avg_worker_mem_pct_of_node"] = None

    out["avg_workers"] = float(cluster_df["num_workers"].mean())
    out["total_estimated_cost"] = float(cluster_df["cost_estimated_usd"].sum())
    out["failed_runs_total"] = int(cluster_df["failed_jobs"].sum()) if "failed_jobs" in cluster_df.columns else 0

    return out


st.sidebar.header("Use these filters to navigate through the analysis.")


# 2. Structured data mode

def render_structured_mode():
    """Main flow for CSV/JSON data → tables, altair charts, AI explanation."""

    st.sidebar.subheader("Structured Data (CSV / JSON)")
    uploaded_file = st.sidebar.file_uploader("Upload CSV or JSON file", type=["csv", "json"])

    # Choose DataFrame
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = sanitize_df(df)
            st.success("Uploaded data loaded successfully.")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return
    else:
        st.info("No file uploaded. Using local data/cluster_sample_data.json for demo.")
        df = load_sample_data()

    if df.empty:
        st.warning("Data is empty. Please upload a valid file or add cluster_sample_data.json.")
        return

    # Validate
    required = [
        "cluster_id",
        "cluster_size",
        "timestamp",
        "num_workers",
        "cpu_usage_percent_workers_avg",
        "cpu_usage_percent_driver",
        "memory_used_gb_workers_avg",
        "memory_used_gb_driver",
        "running_jobs",
        "failed_jobs",
        "cost_estimated_usd",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in data: {missing}")
        return

    # Cluster selector
    st.sidebar.subheader("Cluster Filter")
    cluster_options = sorted(df["cluster_id"].unique())
    selected_cluster = st.sidebar.selectbox("Select cluster (by cluster_id)", cluster_options)

    cluster_df = df[df["cluster_id"] == selected_cluster].copy().sort_values("timestamp")
    if cluster_df.empty:
        st.warning("No samples found for the selected cluster.")
        return

    # Top summary cards
    derived = compute_derived_metrics(cluster_df)
    r1, r2, r3, r4 = st.columns([2, 2, 2, 2])
    r1.metric("Avg Worker CPU (%)", f"{derived['avg_worker_cpu_pct']:.1f}")
    r2.metric("Peak Worker CPU (%)", f"{derived['peak_worker_cpu_pct']:.1f}")
    r3.metric("Avg Worker Mem (GB)", f"{derived['avg_worker_mem_gb']:.2f}")
    r4.metric("Estimated Cost (sum)", f"${derived['total_estimated_cost']:.4f}")

    st.markdown("### Cluster sample details")
    st.dataframe(
        cluster_df[[
            "timestamp",
            "num_workers",
            "cpu_usage_percent_workers_avg",
            "cpu_usage_percent_driver",
            "memory_used_gb_workers_avg",
            "memory_used_gb_driver",
            "running_jobs",
            "failed_jobs",
            "cost_estimated_usd",
        ]],
        use_container_width=True
    )

    # Charts: Altair for labeled axes and colors

    st.markdown("### Visual Trends")
    cpu_chart_df = cluster_df[["timestamp", "cpu_usage_percent_workers_avg", "cpu_usage_percent_driver"]].melt("timestamp", var_name="series", value_name="value")
    cpu_chart = (
        alt.Chart(cpu_chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp:T", axis=alt.Axis(title="Time")),
            y=alt.Y("value:Q", axis=alt.Axis(title="CPU %")),
            color=alt.Color("series:N", scale=alt.Scale(domain=["cpu_usage_percent_workers_avg", "cpu_usage_percent_driver"],
                                                         range=["#1f77b4", "#ff7f0e"]),
                            legend=alt.Legend(title="Series", orient="bottom")),
            tooltip=["timestamp:T", "series:N", "value:Q"],
        )
        .properties(height=300)
    )
    mem_chart_df = cluster_df[["timestamp", "memory_used_gb_workers_avg", "memory_used_gb_driver"]].melt("timestamp", var_name="series", value_name="value")
    mem_chart = (
        alt.Chart(mem_chart_df)
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X("timestamp:T", axis=alt.Axis(title="Time")),
            y=alt.Y("value:Q", axis=alt.Axis(title="Memory (GB)")),
            color=alt.Color("series:N", scale=alt.Scale(domain=["memory_used_gb_workers_avg", "memory_used_gb_driver"],
                                                       range=["#2ca02c", "#d62728"]),
                            legend=alt.Legend(title="Series", orient="bottom")),
            tooltip=["timestamp:T", "series:N", "value:Q"],
        )
        .properties(height=300)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(cpu_chart, use_container_width=True)
    with col2:
        st.altair_chart(mem_chart, use_container_width=True)

    # Worker count and cost over time
    wc = alt.Chart(cluster_df).mark_line(point=True).encode(
        x=alt.X("timestamp:T", axis=alt.Axis(title="Time")),
        y=alt.Y("num_workers:Q", axis=alt.Axis(title="Workers")),
        tooltip=["timestamp:T", "num_workers:Q"],
        color=alt.value("#9467bd"),
    ).properties(height=220)

    costc = alt.Chart(cluster_df).mark_bar().encode(
        x=alt.X("timestamp:T", axis=alt.Axis(title="Time")),
        y=alt.Y("cost_estimated_usd:Q", axis=alt.Axis(title="Estimated Cost (USD)")),
        tooltip=["timestamp:T", "cost_estimated_usd:Q"],
        color=alt.value("#8c564b"),
    ).properties(height=220)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Workers over time**")
        st.altair_chart(wc, use_container_width=True)
    with c2:
        st.markdown("**Estimated cost per sample window**")
        st.altair_chart(costc, use_container_width=True)

    # Quick rule-based cluster health summary
    st.subheader("Quick cluster health summary (rule-based)")
    summary_lines = []
    if derived["avg_worker_cpu_pct"] > 85 or derived["avg_driver_cpu_pct"] > 85:
        summary_lines.append("• CPU is running very high on average — possible performance risk.")
    elif derived["avg_worker_cpu_pct"] < 40 and derived["avg_driver_cpu_pct"] < 40:
        summary_lines.append("• CPU utilisation is low — possible over-provisioning.")

    if derived["avg_worker_mem_pct_of_node"] is not None and derived["avg_worker_mem_pct_of_node"] > 75:
        summary_lines.append("• Memory usage per worker is high relative to node memory — review caching/shuffles.")
    else:
        summary_lines.append("• Memory usage per worker is within expected bounds.")

    summary_lines.append(f"• Typical sample uses ~{derived['avg_workers']:.1f} workers with avg worker CPU {derived['avg_worker_cpu_pct']:.1f}%.")

    with st.expander("Show quick summary"):
        for ln in summary_lines:
            st.write(ln)

    # LLM-based explanation (structured data) 

    st.subheader("🤖 AI Explanation (Structured Data)")

    if not OPENROUTER_API_KEY:
        st.info("🔑 No OpenRouter API key configured. AI explanation is disabled for structured data.")
    else:
        st.session_state.setdefault("ai_response_structured", None)
        if st.button("Generate AI Explanation from Cluster Metrics"):
            latest = cluster_df.iloc[-1].to_dict()
            latest_safe = make_jsonable(latest)
            summary_payload = {
                "cluster_id": selected_cluster,
                "cluster_size": latest_safe.get("cluster_size"),
                "samples": derived["samples"],
                "avg_worker_cpu_pct": derived["avg_worker_cpu_pct"],
                "p95_worker_cpu_pct": derived["p95_worker_cpu_pct"],
                "peak_worker_cpu_pct": derived["peak_worker_cpu_pct"],
                "avg_driver_cpu_pct": derived["avg_driver_cpu_pct"],
                "avg_worker_mem_gb": derived["avg_worker_mem_gb"],
                "avg_worker_mem_pct_of_node": derived["avg_worker_mem_pct_of_node"],
                "avg_workers": derived["avg_workers"],
                "total_estimated_cost": derived["total_estimated_cost"],
                "failed_runs_total": derived["failed_runs_total"],
                "latest_sample": latest_safe,
            }

            prompt = f"""
You are summarizing Databricks cluster metrics for a non-technical business manager.

Below is the cluster summary data (JSON-like):
{json.dumps(summary_payload, indent=2)}

Your task:
1. Provide a clear, simple explanation of the overall cluster health and performance.
2. Highlight any risks, inefficiencies, or bottlenecks that could impact reliability or cost.
3. Give 2-3 practical, business-oriented recommendations for improvement (examples: reducing cost, right-sizing the cluster, improving job scheduling, avoiding failures).

Guidelines:
- Keep the explanation concise and business-friendly.
- Avoid technical jargon unless absolutely necessary.
- Focus on what matters to leadership: reliability, cost, scalability, and efficiency.
- Don't list raw metrics—translate them into meaningful insights.
- Use short paragraphs or bullet points for clarity.



"""
            try:
                client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
                response = client.chat.completions.create(
                    model="x-ai/grok-4.1-fast:free",
                    messages=[
                        {"role": "system", "content": "Explain cluster metrics in plain business language."},
                        {"role": "user", "content": prompt},
                    ],
                )
                st.session_state.ai_response_structured = response.choices[0].message.content
            except Exception as e:
                st.error(f"AI Error: {e}")

        if st.session_state.ai_response_structured:
            st.markdown("### 🧠 AI Explanation (Structured Data)")
            st.write(st.session_state.ai_response_structured)


# 3. Image mode + LLM extraction & explanation
def render_image_mode():
    """
    Image-based mode: upload two screenshots.
    The vision model is asked to return an expanded JSON schema including:
      - driver_summary: avg cpu, memory, cpu_breakdown (user/system/iowait/steal), notes
      - executor_summary: avg cpu, memory, cpu_breakdown, notes
      - worker_table_summary: alive_workers, dead_workers
      - anomalies: list of short strings
    """

    st.sidebar.subheader("📸 Cluster Screenshots (Upload both)")
    driver_image = st.sidebar.file_uploader("Upload Cluster / Driver screenshot", type=["png", "jpg", "jpeg"], key="driver_image")
    executor_image = st.sidebar.file_uploader("Upload Executors screenshot", type=["png", "jpg", "jpeg"], key="executor_image")

    if driver_image is None or executor_image is None:
        st.info("👈 Please upload both screenshots (cluster/driver + executors).")
        return

    st.subheader("📸 Uploaded Cluster Screenshots")
    st.markdown("**1️⃣ Cluster / Driver Stats**")
    st.image(driver_image, caption="Cluster / Driver Stats")
    st.markdown("**2️⃣ Executor Nodes Stats**")
    st.image(executor_image, caption="Executor Nodes Stats")
    st.markdown("---")

    st.subheader("🤖 AI Extraction & Explanation from Screenshots (Beta)")
    if not OPENROUTER_API_KEY:
        st.info("🔑 No OpenRouter API key configured. Image-based AI extraction is disabled.")
        return

    vision_model = st.text_input("Vision-capable model (for reading screenshots)", value="x-ai/grok-4.1-fast:free")
    text_model = st.text_input("Text model (for explanation)", value="x-ai/grok-4.1-fast:free")

    st.session_state.setdefault("image_ai_parsed", None)
    st.session_state.setdefault("image_ai_raw", None)
    st.session_state.setdefault("image_ai_explanation", None)

    if st.button("Extract stats & explain (from screenshots)"):
        driver_b64 = base64.b64encode(driver_image.getvalue()).decode("utf-8")
        executor_b64 = base64.b64encode(executor_image.getvalue()).decode("utf-8")
        driver_url = f"data:image/png;base64,{driver_b64}"
        executor_url = f"data:image/png;base64,{executor_b64}"

        # Expanded JSON schema requested
        vision_user_text = """
You are reading two Databricks cluster metric screenshots (driver + executors).
Return STRICT JSON ONLY with the exact schema below (no extra text):

{
  "driver_summary": {
    "approx_avg_cpu_percent": <number>,
    "approx_avg_memory_gb": <number>,
    "cpu_breakdown_pct": {"user":<n>,"system":<n>,"iowait":<n>,"steal":<n>},
    "notes": "<short text>"
  },
  "executor_summary": {
    "approx_avg_cpu_percent": <number>,
    "approx_avg_memory_gb": <number>,
    "cpu_breakdown_pct": {"user":<n>,"system":<n>,"iowait":<n>,"steal":<n>},
    "notes": "<short text>"
  },
  "worker_table_summary": {
    "alive_workers": <int>,
    "dead_workers": <int>
  },
  "anomalies": ["<short text>", "..."]
}

If you cannot read exact numbers, provide your best reasonable estimates. Use numeric values for numbers, and brief 1-2 line notes. Do NOT include markdown or explanation outside this JSON.
"""

        vision_user_content = [
            {"type": "text", "text": vision_user_text},
            {"type": "image_url", "image_url": {"url": driver_url}},
            {"type": "image_url", "image_url": {"url": executor_url}},
        ]

        try:
            client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
            with st.spinner("Reading screenshots & extracting metrics..."):
                resp = client.chat.completions.create(
                    model=vision_model,
                    messages=[
                        {"role": "system", "content": "Read Databricks cluster screenshots and output ONLY the requested JSON."},
                        {"role": "user", "content": vision_user_content},
                    ],
                )
            raw_text = resp.choices[0].message.content
            st.session_state.image_ai_raw = raw_text
            try:
                parsed = json.loads(raw_text)
                st.session_state.image_ai_parsed = parsed
            except json.JSONDecodeError:
                st.session_state.image_ai_parsed = None
                st.warning("Vision model did not return strict JSON. Showing raw text instead.")
        except Exception as e:
            st.error(f"AI Error while processing images: {e}")
            return

        # If parsed JSON exists -> explain it
        if st.session_state.image_ai_parsed:
            explanation_prompt = f"""
You are helping a non-technical project manager understand Databricks cluster performance.

Here is a JSON summary extracted from screenshots:
{json.dumps(st.session_state.image_ai_parsed, indent=2)}

Based on this, explain in simple business language:
1. Overall cluster behaviour.
2. Differences between driver and executor utilization.
3. Any risks and 2 practical suggestions (scaling, cost, optimization).

Keep it under 200 words and avoid deep technical jargon.
"""
            try:
                with st.spinner("Generating plain-English explanation..."):
                    explain_resp = client.chat.completions.create(
                        model=text_model,
                        messages=[
                            {"role": "system", "content": "Explain cluster metrics in simple business-friendly language."},
                            {"role": "user", "content": explanation_prompt},
                        ],
                    )
                st.session_state.image_ai_explanation = explain_resp.choices[0].message.content
            except Exception as e:
                st.error(f"AI Error while generating explanation: {e}")

    # Display parsed JSON and explanation
    if st.session_state.image_ai_parsed:
        st.markdown("### Extracted JSON Metrics (from images)")
        st.json(st.session_state.image_ai_parsed)

        # try to show a small comparison table
        try:
            drv = st.session_state.image_ai_parsed.get("driver_summary", {})
            exe = st.session_state.image_ai_parsed.get("executor_summary", {})
            rows = [
                {"scope": "Driver", "avg_cpu (%)": drv.get("approx_avg_cpu_percent"), "avg_mem (GB)": drv.get("approx_avg_memory_gb"), "notes": drv.get("notes")},
                {"scope": "Executors", "avg_cpu (%)": exe.get("approx_avg_cpu_percent"), "avg_mem (GB)": exe.get("approx_avg_memory_gb"), "notes": exe.get("notes")},
            ]
            df_image_stats = pd.DataFrame(rows)
            st.subheader("📊 Approximate Metrics (Driver vs Executors)")
            st.dataframe(df_image_stats, use_container_width=True)
        except Exception:
            pass
    elif st.session_state.image_ai_raw:
        st.markdown("### 🧠 Raw AI response (could not parse JSON)")
        st.write(st.session_state.image_ai_raw)

    if st.session_state.image_ai_explanation:
        st.markdown("### 📝 Plain-English Explanation for Stakeholders")
        st.write(st.session_state.image_ai_explanation)


# 4. Top-level mode selection
st.sidebar.markdown("---")
st.sidebar.header("Data Input Type")

data_type = st.sidebar.selectbox(
    "What type of data do you have?",
    options=[
        "Please select an option",
        "Structured data (CSV / JSON)",
        "Images (Pipeline Screenshots)",
    ],
)

if data_type == "Structured data (CSV / JSON)":
    render_structured_mode()
elif data_type == "Images (Pipeline Screenshots)":
    render_image_mode()
else:
    st.info("👈 Please select a data input type from the sidebar to get started.")
