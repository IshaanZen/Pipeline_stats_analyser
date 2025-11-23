import os
import json
import base64

import pandas as pd
import streamlit as st
# from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------------------------------------------
# 0. Setup: page config + API key
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Databricks Pipeline Stats Analyser",
    layout="wide"
)

# load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

st.title("Databricks Pipeline Stats Analyser")
st.write(
    "Prototype dashboard to explore Databricks pipeline run statistics "
    "and explain them in simple terms for non-technical viewers."
)

st.caption(f"🔐 OpenRouter API key loaded: {'✅ Yes' if OPENROUTER_API_KEY else '❌ No'}")


# -------------------------------------------------------------------
# 1. Helpers
# -------------------------------------------------------------------
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """Load local sample JSON as a fallback."""
    try:
        return pd.read_json("data/sample_data.json")
    except Exception as e:
        st.error(f"Error loading sample JSON data: {e}")
        return pd.DataFrame()


def validate_required_columns(df: pd.DataFrame) -> bool:
    required_cols = [
        "pipeline_name",
        "run_id",
        "start_time",
        "duration_min",
        "cpu_usage_percent",
        "memory_used_gb",
        "status",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns in data: {missing}")
        return False
    return True


st.sidebar.header("Use these filters to navigate through the analysis.")

# -------------------------------------------------------------------
# 2. Structured data mode
# -------------------------------------------------------------------
def render_structured_mode():
    """Main flow for CSV/JSON data → tables, charts, AI explanation."""

    st.sidebar.subheader("📁 Structured Data (CSV / JSON)")

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or JSON file",
        type=["csv", "json"]
    )

    # Decide which DataFrame to use
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            st.success("✅ Uploaded data loaded successfully.")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return
    else:
        st.info("No file uploaded. Using local data/sample_data.json for demo.")
        df = load_sample_data()

    if df.empty:
        st.warning("Data is empty. Please upload a valid file or fix sample_data.json.")
        return

    if not validate_required_columns(df):
        return

    # -------- Pipeline selector --------
    st.sidebar.subheader("Pipeline Filter")
    pipeline_options = sorted(df["pipeline_name"].unique())
    selected_pipeline = st.sidebar.selectbox("Select pipeline", pipeline_options)

    pipeline_df = df[df["pipeline_name"] == selected_pipeline].copy()
    pipeline_df = pipeline_df.sort_values("run_id")

    if pipeline_df.empty:
        st.warning("No runs found for the selected pipeline.")
        return

    # -------- Table --------
    st.subheader(f"📊 Run Metrics — `{selected_pipeline}`")
    st.dataframe(
        pipeline_df[[
            "run_id",
            "start_time",
            "duration_min",
            "cpu_usage_percent",
            "memory_used_gb",
            "status",
        ]],
        use_container_width=True,
    )

    # -------- Charts --------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**CPU Utilization per Run (%)**")
        cpu_chart_data = pipeline_df.set_index("run_id")[["cpu_usage_percent"]]
        st.line_chart(cpu_chart_data)

    with col2:
        st.markdown("**Memory Usage per Run (GB)**")
        mem_chart_data = pipeline_df.set_index("run_id")[["memory_used_gb"]]
        st.bar_chart(mem_chart_data)

    # -------- Aggregate stats --------
    st.subheader("📈 Aggregate Stats")

    avg_duration = pipeline_df["duration_min"].mean()
    avg_cpu = pipeline_df["cpu_usage_percent"].mean()
    avg_mem = pipeline_df["memory_used_gb"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Duration (min)", f"{avg_duration:.1f}")
    c2.metric("Average CPU (%)", f"{avg_cpu:.1f}")
    c3.metric("Average Memory (GB)", f"{avg_mem:.1f}")

    # -------- AI Explanation (OpenRouter, text-only) --------
    st.subheader("🤖 AI Explanation (Structured Data)")

    if not OPENROUTER_API_KEY:
        st.warning("No OpenRouter API key found in .env file. AI explanation is disabled.")
        return

    if "ai_response" not in st.session_state:
        st.session_state.ai_response = None

    if st.button("Generate AI Explanation from Metrics"):
        summary = {
            "pipeline_name": selected_pipeline,
            "total_runs": len(pipeline_df),
            "average_duration_min": float(avg_duration),
            "average_cpu_percent": float(avg_cpu),
            "average_memory_gb": float(avg_mem),
            "latest_run": pipeline_df.iloc[-1].to_dict(),
        }

        prompt = f"""
You are explaining Databricks pipeline stats to a non-technical manager.

Here is the pipeline summary (JSON-like):
{summary}

Explain in simple terms:
1. Overall performance and health.
2. Any risks (e.g., high CPU, high memory, failures).
3. 2–3 suggestions for improvement.

Keep it short and business-friendly.
"""

        try:
            client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
            )

            response = client.chat.completions.create(
                model="x-ai/grok-4.1-fast:free",
                messages=[
                    {
                        "role": "system",
                        "content": "You explain technical pipeline metrics in very simple business language.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            st.session_state.ai_response = response.choices[0].message.content

        except Exception as e:
            st.error(f"AI Error: {e}")

    if st.session_state.ai_response:
        st.markdown("### 🧠 AI Explanation")
        st.write(st.session_state.ai_response)


# -------------------------------------------------------------------
# 3. Image mode + LLM extraction
# -------------------------------------------------------------------
def render_image_mode():
    """
    Image-based mode: for managers who only have Databricks screenshots.
    - User uploads TWO images:
        1) Cluster/driver stats
        2) "Only executor nodes" stats
    - We display them
    - If API key exists, we:
        (1) send both to a vision model via OpenRouter to extract JSON stats
        (2) send that JSON to a text model to produce a plain-English explanation
    """

    st.sidebar.subheader("📸 Pipeline Screenshots")

    driver_image = st.sidebar.file_uploader(
        "Upload *Cluster / Driver* screenshot",
        type=["png", "jpg", "jpeg"],
        key="driver_image",
    )

    executor_image = st.sidebar.file_uploader(
        "Upload *Only Executor Nodes* screenshot",
        type=["png", "jpg", "jpeg"],
        key="executor_image",
    )

    if driver_image is None or executor_image is None:
        st.info(
            "👈 Please upload **both** screenshots:\n\n"
            "- The main cluster/driver stats panel\n"
            "- The 'Only Executor Nodes' stats panel\n\n"
            "Once uploaded, they will be shown below."
        )
        return

    st.subheader("📸 Uploaded Pipeline Screenshots")

    st.markdown("**1️⃣ Cluster / Driver Stats**")
    st.image(driver_image, caption="Cluster / Driver Stats Screenshot")

    st.markdown("**2️⃣ Only Executor Nodes Stats**")
    st.image(executor_image, caption="Only Executor Nodes Stats Screenshot")

    st.markdown("---")

    st.subheader("🤖 AI Extraction & Explanation from Screenshots (Beta)")

    if not OPENROUTER_API_KEY:
        st.warning("No OpenRouter API key found in .env file. Image-based AI is disabled.")
        return

    # Vision model (can handle images) and text model (cheaper, for explanation)
    vision_model = st.text_input(
        "Vision-capable model (for reading screenshots)",
        value="x-ai/grok-4.1-fast:free",
        help="Must support images on OpenRouter (e.g. openai/gpt-4.1-mini, openai/gpt-4o-mini).",
    )

    text_model = st.text_input(
        "Text model (for explanation)",
        value="x-ai/grok-4.1-fast:free",
        help="Any chat model on OpenRouter, used to generate the layman explanation.",
    )

    # Session state for persistence
    if "image_ai_parsed" not in st.session_state:
        st.session_state.image_ai_parsed = None
    if "image_ai_explanation" not in st.session_state:
        st.session_state.image_ai_explanation = None
    if "image_ai_raw" not in st.session_state:
        st.session_state.image_ai_raw = None

    if st.button("Extract stats & explain (from screenshots)"):

        # ---- 1) Encode images & call vision model to get JSON ----
        driver_b64 = base64.b64encode(driver_image.getvalue()).decode("utf-8")
        executor_b64 = base64.b64encode(executor_image.getvalue()).decode("utf-8")

        driver_url = f"data:image/png;base64,{driver_b64}"
        executor_url = f"data:image/png;base64,{executor_b64}"

        vision_user_content = [
            {
                "type": "text",
                "text": (
                    "These are two Databricks screenshots of cluster metrics. "
                    "The first is cluster/driver stats, the second is only executor nodes.\n\n"
                    "Please read the graphs and text, and output a STRICT JSON object only, "
                    "with this structure:\n\n"
                    "{\n"
                    '  \"driver_summary\": {\n'
                    '    \"approx_avg_cpu_percent\": <number>,\n'
                    '    \"approx_avg_memory_gb\": <number>,\n'
                    '    \"notes\": \"<short text>\"\n'
                    "  },\n"
                    '  \"executor_summary\": {\n'
                    '    \"approx_avg_cpu_percent\": <number>,\n'
                    '    \"approx_avg_memory_gb\": <number>,\n'
                    '    \"notes\": \"<short text>\"\n'
                    "  }\n"
                    "}\n\n"
                    "Do NOT include markdown or explanation outside the JSON. "
                    "If you are unsure, make best estimates."
                ),
            },
            {"type": "image_url", "image_url": {"url": driver_url}},
            {"type": "image_url", "image_url": {"url": executor_url}},
        ]

        try:
            client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
            )

            with st.spinner("📥 Reading screenshots & extracting metrics..."):
                resp = client.chat.completions.create(
                    model=vision_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You read Databricks cluster screenshots and output ONLY JSON "
                                "with approximate metrics."
                            ),
                        },
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
                st.warning("The vision model did not return valid JSON. Showing raw text instead.")

        except Exception as e:
            st.error(f"AI Error while processing images: {e}")
            return

        # ---- 2) If we have JSON, call text model for layman explanation ----
        if st.session_state.image_ai_parsed is not None:
            explanation_prompt = f"""
You are helping a non-technical project manager understand Databricks cluster performance.

Here is a JSON summary of driver and executor metrics extracted from screenshots:
{json.dumps(st.session_state.image_ai_parsed, indent=2)}

Based on this, explain in very simple, business-friendly language:
1. How the cluster is behaving overall.
2. Differences between driver and executor utilization.
3. Any risks or bottlenecks you see.
4. 2–3 practical suggestions (e.g. scale up/down, optimize jobs, schedule changes).

Avoid deep technical jargon. Keep it under 250 words.
"""

            try:
                with st.spinner("🧠 Generating plain-English explanation..."):
                    explain_resp = client.chat.completions.create(
                        model=text_model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You explain data engineering and infrastructure metrics "
                                    "in simple language for business stakeholders."
                                ),
                            },
                            {"role": "user", "content": explanation_prompt},
                        ],
                    )

                st.session_state.image_ai_explanation = explain_resp.choices[0].message.content

            except Exception as e:
                st.error(f"AI Error while generating explanation: {e}")

    # ---- Display results (JSON + explanation) ----
    if st.session_state.image_ai_parsed:
        st.markdown("### 📦 Extracted JSON Metrics")
        st.json(st.session_state.image_ai_parsed)

        # Tiny comparison table
        try:
            rows = []
            drv = st.session_state.image_ai_parsed.get("driver_summary", {})
            exe = st.session_state.image_ai_parsed.get("executor_summary", {})
            rows.append({
                "scope": "Driver",
                "avg_cpu (%)": drv.get("approx_avg_cpu_percent"),
                "avg_mem (GB)": drv.get("approx_avg_memory_gb"),
                "notes": drv.get("notes"),
            })
            rows.append({
                "scope": "Executors",
                "avg_cpu (%)": exe.get("approx_avg_cpu_percent"),
                "avg_mem (GB)": exe.get("approx_avg_memory_gb"),
                "notes": exe.get("notes"),
            })
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

# -------------------------------------------------------------------
# 4. Top-level mode selection
# -------------------------------------------------------------------
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

