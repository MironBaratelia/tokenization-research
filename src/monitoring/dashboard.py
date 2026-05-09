import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

project_root = Path(__file__).parent.parent.parent
outputs_dir = project_root / "outputs"


def load_experiment_state(exp_path):
    state_file = exp_path / "experiment_state.json"
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_metrics(exp_path):
    metrics_file = exp_path / "logs" / "metrics.jsonl"
    if metrics_file.exists():
        data = []
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)
    return None


st.set_page_config(page_title="MIRON Dashboard 2.0", layout="wide")
st.title("MIRON Experiment Dashboard 2.0")

st.sidebar.header("Experiments")
all_exps = [d for d in outputs_dir.iterdir() if d.is_dir()]
selected_exps = st.sidebar.multiselect(
    "Select Experiments",
    [d.name for d in all_exps],
    default=[d.name for d in all_exps][:1] if all_exps else [],
)

if not selected_exps:
    st.warning("Please select at least one experiment in the sidebar.")
else:
    tabs = st.tabs(["Overview", "Training Curves", "Manifold Analysis", "Evaluation"])

    with tabs[0]:
        st.header("Experiment Metadata")
        for exp_name in selected_exps:
            exp_path = outputs_dir / exp_name
            state = load_experiment_state(exp_path)
            if state:
                with st.expander(f"Metadata: {exp_name}"):
                    col1, col2 = st.columns(2)
                    col1.metric("Start Time", state.get("start_time", "N/A"))
                    col1.metric("Status", "Finished" if state.get("end_time") else "Running")
                    col2.metric("Checkpoints", len(state.get("checkpoints", [])))
                    col2.json(state.get("config", {}))

    with tabs[1]:
        st.header("Loss & Metrics")
        for exp_name in selected_exps:
            df = load_metrics(outputs_dir / exp_name)
            if df is not None:
                st.subheader(f"Experiment: {exp_name}")
                train_df = df[df["phase"] == "train"] if "phase" in df.columns else df

                loss_cols = [
                    c for c in train_df.columns if "loss" in c.lower() and "val" not in c.lower()
                ]
                if loss_cols:
                    fig = px.line(train_df, x="step", y=loss_cols, title=f"Training Loss: {exp_name}")
                    st.plotly_chart(fig, use_container_width=True)

                if "Speed/TPS" in train_df.columns:
                    fig_speed = px.line(
                        train_df,
                        x="step",
                        y="Speed/TPS",
                        title=f"Throughput (TPS): {exp_name}",
                    )
                    st.plotly_chart(fig_speed, use_container_width=True)

    with tabs[2]:
        st.header("Manifold Analysis")
        st.info("Visualizing manifold metrics logged during training.")

    with tabs[3]:
        st.header("Evaluation")
