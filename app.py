# app_lite.py
import streamlit as st
import pandas as pd
import config
from gen_values import get_raw_df, value_players   # adjust module names

st.set_page_config(page_title="Draft Assistant", layout="wide")

# ---------- 1️⃣  one-time init ----------
if "base_df" not in st.session_state:
    raw_df = get_raw_df()
    st.session_state["base_df"] = raw_df.assign(drafted=0)

# ---------- 2️⃣  interactive editor ----------
edited_df = st.data_editor(
    st.session_state["base_df"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "drafted": st.column_config.NumberColumn(
            "Draft Pick Number",  # Label for the column header
            min_value=0,  # set a minimum allowed integer value
            # max_value=250, # set a maximum allowed integer value
            step=1,  # set the step for the input (e.g., 1 for integers)
            format="%d"  # format the display as an integer
        )
    },
    key="draft_table"
)

# optional manual trigger
recalc = st.button("Re-compute rankings ↻", type="primary")

# ---------- 3️⃣  heavy calc (cached) ----------
@st.cache_data(show_spinner="Calculating values…")
def run_model(df: pd.DataFrame) -> pd.DataFrame:
    return value_players(df, projection_column_prefix=config.PROJECTION_COLUMN_PREFIX,
                         vopn=5, draft_mode=True).drop(columns=["id"]).set_index("player")

if recalc or st.session_state.get("first_run", True):
    result_df = run_model(edited_df)
    st.session_state["result_df"] = result_df
    st.session_state["first_run"] = False
else:
    result_df = st.session_state["result_df"]

# ---------- 4️⃣  display ----------
left, right = st.columns([4, 1])
with left:
    st.subheader("📈 Live rankings")
    st.dataframe(
        result_df.style.background_gradient(subset=["draft_value", "static_value", "dynamic_value"], cmap="RdYlGn"),
        use_container_width=True
    )
with right:
    # create small selection showing top 3 at each position
    top3 = result_df.groupby("position").head(3)[["position", "draft_value", "rank", "rank_pos"]]
    st.subheader("Pos. Top 3 Remaining")
    st.table(top3)

# reset button
if st.button("Reset draft"):
    st.session_state.clear()
    # st.experimental_rerun()
