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
        "drafted": st.column_config.CheckboxColumn("🚫 Drafted")
    },
    key="draft_table"
)

# optional manual trigger
recalc = st.button("Re-compute rankings ↻", type="primary")

# ---------- 3️⃣  heavy calc (cached) ----------
@st.cache_data(show_spinner="Calculating values…")
def run_model(df: pd.DataFrame) -> pd.DataFrame:
    return value_players(df, projection_column_prefix=config.PROJECTION_COLUMN_PREFIX,
                         vopn=5, draft_mode=True)

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
        result_df.style.background_gradient(
            subset=["draft_value"], cmap="Greens"
        ),
        use_container_width=True
    )
with right:
    top5 = result_df.head(5)[["player", "draft_value", "position"]]
    st.subheader("📝 Next-ups")
    st.table(top5)

# reset button
if st.button("Reset draft"):
    st.session_state.clear()
    # st.experimental_rerun()
