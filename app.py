# app_lite.py
import streamlit as st
import pandas as pd
import config
from gen_values import get_raw_df, value_players

# --- constants you can tweak ---------------------------------
VISIBLE_EDIT_COLS = ["team", "position", "drafted"]
INT_COLS          = {"drafted", "rank", "rank_pos", "rank_pos_team"}
ADD_KEEPERS = True
KEEPERS = {'breece hall': 21,
           'terry mclaurin': 50,
           'jayden daniels': 74,
           'ladd mcconkey': 77,
           'jaxon smith-njigba': 78,
           'chuba hubbard': 99,
           "de'von achane": 105,
           'puka nacua': 132,
           'kyren williams': 133,
           'rashid shaheed': 135}
# --------------------------------------------------------------

st.set_page_config(page_title="Draft Assistant", layout="wide")

# ---------- 1️⃣  one-time init ----------
if "base_df" not in st.session_state:
    if not ADD_KEEPERS:
        raw_df = get_raw_df()
        st.session_state["base_df"] = raw_df.assign(drafted=0)
    else:
        raw_df = get_raw_df()
        raw_df = raw_df.assign(drafted=0)
        for k, v in KEEPERS.items():
            raw_df.loc[raw_df['player'] == k, 'drafted'] = v
        st.session_state["base_df"] = raw_df

# ---------- design page layout ----------
left, middle, right = st.columns([3, 6, 2])

# ---------- draft board: interactive editor ----------
edit_view = (
    st.session_state["base_df"]
    .set_index("player")                # <-- row labels become names, saves a column
    [VISIBLE_EDIT_COLS]                 # keep only the slim set
)

with left:
    st.subheader("Draft Board")
    edited_view = st.data_editor(
        edit_view,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "drafted": st.column_config.NumberColumn("drafted", min_value=0, step=1, format="%d")
        },
        key="draft_table"
    )
    # pull the drafted column out as a Series
    draft_updates = edited_view["drafted"]

    # map back by player name
    st.session_state["base_df"]["drafted"] = (st.session_state["base_df"]["player"].map(draft_updates).fillna(0).astype(int))

# ---------- 3️⃣  heavy calc (cached) ----------
@st.cache_data(show_spinner="Calculating values…")
def run_model(df: pd.DataFrame) -> pd.DataFrame:
    return (
        value_players(
            df,
            projection_column_prefix=config.PROJECTION_COLUMN_PREFIX,
            vopn=5,
            draft_mode=True,
        )
        .drop(columns=["id"])
        .set_index("player")
    )

# ⏩ Auto-recompute every rerun (cheap if df hash unchanged)
result_df = run_model(st.session_state["base_df"])

# 🔄 Manual fallback: clear cache & re-run
if st.button("Re-compute rankings ↻", type="primary"):
    run_model.clear()                # toss cached result
    result_df = run_model(st.session_state["base_df"])

# ---------- draft board styler ----------
def round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    num_cols = df2.select_dtypes("number").columns.difference(INT_COLS)
    # df2[num_cols] = df2[num_cols].round(1)
    df2[num_cols] = df2[num_cols].astype(int) # todo: place holder, working on this, should be one decimal
    df2[list(INT_COLS)] = df2[list(INT_COLS)].astype(int)
    return df2

cols_coolwarm = ["rank", "rank_pos", "rank_pos_team"]
cols_rdylgn = (["draft_value", "static_value", "dynamic_value", "mkt_share", "available_pts"]
               + [col for col in result_df.columns if col.startswith('value_')]
               + [col for col in result_df.columns if col.endswith('_projection')])

styler = (
    round_numeric(result_df)
      .style
      .background_gradient(subset=cols_rdylgn, cmap="RdYlGn")
      .background_gradient(subset=cols_coolwarm, cmap="coolwarm")
      .format(precision=1)
)

# ---------- draft board ----------
with middle:
    st.subheader("Live Rankings")
    display_df = round_numeric(result_df)
    st.dataframe(styler, use_container_width=True, height=750)

# ---------- top remaining players by position ----------
with right:
    # create small selection showing top 3 at each position
    top3 = result_df.groupby("position").head(3)[["position", "draft_value", "rank", "rank_pos"]]
    top3.sort_values(["position", "draft_value"], ascending=[True, False], inplace=True)
    top3[["draft_value", "rank", "rank_pos"]] = top3[["draft_value", "rank", "rank_pos"]].astype(int)
    st.subheader("Pos. Top 3 Remaining")
    st.table(top3)

# reset button
if st.button("Reset draft"):
    st.session_state.clear()
    # st.experimental_rerun()
