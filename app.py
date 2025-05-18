import streamlit as st
import config
from utils_app import (
    initialize_draft_state,
    render_draft_board,
    render_rankings,
    render_position_top_picks,
    write_current_draft_pick,
    write_draft_notes,
    render_draft_history,
    render_my_roster,
    render_placeholder_table,
    re_calc_rankings,
    render_full_draft_board,
    reset_draft
)

# Set page configuration
st.set_page_config(page_title="Draft Assistant", layout="wide")

# Initialize app state on first load
if "base_df" not in st.session_state:
    initialize_draft_state()

# Create page layout with three columns
left, middle, right = st.columns([3, 6, 2])

# Render the interactive draft board in left column
with left:
    st.subheader("Draft Board")
    render_draft_board()

# Render live rankings in middle column
with middle:
    st.subheader("Live Rankings")
    render_rankings()

# Render top picks by position in right column
with right:
    st.subheader("Top Remaining")
    render_position_top_picks()

# Add button to manually clear cache and recalculate
if st.button("Re-compute rankings ↻", type="primary"):
    re_calc_rankings()

# Add a second row of three columns for additional tables
st.write("---")  # Add horizontal separator

round_num = write_current_draft_pick()
write_draft_notes(round_num)
# todo: add relative performance by position, e.g., "Rank 1/10 in RBs, Rank 6/10 in WRs" -- to help with position targeting

st.write("---")  # Add horizontal separator

bottom_left, bottom_middle, bottom_right = st.columns(3)

# Render draft history table
with bottom_left:
    st.subheader("Draft History")
    render_draft_history()

# Render my roster table
with bottom_middle:
    st.subheader("My Roster")
    render_my_roster()

# Render placeholder table
with bottom_right:
    st.subheader("Position Group Details")
    render_placeholder_table()

st.write("---")  # Add another separator
st.subheader("Complete Draft Board")
render_full_draft_board()

# Add reset button at the bottom
if st.button("Reset draft"):
    reset_draft()

# TODO: add a daraframe with teams as columns, and one row per position (QB, RB, WR, TE, BN). Each cell will be the sum of that team's median_projection grouped by position for the starting roster spots,
#  e.g., Top 1 QB, Top 2 RB, Top 2 WR, Top 1 TE, and Top 2 remaining RB/WR/TE.
#  Rest of players median_projections will sum into a BN row.
#  Conditionally formatted across the row, so the team with the most QB points will be green, and the team with the least QB points will be Red, etc. etc. for each row/position.