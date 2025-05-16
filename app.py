import streamlit as st
import config
from utils_app import (
    initialize_draft_state,
    write_current_draft_pick,
    render_draft_board,
    render_rankings,
    render_position_top_picks,
    render_draft_history,
    render_my_roster,
    render_placeholder_table,
    re_calc_rankings,
    reset_draft
)

# Set page configuration
st.set_page_config(page_title="Draft Assistant", layout="wide")

# Initialize app state on first load
if "base_df" not in st.session_state:
    initialize_draft_state()

round_num = write_current_draft_pick()
st.write(f"Tip: {config.draft_tips[round_num]}")
st.write("---")  # Add horizontal separator

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
    st.subheader("Pos. Top 3 Remaining")
    render_position_top_picks()

# Add button to manually clear cache and recalculate
if st.button("Re-compute rankings ↻", type="primary"):
    re_calc_rankings()

# Add a second row of three columns for additional tables
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
    st.subheader("Future Features")
    render_placeholder_table()

# Add reset button at the bottom
if st.button("Reset draft"):
    reset_draft()