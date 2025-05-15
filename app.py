import streamlit as st
import config
from utils_app import (
    initialize_draft_state,
    render_draft_board,
    render_rankings,
    render_position_top_picks,
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
    st.subheader("Pos. Top 3 Remaining")
    render_position_top_picks()

# Add reset button at the bottom
if st.button("Reset draft"):
    reset_draft()