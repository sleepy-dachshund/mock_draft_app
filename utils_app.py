"""
Fantasy Football Draft Assistant
================================

This Streamlit app provides a real-time fantasy football draft assistant that:
1. Displays an interactive draft board for marking players as drafted
2. Calculates and updates player values based on draft progress
3. Shows live rankings with color-coded information for easy decision-making
4. Provides top remaining players by position
5. Allows resetting the draft state

The app calculates player value using both static projections and dynamic
value calculations that account for positional scarcity as the draft progresses.
"""

import streamlit as st
import pandas as pd
import config
from gen_values import get_raw_df, value_players

# Global parameters
MY_DRAFT_PICK = config.DRAFT_POSITION
N_TEAMS = config.N_TEAMS
VISIBLE_EDIT_COLS = ["team", "position", "drafted"]
INT_COLS = {"drafted", "rank", "rank_pos", "rank_pos_team"}

# Position-based color scheme for visual differentiation
POS_COLORS = {
    "QB": "#d0e7ff",  # light blue
    "RB": "#d7f9d7",  # light green
    "WR": "#eadcff",  # light purple
    "TE": "#ffe9d6",  # light orange
}

# Keeper settings
ADD_KEEPERS = True
KEEPERS = {
    'breece hall': 21,
    'terry mclaurin': 50,
    'jayden daniels': 74,
    'ladd mcconkey': 77,
    'jaxon smith-njigba': 78,
    'chuba hubbard': 99,
    "de'von achane": 105,
    'puka nacua': 132,
    'kyren williams': 133,
    'rashid shaheed': 135
}


def initialize_draft_state():
    """
    Initialize the app's draft state with player data and keeper information.

    This function is called once when the app starts. It loads player data,
    marks any keepers as drafted, and stores the resulting DataFrame in
    the session state.

    Returns
    -------
    None
        Results are stored in st.session_state["base_df"]
    """
    raw_df = get_raw_df()
    raw_df = raw_df.assign(drafted=0)

    # Mark keepers as drafted if enabled
    if ADD_KEEPERS:
        for player, pick_num in KEEPERS.items():
            raw_df.loc[raw_df['player'] == player, 'drafted'] = pick_num

    # Store in session state
    st.session_state["base_df"] = raw_df

    # Initialize the editable view for the draft board
    st.session_state["edit_view"] = (
        raw_df.set_index("player")[VISIBLE_EDIT_COLS]
    )


def render_draft_board():
    """
    Render the interactive draft board that allows marking players as drafted.

    This function displays an editable data table where users can mark
    players as drafted by changing the "drafted" column value. Changes
    are automatically synchronized with the base DataFrame.

    Returns
    -------
    None
        Updates st.session_state["base_df"] based on user edits
    """
    # Display editable data table
    edited_view = st.data_editor(
        st.session_state["edit_view"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "drafted": st.column_config.NumberColumn(
                "drafted", min_value=0, step=1, format="%d"
            )
        },
        key="draft_table"
    )

    # Extract drafted status and update the base DataFrame
    draft_updates = edited_view["drafted"]
    st.session_state["base_df"]["drafted"] = (
        st.session_state["base_df"]["player"]
        .map(draft_updates)
        .fillna(0)
        .astype(int)
    )


@st.cache_data(show_spinner="Calculating values…")
def run_model(df):
    """
    Run the player valuation model with the current draft state.

    This function calculates player values based on projections and
    the current draft state. Results are cached for performance.

    Parameters
    ----------
    df : pandas.DataFrame
        The current player data including drafted status

    Returns
    -------
    pandas.DataFrame
        DataFrame with calculated player values and rankings
    """
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


def round_numeric(df):
    """
    Round numeric values in the DataFrame for display.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing player data and calculations

    Returns
    -------
    pandas.DataFrame
        DataFrame with appropriately rounded numeric values
    """
    df2 = df.copy()
    num_cols = df2.select_dtypes("number").columns.difference(INT_COLS)
    df2[num_cols] = df2[num_cols].astype(int)  # TODO: Change to round(1) for one decimal
    df2[list(INT_COLS)] = df2[list(INT_COLS)].astype(int)
    return df2


def position_tint(row):
    """
    Apply position-based color tinting to a row.

    This function determines the background color for a row
    based on the player's position.

    Parameters
    ----------
    row : pandas.Series
        A row from the player DataFrame

    Returns
    -------
    list
        List of CSS style strings for each cell in the row
    """
    color = POS_COLORS.get(row["position"], "")
    return [f"background-color: {color}" for _ in row]


def create_rankings_styler(df):
    """
    Create a styled DataFrame for the rankings display.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with player values and rankings

    Returns
    -------
    pandas.io.formats.style.Styler
        Styled DataFrame with color gradients and formatting
    """
    # Define columns for different color gradient styles
    cols_coolwarm = ["rank", "rank_pos", "rank_pos_team"]
    cols_rdylgn = (
            ["draft_value", "static_value", "dynamic_value", "mkt_share", "available_pts"]
            + [col for col in df.columns if col.startswith('value_')]
            + [col for col in df.columns if col.endswith('_projection')]
    )

    # Apply styling
    return (
        round_numeric(df)
        .style
        .apply(position_tint, axis=1)
        .background_gradient(subset=cols_rdylgn, cmap="RdYlGn")
        .background_gradient(subset=cols_coolwarm, cmap="coolwarm")
        .format(precision=1)
    )


def render_rankings():
    """
    Render the live player rankings table.

    This function runs the valuation model and displays the
    results in a styled table.

    Returns
    -------
    None
        Displays the rankings table in the Streamlit app
    """
    # Run model or use cached results
    result_df = run_model(st.session_state["base_df"])

    # Add button to manually clear cache and recalculate
    if st.button("Re-compute rankings ↻", type="primary"):
        run_model.clear()
        result_df = run_model(st.session_state["base_df"])

    # Create styler and display rankings
    styler = create_rankings_styler(result_df)
    st.dataframe(styler, use_container_width=True, height=750)


def render_position_top_picks():
    """
    Render a table showing the top 3 available players at each position.

    This function extracts the top 3 players by position from the
    rankings and displays them in a styled table.

    Returns
    -------
    None
        Displays the top picks table in the Streamlit app
    """
    # Get the latest rankings
    result_df = run_model(st.session_state["base_df"])

    # Extract top 3 players by position
    top3 = result_df.groupby("position").head(3)[["position", "draft_value", "rank", "rank_pos"]]
    top3.sort_values(["position", "draft_value"], ascending=[True, False], inplace=True)
    top3[["draft_value", "rank", "rank_pos"]] = top3[["draft_value", "rank", "rank_pos"]].astype(int)

    # Display with styling
    st.dataframe(
        top3
        .style
        .apply(position_tint, axis=1)
        .background_gradient(subset=["draft_value"], cmap="RdYlGn")
        .format(precision=1),
        use_container_width=True,
        height=500,
    )


def reset_draft():
    """
    Reset the draft state completely.

    This function clears the session state and triggers a
    rerun of the app, effectively resetting the draft.

    Returns
    -------
    None
        Clears the session state and reruns the app
    """
    st.session_state.clear()
    st.rerun()