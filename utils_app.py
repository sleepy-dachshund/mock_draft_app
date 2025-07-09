"""
Fantasy Football Draft Assistant
================================

This Streamlit app provides a real-time fantasy football draft assistant that:
1. Displays an interactive draft board for marking players as drafted
2. Calculates and updates player values based on draft progress
3. Shows live rankings with color-coded information for easy decision making
4. Provides top remaining players by position
5. Allows resetting the draft state

The app calculates player value using both static projections and dynamic
value calculations that account for positional scarcity as the draft progresses.
"""

import streamlit as st
import pandas as pd
import config
from draft_analyzer import DraftAnalyzer
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
    "devon achane": 105,
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
    st.info("Mark players as drafted by changing the 'drafted' column value.")
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
            vopn=3,
            dynamic_multiplier=0.05,
            draft_mode=True
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

    # Create styler and display rankings
    styler = create_rankings_styler(result_df)
    st.dataframe(styler, use_container_width=True, height=750)

def render_position_top_picks(n=5):
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
    top_n = result_df.groupby("position").head(n)[["position", "draft_value", "static_value", "rank", "rank_pos"]]
    top_n.sort_values(["position", "draft_value"], ascending=[True, False], inplace=True)
    top_n[["draft_value", "static_value", "rank", "rank_pos"]] = top_n[["draft_value", "static_value", "rank", "rank_pos"]].astype(int)

    # Display with styling
    st.dataframe(
        top_n
        .style
        .apply(position_tint, axis=1)
        .background_gradient(subset=["draft_value", "static_value"], cmap="RdYlGn")
        .format(precision=1),
        use_container_width=True,
        height=500,
    )

def write_current_draft_pick():
    """
    Write the current draft pick number to the app.

    This function displays the current draft pick number in the app.

    Returns
    -------
    None
        Displays the current draft pick number in the Streamlit app
    """
    draft_picks = [
        pick
        for pick in st.session_state["base_df"]["drafted"].sort_values().tolist()
        if pick > 0 and pick not in KEEPERS.values()  # ← need () after values
    ]

    current_draft_pick = draft_picks[-1] + 1 if draft_picks else 1

    pick_in_round = (current_draft_pick - 1) % config.N_TEAMS + 1
    round_num = (current_draft_pick - 1) // config.N_TEAMS + 1

    st.write(f"## Current Draft Pick: Round {round_num}, Pick {pick_in_round} ({current_draft_pick} Overall)")

    return round_num

def write_draft_notes(round_num):

    # hard-coded tips from config:
    st.write(f"Tip: {config.draft_tips[round_num]}")

    # # draft analyzer:
    # result_df = run_model(st.session_state["base_df"])
    # lineup = DraftAnalyzer(result_df)
    # roster_df = lineup.roster_table()
    # pos_totals = lineup.pos_sums()
    # pct_max, pct_league = lineup.strength_scores()



def re_calc_rankings():
    """
    Recalculate the rankings based on the current draft state.

    This function clears the cache and runs the valuation model
    to update the rankings.

    Returns
    -------
    pandas.DataFrame
        DataFrame with updated player values and rankings
    """
    run_model.clear()
    return run_model(st.session_state["base_df"])


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

def render_draft_history():
    """
    Render a table showing all drafted players in pick order.

    This function creates a view of all drafted players sorted by
    draft pick number, allowing users to see the draft history.

    Returns
    -------
    None
        Displays the draft history table in the Streamlit app
    """
    # Get the latest rankings
    result_df = run_model(st.session_state["base_df"])

    # Reset index to get player as a column
    result_df_with_player = result_df.reset_index()

    # Prepare draft history dataframe
    # Start with all players who have been drafted (drafted > 0)
    drafted_df = result_df_with_player[result_df_with_player['drafted'] > 0].copy()

    # Select and arrange columns
    draft_history = drafted_df[['drafted', 'player', 'position', 'rank', 'static_value']]

    # Add median_projection column if available
    projection_cols = [col for col in result_df_with_player.columns if col.endswith('_projection')]
    if projection_cols:
        # Assuming the first projection column is the median
        draft_history['median_projection'] = drafted_df[projection_cols[0]]

    # Handle duplicate pick numbers (if any)
    if draft_history['drafted'].duplicated().any():
        # Identify duplicates
        duplicates = draft_history[draft_history['drafted'].duplicated(keep=False)]
        st.warning(f"Warning: Multiple players assigned to the same pick number: {duplicates['drafted'].unique().tolist()}")

        # Keep the first player for each pick number
        draft_history = draft_history.drop_duplicates(subset=['drafted'], keep='first')

    # Sort by draft pick
    draft_history.sort_values('drafted', inplace=True)

    # Calculate the total number of draft picks
    required_positions = {
        'QB': config.ROSTER_N_QB,
        'RB': config.ROSTER_N_RB,
        'WR': config.ROSTER_N_WR,
        'TE': config.ROSTER_N_TE,
        'FLEX': config.ROSTER_N_FLEX,
        'K': getattr(config, 'ROSTER_N_K', 0),
        'DST': getattr(config, 'ROSTER_N_DST', 0),
        'BN': config.ROSTER_N_BENCH
    }
    total_picks = sum(required_positions.values()) * config.N_TEAMS

    # Create a complete pick list with empty rows for unfilled picks
    all_picks = pd.DataFrame({'drafted': range(1, total_picks + 1)})
    complete_history = pd.merge(all_picks, draft_history, on='drafted', how='left')

    # Apply styling
    styled_history = (
        complete_history
        .style
        .apply(lambda row: [f"background-color: {POS_COLORS.get(row['position'], '')}"
                           if not pd.isna(row['position']) else "" for _ in row], axis=1)
        .format(precision=1)
    )

    st.dataframe(styled_history, use_container_width=True, height=400)

def calculate_my_draft_picks():
    """
    Calculate my draft picks based on draft position and number of teams.

    This function determines which picks belong to my team based on
    the draft position and total number of teams in a snake draft.

    Returns
    -------
    list
        List of pick numbers that belong to my team
    """
    # Get draft position and number of teams
    draft_pos = config.DRAFT_POSITION
    n_teams = config.N_TEAMS

    # Calculate the required number of players per team
    required_positions = {
        'QB': config.ROSTER_N_QB,
        'RB': config.ROSTER_N_RB,
        'WR': config.ROSTER_N_WR,
        'TE': config.ROSTER_N_TE,
        'FLEX': config.ROSTER_N_FLEX,
        'K': getattr(config, 'ROSTER_N_K', 0),
        'DST': getattr(config, 'ROSTER_N_DST', 0),
        'BN': config.ROSTER_N_BENCH
    }
    players_per_team = sum(required_positions.values())

    my_picks = []
    for round_num in range(1, players_per_team + 1):
        if round_num % 2 == 1:  # Odd rounds go 1 to N
            pick_num = (round_num - 1) * n_teams + draft_pos
        else:  # Even rounds go N to 1 (snake draft)
            pick_num = round_num * n_teams - draft_pos + 1
        my_picks.append(pick_num)

    return my_picks

def render_my_roster():
    """
    Render a table showing my current roster.

    This function identifies my drafted players, organizes them into a lineup, and displays the result.

    Returns
    -------
    None
        Displays the my roster table in the Streamlit app
    """

    # draft analyzer:
    result_df = value_players(st.session_state["base_df"], draft_mode=True)
    lineup = DraftAnalyzer(result_df)
    my_roster = lineup.roster_table()

    # Apply styling
    styled_roster = (
        my_roster
        .style
        .apply(lambda row: [f"background-color: {POS_COLORS.get(row['position'], '')}" for _ in row], axis=1)
        .format(precision=1)
    )

    st.dataframe(styled_roster, use_container_width=True, height=400)

def render_placeholder_table():
    """
    Render a placeholder table for future expansion.

    This function creates a simple placeholder that can be
    replaced with actual content in the future.

    Returns
    -------
    None
        Displays a placeholder message in the Streamlit app
    """
    st.info("This space is reserved for future features. Some ideas include position balance analysis, team needs, or league-wide position scarcity tracking.")

    # draft analyzer:
    result_df = value_players(st.session_state["base_df"], draft_mode=True)
    lineup = DraftAnalyzer(result_df)
    roster_df = lineup.roster_table()
    pos_totals = lineup.pos_sums()
    pct_max, pct_league = lineup.strength_scores()

    # positional group value dataframe -- turn pos_totals dict, pct_league dict, and pct_max dict into a dataframe
    pos_totals_df = pd.DataFrame.from_dict(pos_totals, orient='index', columns=['points'])
    pos_totals_df['pct_league'] = pct_league.values()
    pos_totals_df['pct_max'] = pct_max.values()
    pos_totals_df['position'] = pos_totals_df.index


    st.dataframe(pos_totals_df, use_container_width=True, height=400)


def render_full_draft_board():
    """
    Render a complete draft board showing all teams and rounds.

    This function creates a visual representation of the entire draft,
    with teams as columns and rounds as rows, showing which player
    was taken with each pick.

    Returns
    -------
    None
        Displays the full draft board in the Streamlit app
    """
    # Get the latest rankings
    result_df = run_model(st.session_state["base_df"])
    result_df_with_player = result_df.reset_index()

    # Get drafted players
    drafted_players = result_df_with_player[result_df_with_player['drafted'] > 0].copy()

    # Calculate total rounds based on roster settings
    required_positions = {
        'QB': config.ROSTER_N_QB,
        'RB': config.ROSTER_N_RB,
        'WR': config.ROSTER_N_WR,
        'TE': config.ROSTER_N_TE,
        'FLEX': config.ROSTER_N_FLEX,
        'K': getattr(config, 'ROSTER_N_K', 0),
        'DST': getattr(config, 'ROSTER_N_DST', 0),
        'BN': config.ROSTER_N_BENCH
    }
    total_rounds = sum(required_positions.values())
    n_teams = config.N_TEAMS

    # Create empty draft board
    # Initialize with empty strings
    draft_board = pd.DataFrame(
        index=range(1, total_rounds + 1),
        columns=range(1, n_teams + 1),
        data=""
    )

    # Rename index and columns for clarity
    draft_board.index.name = "Round"
    draft_board.columns = [f"Team {i}" for i in range(1, n_teams + 1)]

    # Function to convert pick number to (round, team) coordinates
    def pick_to_coordinates(pick_num, n_teams):
        round_num = (pick_num - 1) // n_teams + 1
        pick_in_round = (pick_num - 1) % n_teams

        if round_num % 2 == 1:  # Odd rounds go left to right
            team_num = pick_in_round + 1
        else:  # Even rounds go right to left (snake)
            team_num = n_teams - pick_in_round

        return round_num, team_num

    # Fill draft board with drafted players
    for _, row in drafted_players.iterrows():
        pick_num = int(row['drafted'])
        if pick_num <= total_rounds * n_teams:  # Ensure pick is within board range
            round_num, team_num = pick_to_coordinates(pick_num, n_teams)

            # Create formatted cell content with player info
            player_text = (
                f"{row['player']} ({row['position']})\n"
                f"Rank: {int(row['rank'])}, Value: {int(row['static_value'])}"
            )

            # Add to draft board
            draft_board.at[round_num, f"Team {team_num}"] = player_text

    # Highlight the user's team column
    my_team_col = f"Team {config.DRAFT_POSITION}"

    # Create a color mapping based on positions in the data
    position_color_map = {}
    for idx, row in draft_board.iterrows():
        for col in draft_board.columns:
            cell_value = draft_board.at[idx, col]
            if cell_value and '(' in cell_value and ')' in cell_value:
                try:
                    position = cell_value.split('(')[1].split(')')[0]
                    position_color_map[(idx, col)] = POS_COLORS.get(position, '')
                except:
                    # Skip if parsing fails
                    pass

    # Apply styling in a safer way
    def style_draft_board(df):
        # Start with a basic style
        styler = df.style.set_properties(**{
            'white-space': 'pre-wrap',
            'text-align': 'left',
            'vertical-align': 'top',
            'padding': '10px',
            'border': '1px solid #ddd'
        })

        # Highlight my team's column if it exists
        if my_team_col in df.columns:
            styler = styler.set_properties(
                subset=pd.IndexSlice[:, my_team_col],
                **{'background-color': '#e6f3ff', 'font-weight': 'bold'}
            )

        # Add alternating row colors for readability
        styler = styler.set_table_styles([
            {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f5f5f5')]},
            {'selector': 'tr:nth-child(odd)', 'props': [('background-color', '#ffffff')]},
            {'selector': 'th', 'props': [('background-color', '#4CAF50'), ('color', 'white'), ('font-weight', 'bold')]}
        ])

        # Apply position-based colors to individual cells
        for (idx, col), color in position_color_map.items():
            if color:
                styler = styler.set_properties(
                    subset=pd.IndexSlice[idx, col],
                    **{'background-color': color}
                )

        return styler

    # Apply styling and display
    st.dataframe(
        style_draft_board(draft_board),
        use_container_width=True,
        height=max(400, total_rounds * 45)  # Dynamic height based on rounds
    )