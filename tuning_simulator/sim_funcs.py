import pandas as pd
from typing import Dict, List, Any, Tuple

import config
from gen_values import value_players, get_raw_df

def simulate_one_draft(
    param_set: pd.Series,
    df_players: pd.DataFrame,
    draft_cfg: "DraftFig"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates a single fantasy football draft for a given parameter set.

    Args:
        param_set (pd.Series): A row from df_params containing the parameters for this simulation.
        df_players (pd.DataFrame): The base dataframe of players and their projections.
        draft_cfg (DraftFig): The draft configuration object.

    Returns:
        pd.DataFrame: A dataframe with final draft results, where the 'drafted' column indicates the pick number for each player.
    """
    input_draft_df = df_players.copy()
    input_draft_df['drafted'] = 0
    filled_starter_positions = []

    for pick in range(1, draft_cfg.n_picks + 1):
        # Dynamically calculate player values before each pick
        live_draft_df = value_players(
            df=input_draft_df,
            static_value_weights={'elite': param_set['elite'],
                                  'last_starter': param_set['last_starter'],
                                  'replacement': param_set['replacement']},
            vopn=int(param_set['vopn']),
            projection_column_prefix=draft_cfg.proj_col_prefix,
            dynamic_multiplier=param_set['dynamic_multiplier'],
            filled_roster_spots=filled_starter_positions,
            team_needs=param_set['team_needs'],
            draft_mode=True
        )

        # player_drafted_idx, drafted_pos = make_a_pick(live_draft_df, pick, draft_cfg)
        player_drafted_id, drafted_pos = make_a_pick(live_draft_df, pick, draft_cfg)

        input_draft_df.loc[input_draft_df['id'] == player_drafted_id, 'drafted'] = pick
        live_draft_df.loc[live_draft_df['id'] == player_drafted_id, 'drafted'] = pick

        if pick in draft_cfg.my_picks:
            my_team_df = live_draft_df[live_draft_df['drafted'].isin(draft_cfg.my_picks)]
            filled_starter_positions = check_filled_starters(my_team_df, draft_cfg)

    return live_draft_df, input_draft_df

def make_a_pick(
    live_draft_df: pd.DataFrame,
    pick: int,
    draft_cfg: "DraftFig"
) -> (int, str):
    """
    Selects a player based on the current pick number.

    Args:
        live_draft_df (pd.DataFrame): The current state of the draft board.
        pick (int): The current overall pick number.
        draft_cfg (DraftFig): The draft configuration object.

    Returns:
        tuple[int, str]: The index and position of the drafted player.
    """
    available_players = live_draft_df['drafted'] == 0
    is_my_pick = pick in draft_cfg.my_picks

    if is_my_pick:
        player_drafted_idx = live_draft_df.loc[available_players, 'draft_value'].nlargest(1).index[0]
        player_drafted_id = live_draft_df.loc[player_drafted_idx, 'id']
    else:
        # Opponent teams draft based on a simplified ADP-like strategy
        top_n = 15
        if pick <= 10:
            top_n = 3
        elif pick <= 25:
            top_n = 7
        elif pick <= 40:
            top_n = 12
        player_drafted_idx = live_draft_df.loc[available_players, 'draft_value'].nlargest(top_n).sample(1).index[0]
        player_drafted_id = live_draft_df.loc[player_drafted_idx, 'id']

    drafted_position = live_draft_df.loc[live_draft_df['id'] == player_drafted_id, 'position']
    return player_drafted_id, drafted_position


def check_filled_starters(
        my_team_df: pd.DataFrame,
        draft_cfg: "DraftFig"
) -> List[str]:
    """
    Checks which starting positions have been filled on a team.

    Args:
        my_team_df (pd.DataFrame): Dataframe of players on my team.
        draft_cfg (DraftFig): The draft configuration object.

    Returns:
        List[str]: A list of positions where the starting slots are full.
    """
    roster_needs = {
        'QB': draft_cfg.n_qb,
        'RB': draft_cfg.n_rb,
        'WR': draft_cfg.n_wr,
        'TE': draft_cfg.n_te
    }
    position_counts = my_team_df['position'].value_counts().to_dict()

    filled_positions = [pos for pos, needed in roster_needs.items() if position_counts.get(pos, 0) >= needed]
    return filled_positions

import pandas as pd
from typing import Dict, Any

def _get_team_starters(
    team_df: pd.DataFrame,
    draft_cfg: "DraftFig"
) -> pd.DataFrame:
    """
    Identifies the optimal starting lineup for a given team's roster.

    Args:
        team_df (pd.DataFrame): DataFrame of players on a single team.
        draft_cfg (DraftFig): The draft configuration object, which contains
                              settings like roster spots (n_qb, n_rb, etc.)
                              and flex positions.

    Returns:
        pd.DataFrame: A DataFrame containing the starting players for the team.
    """
    # Identify starters for fixed positions
    starters = pd.concat([
        team_df[team_df['position'] == 'QB'].nlargest(draft_cfg.n_qb, 'median_projection'),
        team_df[team_df['position'] == 'RB'].nlargest(draft_cfg.n_rb, 'median_projection'),
        team_df[team_df['position'] == 'WR'].nlargest(draft_cfg.n_wr, 'median_projection'),
        team_df[team_df['position'] == 'TE'].nlargest(draft_cfg.n_te, 'median_projection')
    ])

    # Identify the best flex players from the remaining pool
    flex_candidates = team_df[
        ~team_df.index.isin(starters.index) &
        team_df['position'].isin(draft_cfg.flex_pos)
    ]
    flex_starters = flex_candidates.nlargest(draft_cfg.n_flex, 'median_projection')

    # Combine all starters into a single DataFrame
    all_starters = pd.concat([starters, flex_starters])
    return all_starters

def evaluate_draft(
    completed_draft_df: pd.DataFrame,
    param_set: pd.Series,
    sim_count: int,
    draft_cfg: "DraftFig"
) -> Dict[str, Any]:
    """
    Evaluates the results of a completed draft for a given team.

    Args:
        completed_draft_df (pd.DataFrame): The dataframe after the draft is finished.
        param_set (pd.Series): The parameter set used for the simulation.
        sim_count (int): The simulation number for this parameter set.
        draft_cfg (DraftFig): The draft configuration object.

    Returns:
        Dict[str, Any]: A dictionary summarizing the draft results.
    """
    my_team_df = completed_draft_df[completed_draft_df['drafted'].isin(draft_cfg.my_picks)].copy()
    my_starters = _get_team_starters(my_team_df, draft_cfg)

    my_starters_projection = my_starters['median_projection'].sum()
    my_starters_static_value = my_starters['static_value'].sum()

    league_results = {}
    all_team_numbers = list(range(1, draft_cfg.n_teams + 1))

    for team_num in all_team_numbers:
        if team_num == draft_cfg.draft_pos:
            team_df = my_team_df
        else:
            team_picks = draft_cfg.other_teams_picks_dict[team_num]
            team_df = completed_draft_df[completed_draft_df['drafted'].isin(team_picks)].copy()

        team_starters = _get_team_starters(team_df, draft_cfg)

        # Add each team's median_projection and static_value to league_results
        league_results[team_num] = {
            'projection_sum': team_starters['median_projection'].sum(),
            'static_value_sum': team_starters['static_value'].sum()
        }

    # Convert results to a DataFrame for easy ranking
    league_results_df = pd.DataFrame.from_dict(league_results, orient='index')

    # Find where my team ranks in terms of summed starter values
    league_results_df['rank_proj'] = league_results_df['projection_sum'].rank(method='min', ascending=False)
    league_results_df['rank_static'] = league_results_df['static_value_sum'].rank(method='min', ascending=False)

    # Extract my team's ranks and cast to int
    my_team_ranks = league_results_df.loc[draft_cfg.draft_pos]
    rank_proj = int(my_team_ranks['rank_proj'])
    rank_static = int(my_team_ranks['rank_static'])

    # Compile the final results dictionary
    final_results = {
        'param_set_id': param_set.name,
        'sim_num': sim_count,
        'my_starters_projection': my_starters_projection,
        'my_starters_static_value': my_starters_static_value,
        'rank_proj': rank_proj,
        'rank_static': rank_static,
        'top_5_picks': my_team_df.nsmallest(5, 'drafted')['player'].tolist(),
        **param_set.to_dict()
    }
    final_results.update(param_set.to_dict())

    return final_results

def run_all_simulations(
    df_params: pd.DataFrame,
    base_df_players: pd.DataFrame,
    draft_cfg: "DraftFig",
    n_sims: int
) -> pd.DataFrame:
    """
    Runs draft simulations for each parameter set and collects the results.

    Args:
        df_params (pd.DataFrame): Dataframe where each row is a set of parameters to test.
        base_df_players (pd.DataFrame): The base dataframe of players and their projections.
        draft_cfg (DraftFig): The draft configuration object.
        n_sims (int): The number of simulations to run for each parameter set.

    Returns:
        pd.DataFrame: A dataframe containing the results of all simulations.
    """

    all_results = []
    for param_set_id, param_set in df_params.iterrows():
        print(f"Processing Param Set {param_set_id} ({df_params.index.get_loc(param_set_id) + 1}/{len(df_params)})...")
        for i in range(1, n_sims + 1):
            input_df = base_df_players.copy()
            final_draft_df, input_draft_df = simulate_one_draft(param_set, input_df, draft_cfg)
            result = evaluate_draft(final_draft_df, param_set, i, draft_cfg)
            all_results.append(result)

    # results_df = pd.DataFrame(all_results)
    #
    # all_results = []
    # for param_set_id, param_set in df_params.iterrows():
    #     for i in range(1, n_sims + 1):
    #         print(f"Running Sim {i}/{n_sims} for Param Set {param_set_id}...")
    #         final_draft_df = simulate_one_draft(param_set, base_df_players, draft_cfg)
    #         result = evaluate_draft(final_draft_df, param_set, i, draft_cfg)
    #         all_results.append(result)

    return pd.DataFrame(all_results)