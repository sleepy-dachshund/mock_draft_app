import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Set

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
    # TODO: add keepers here if needed

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

def _get_team_for_pick(
    pick: int,
    other_teams_picks_dict: Dict[int, List[int]]
) -> int:
    """
    Finds the team number corresponding to a given draft pick.

    Args:
        pick (int): The current overall pick number.
        other_teams_picks_dict (Dict[int, List[int]]):
            A dictionary mapping team numbers to their list of picks.

    Returns:
        int: The team number that owns the pick.
    """
    for team_num, picks_list in other_teams_picks_dict.items():
        if pick in picks_list:
            return team_num
    return -1 # Should not happen in a valid draft

def _get_roster_needs(
    team_df: pd.DataFrame,
    draft_cfg: "DraftFig"
) -> List[str]:
    """
    Determines a team's remaining open starting roster spots.

    Args:
        team_starters_df (pd.DataFrame):
            A DataFrame of players in the team's starting lineup.
        draft_cfg (DraftFig): The draft configuration object.

    Returns:
        Set[str]: A set of position strings (e.g., {'QB', 'RB'}) that
                  are not yet filled in the starting roster.
    """
    needs = set()
    position_counts = team_df['position'].value_counts().to_dict()

    if position_counts.get('QB', 0) < draft_cfg.n_qb:
        needs.add('QB')
    if position_counts.get('RB', 0) < draft_cfg.n_rb:
        needs.add('RB')
    if position_counts.get('WR', 0) < draft_cfg.n_wr:
        needs.add('WR')
    if position_counts.get('TE', 0) < draft_cfg.n_te:
        needs.add('TE')

    # Note: FLEX is handled differently, as it's not a primary position.
    # We prioritize filling dedicated spots first.

    return list(needs)


def _select_opponent_player(
        available_players_df: pd.DataFrame,
        pick: int,
        roster_needs: List[str]
) -> int:
    """
    Selects a player for an opponent based on team needs and ADP.

    Args:
        available_players_df (pd.DataFrame):
            DataFrame of players not yet drafted. Must include 'adp',
            'stdev', and 'position' columns.
        pick (int): The current overall pick number.
        roster_needs (Set[str]): A set of positions the team needs to fill.

    Returns:
        int: The index of the player to be drafted.
    """
    target_df = available_players_df.copy()

    # 1. Prioritize filling roster needs
    if len(roster_needs) > 0:
        needed_players_df = target_df[target_df['position'].isin(roster_needs)]
        if not needed_players_df.empty:
            target_df = needed_players_df

    # 2. Calculate pick likelihood based on a normal distribution around ADP
    # We use the cumulative density function (CDF).
    # A player is most likely to be picked around their ADP. Probability stays high if they fall past ADP.
    mean_adp = target_df['adp']
    std_dev = target_df['stdev'].fillna(10).replace(0, 10)
    earliest_pick = target_df['high']
    latest_pick = target_df['low']

    # Calculate the CDF of the normal distribution
    from scipy.stats import norm
    z_score = (pick - mean_adp) / std_dev  # z-score = how far are we from this players sweet spot? (earlier is negative, later is positive)
    likelihood = norm.cdf(z_score)
    # if pick << adp, z_score << 0, probability = 0
    # if pick ~= adp, z_score ~= 0, probability = 0.5
    # if pick >> adp, z_score >> 0, probability = 1

    # likelihood should be guided such that we stay within realistic pick ranges (high and low)
    likelihood = np.where(pick < earliest_pick, 0, likelihood)  # No chance of picking before earliest ADP
    likelihood = np.where(pick > latest_pick, 20, likelihood)  # If pick is after the latest ADP, boost odds significantly

    target_df['pick_likelihood'] = likelihood

    # 3. Make a weighted random selection
    # Players with a higher likelihood score have a higher chance of being picked.
    # If all likelihoods are zero (e.g., late in the draft), fall back to best ADP.
    if target_df['pick_likelihood'].sum() > 0:
        player_drafted_idx = target_df.sample(1, weights='pick_likelihood').index[0]
    else:
        player_drafted_idx = target_df.sort_values(by='adp').index[0]

    return player_drafted_idx

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
    assert len(available_players) > 0, "No available players to draft."
    is_my_pick = pick in draft_cfg.my_picks

    if is_my_pick:
        team_df = live_draft_df[live_draft_df['drafted'].isin(draft_cfg.my_picks)].copy()
        if team_df.empty:
            team_roster_needs = ['QB', 'RB', 'WR', 'TE']  # No players drafted yet, fill all needs
        else:
            team_roster_needs = _get_roster_needs(team_df, draft_cfg)

        # if team roster needs empty (drafted all starters), fill with best available
        team_roster_needs = ['QB', 'RB', 'WR', 'TE'] if len(team_roster_needs) == 0 else team_roster_needs
        my_realistic_picks = live_draft_df.loc[(available_players) & (live_draft_df.high < pick+10) & (live_draft_df.position.isin(team_roster_needs)), :].copy()
        player_drafted_idx = my_realistic_picks.loc[:, 'draft_value'].nlargest(1).index[0]
    else:
        # Opponent teams draft based on a simplified ADP-like strategy
        # TODO: add adp logic: 'adp', 'stdev', 'high', 'low'
        # top_n = 15
        # if pick <= 10:
        #     top_n = 3
        # elif pick <= 25:
        #     top_n = 7
        # elif pick <= 40:
        #     top_n = 12
        # player_drafted_idx = live_draft_df.loc[available_players, 'draft_value'].nlargest(top_n).sample(1).index[0]
        # player_drafted_id = live_draft_df.loc[player_drafted_idx, 'id']

        team_num = _get_team_for_pick(pick, draft_cfg.other_teams_picks_dict)
        teams_picks = draft_cfg.other_teams_picks_dict[team_num]
        team_df = live_draft_df[live_draft_df['drafted'].isin(teams_picks)].copy()
        if team_df.empty:
            team_roster_needs = ['QB', 'RB', 'WR', 'TE']  # No players drafted yet, fill all needs
        else:
            team_roster_needs = _get_roster_needs(team_df, draft_cfg)
        player_drafted_idx = _select_opponent_player(
            available_players_df=live_draft_df.loc[available_players, :],
            pick=pick,
            roster_needs=team_roster_needs
        )

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

    # when ADP > drafted, player was drafted ahead of their ADP, so lucky_factor is negative (reached)
    my_team_df['lucky_factor'] = (my_team_df['drafted'] - my_team_df['adp']) / my_team_df['stdev']

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
    min_proj_team = league_results_df['projection_sum'].min()
    average_proj_team = league_results_df['projection_sum'].mean()

    # Compile the final results dictionary
    final_results = {
        'param_set_id': param_set.name,
        'sim_num': sim_count,
        'total_proj': my_starters_projection,
        'total_value': my_starters_static_value,
        'luck_mean': my_starters.lucky_factor.mean(),
        'luck_std': my_starters.lucky_factor.std(),
        'luck_max': my_starters.lucky_factor.max(),
        'luck_min': my_starters.lucky_factor.min(),
        'rank_proj': rank_proj,
        'rank_static': rank_static,
        'worst_proj': min_proj_team,
        'avg_proj': average_proj_team,
        'my_top_8_picks': my_team_df.loc[my_team_df['drafted'] != 0].sort_values('drafted').head(8)['player'].tolist(),
        'qb': my_starters[my_starters['position'] == 'QB']['player'].tolist(),
        'rb': my_starters[my_starters['position'] == 'RB']['player'].tolist(),
        'wr': my_starters[my_starters['position'] == 'WR']['player'].tolist(),
        'te': my_starters[my_starters['position'] == 'TE']['player'].tolist(),
        'luckiest_picks': my_team_df.nlargest(3, 'lucky_factor')['player'].tolist(),
        **param_set.to_dict(),
        'first_10_picks': completed_draft_df.loc[completed_draft_df['drafted'] != 0].sort_values('drafted').head(10)['player'].tolist(),
    }
    final_results.update(param_set.to_dict())

    return final_results

def run_all_simulations(
        df_params: pd.DataFrame,
        base_df_players: pd.DataFrame,
        draft_cfg: "DraftFig",
        n_sims: int,
        df_adp: pd.DataFrame
) -> pd.DataFrame:
    """
    Runs draft simulations for each parameter set and collects the results.

    Args:
        df_params (pd.DataFrame): Dataframe where each row is a set of parameters to test.
        base_df_players (pd.DataFrame): The base dataframe of players and their projections.
        draft_cfg (DraftFig): The draft configuration object.
        n_sims (int): The number of simulations to run for each parameter set.
        df_adp (pd.DataFrame): Dataframe containing ADP data for players.

    Returns:
        pd.DataFrame: A dataframe containing the results of all simulations.
    """
    base_df_players = base_df_players.merge(df_adp.drop(columns=['player', 'team', 'position']), how='left', on='id', validate='1:1')

    all_results = []
    for param_set_id, param_set in df_params.iterrows():
        print(f"Processing Param Set {param_set_id} ({df_params.index.get_loc(param_set_id) + 1}/{len(df_params)})...")
        for i in range(1, n_sims + 1):
            input_df = base_df_players.copy()
            final_draft_df, input_draft_df = simulate_one_draft(param_set, input_df, draft_cfg)
            result = evaluate_draft(final_draft_df, param_set, i, draft_cfg)
            all_results.append(result)

    return pd.DataFrame(all_results)