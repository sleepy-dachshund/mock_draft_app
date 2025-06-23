import os
import traceback
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Optional

import pandas as pd
from pandas.tseries.offsets import BDay
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import config


def static_value_weights_generator() -> np.ndarray:
    step = 0.16
    grid = np.arange(0, 1 + step, step)

    # triplets A, B, C with A+B+C == 1.0
    weights = (
        (a, b, 1.0 - a - b)  # C is implied
        for a, b in product(grid, repeat=2)
        if a + b <= 1.0  # keeps C ≥ 0
    )

    weights_array = []
    for a, b, c in weights:
        weights_array.append((a, b, c))

    weights_array = np.array(weights_array, dtype=float)

    return weights_array

def add_vopn_param(weights_array, param_list: list = None):
    """
    Adds a 'vopn' parameter to the weights array.
    Duplicate array n times and add a new column range [0, n].
    """
    if param_list is None:
        vopn = np.array([0, 2, 5, 9])
    else:
        vopn = np.array(param_list)
    # merge the vopn with the static weights
    new_array = np.array(
        [np.concatenate((weights, [v])) for weights, v in product(weights_array, vopn)]
    )
    return new_array

def add_dynamic_multiplier_param(param_array, param_list: list = None):
    """
    Adds a 'dynamic_multiplier' parameter to the weights array.
    Duplicate array n times and add a new column range [0, n].
    """
    if param_list is None:
        dynamic_multiplier = np.array([0.0, 0.1, 0.2, 0.5])
    else:
        dynamic_multiplier = np.array(param_list)
    new_array = np.array(
        [np.concatenate((weights, [v])) for weights, v in product(param_array, dynamic_multiplier)]
    )
    return new_array

def add_team_need_param(param_array: np.ndarray, param_list: list = None) -> np.ndarray:
    if param_list is None:
        team_needs = np.array([0, 0.25, 0.5, 0.75, 1])
    else:
        team_needs = np.array(param_list)
    new_array = np.array(
        [np.concatenate((weights, [v])) for weights, v in product(param_array, team_needs)]
    )
    return new_array

def param_array_to_df(param_array: np.ndarray) -> pd.DataFrame:
    """
    Converts the parameter array to a DataFrame with appropriate column names.
    """
    columns = ['elite', 'last_starter', 'replacement', 'vopn', 'dynamic_multiplier', 'team_needs']
    df = pd.DataFrame(param_array, columns=columns, dtype=float)
    return df


param_array = static_value_weights_generator()
param_array = add_vopn_param(param_array, param_list=[0, 2, 5, 9])
param_array = add_dynamic_multiplier_param(param_array, param_list=[0.0, 0.25, 0.5])
param_array = add_team_need_param(param_array, param_list=[0, 0.5, 0.75, 1])
df_params = param_array_to_df(param_array)

def get_my_picks_list(rounds, teams, draft_pos) -> list:
    my_picks = []
    for round_num in range(1, rounds + 1):
        if round_num % 2 == 1:  # Odd rounds go 1 to N
            pick_num = (round_num - 1) * teams + draft_pos
        else:  # Even rounds go N to 1 (snake draft)
            pick_num = round_num * teams - draft_pos + 1
        my_picks.append(pick_num)
    return my_picks

@dataclass
class DraftFig:
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]  # Gets from 'tuning_simulator' to 'fantasy_football'
    CACHE_DIR: Path = PROJECT_ROOT / "data" / "draft_sims"

    df_params = df_params  # columns = ['elite', 'last_starter', 'replacement', 'vopn', 'dynamic_multiplier', 'team_needs']

    n_qb: int = config.ROSTER_N_QB
    n_rb: int = config.ROSTER_N_RB
    n_wr: int = config.ROSTER_N_WR
    n_te: int = config.ROSTER_N_TE
    n_flex: int = config.ROSTER_N_FLEX
    flex_pos = config.ROSTER_FLEX_ELIGIBLE_POSITIONS
    n_k: int = config.ROSTER_N_K
    n_dst: int = config.ROSTER_N_DST
    n_bench: int = config.ROSTER_N_BENCH

    n_teams: int = config.N_TEAMS
    n_picks: int = config.N_TEAMS * (n_qb + n_rb + n_wr + n_te + n_flex + n_k + n_dst + n_bench)
    n_rounds = n_picks // n_teams

    draft_type: str = 'snake'
    draft_pos: int = config.DRAFT_POSITION

    my_picks = get_my_picks_list(n_rounds, n_teams, draft_pos)


cfg = DraftFig()

from gen_values import value_players, get_raw_df

raw_df = get_raw_df()

'''
example sim run
'''
# todo: find a way to include the 'team_needs' parameter in value_players: the purpose is to reduce the draft_value of positions that are not needed (already filled starter slots for those positions)
input_df = raw_df.copy()
param_set = 0  # index of the parameter set, will loop through range(len(cfg.df_params))
result_df = value_players(input_df,
                          static_value_weights={'elite': cfg.df_params['elite'].values[param_set],
                                                'last_starter': cfg.df_params['last_starter'].values[param_set],
                                                'replacement': cfg.df_params['replacement'].values[param_set]},
                          projection_column_prefix=config.PROJECTION_COLUMN_PREFIX,
                          vopn=int(cfg.df_params['vopn'].values[param_set]),
                          dynamic_multiplier=cfg.df_params['dynamic_multiplier'].values[param_set],
                          draft_mode=True)

'''
print(result_df.columns)
Index(['id', 'player', 'team', 'position',  # player identifying columns 
        'draft_value', 'static_value', 'dynamic_value',  # we want to test out our param set from df_params to figure out how to make the best draft_value (which is a combination of static and dynamic values) 
       'drafted',  # this will update with each pick, the player picked will have a 'drafted' value update from 0 to 1 
       'median_projection',  # we'll use the sum of this (by starters only) to evaluate our draft 
       
       # other columns not necessary for param tuning, draft sim, or draft grading
       'rank', 'rank_pos', 'rank_pos_team', 'mkt_share', 
       'available_pts', 'high_projection', 'low_projection', 
       'value_elite', 'value_last_starter', 'value_replacement'],
      dtype='object')
'''

