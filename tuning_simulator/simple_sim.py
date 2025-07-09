import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import config
from gen_values import value_players, get_raw_df
from sim_utils import (static_value_weights_generator, add_vopn_param, add_dynamic_multiplier_param,
                       add_team_need_param, param_array_to_df, get_my_picks_list)
from sim_funcs import (simulate_one_draft, make_a_pick, check_filled_starters,
                       evaluate_draft, run_all_simulations)

@dataclass
class DraftFig:
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]  # Gets from 'tuning_simulator' to 'fantasy_football'
    CACHE_DIR: Path = PROJECT_ROOT / "data" / "draft_sims"
    YEAR: int = config.SEASON_YEAR  # e.g., 2025

    proj_col_prefix: str = config.PROJECTION_COLUMN_PREFIX  # e.g., 'proj_'

    # draft sim parameter dataframe (each row is a param set for a draft simulation)
    df_params = None  # columns = ['elite', 'last_starter', 'replacement', 'vopn', 'dynamic_multiplier', 'team_needs']

    # starting roster settings
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
    other_teams_picks_dict = None


if __name__ == "__main__":

    TEST = True

    # --- Configuration ---

    '''
        RAW DATA -- PRE-VALUATION
    '''
    raw_df = get_raw_df().reset_index(drop=True)

    '''
        DRAFT CONFIG OBJECT
    '''
    cfg = DraftFig()
    cfg.other_teams_picks_dict = {i: get_my_picks_list(cfg.n_rounds, cfg.n_teams, i) for i in range(1, cfg.n_teams + 1) if i != cfg.draft_pos}

    '''
        LOAD ADP DATA
    '''
    try:
        import datetime
        today = datetime.date.today().strftime("%Y%m%d")
        output_path = config.OUTPUT_DIR / f"{today}_adp.csv"
        df_adp = pd.read_csv(output_path)
    except FileNotFoundError:
        from pull_adp import pull_adp, clean_adp_df, save_adp_data
        df_adp = clean_adp_df(pull_adp(year=cfg.YEAR, teams=cfg.n_teams, position='all'))
        save_adp_data(df_adp)

    input_df = raw_df.merge(df_adp.drop(columns=['player', 'team', 'position']), how='left', on='id', validate='1:1')

    filled_starter_positions = []  # will populate with e.g. 'QB', 'RB', 'WR', 'TE' as these positions are drafted (by me)
    df = value_players(
            df=input_df.copy(),
            static_value_weights={'elite': 0.10,
                                  'last_starter': 0.75,
                                  'replacement': 0.15},
            vopn=5,
            projection_column_prefix=cfg.proj_col_prefix,
            dynamic_multiplier=0.05,
            filled_roster_spots=filled_starter_positions,
            team_needs=1.0,
            draft_mode=True)

    ''' ==================================================================
        START SIMULATIONS HERE
    ================================================================== '''
    N_SIMS = 10 if TEST else 10_000

