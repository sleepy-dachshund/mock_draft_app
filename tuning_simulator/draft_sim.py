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

''' =================================================================================================================
=====================================================================================================================
        EXECUTION
=====================================================================================================================
================================================================================================================= '''

@dataclass
class DraftFig:
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]  # Gets from 'tuning_simulator' to 'fantasy_football'
    CACHE_DIR: Path = PROJECT_ROOT / "data" / "draft_sims"

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

    # DRAFT CONFIG OBJECT
    cfg = DraftFig()
    cfg.other_teams_picks_dict = {i: get_my_picks_list(cfg.n_rounds, cfg.n_teams, i) for i in range(1, cfg.n_teams + 1) if i != cfg.draft_pos}

    # PARAMETER SET TO RUN SIMULATIONS ON
    param_array = static_value_weights_generator()
    param_array = add_vopn_param(param_array, param_list=[2, 5])
    param_array = add_dynamic_multiplier_param(param_array, param_list=[0.0, 0.3])
    param_array = add_team_need_param(param_array, param_list=[0.05, 0.5])
    df_params = param_array_to_df(param_array)

    if TEST:
        df_params_test = df_params.sample(5).reset_index(drop=True)
        cfg.df_params = df_params_test.copy()  # Use a smaller sample for testing
        N_SIMULATIONS_PER_SET = 20
    else:
        cfg.df_params = df_params.copy()  # Make a copy to avoid modifying the original DataFrame
        N_SIMULATIONS_PER_SET = 25

    # RAW DATA -- PRE-VALUATION
    raw_df = get_raw_df().reset_index(drop=True)

    # from gen_values import value_players

    # --- Execution ---
    print(f"Starting simulations for {len(cfg.df_params)} parameter sets.")
    print(f"Each parameter set will be simulated {N_SIMULATIONS_PER_SET} times.")
    print(f'Total of {len(cfg.df_params) * N_SIMULATIONS_PER_SET} simulations will be run.')

    # Run all simulations for all parameter sets
    results_df = run_all_simulations(
        df_params=cfg.df_params,
        base_df_players=raw_df.copy(),
        draft_cfg=cfg,
        n_sims=N_SIMULATIONS_PER_SET
    ).sort_values(by='my_starters_projection', ascending=False).reset_index(drop=True)

    # --- Output ---
    print("\nSimulation complete.")

    param_grading = results_df.groupby('param_set_id').agg({
        'my_starters_projection': ['mean', 'std', 'min', 'max'],
        'my_starters_static_value': 'mean',
        'rank_proj': 'mean',
        'rank_static': 'mean',
        'elite': 'mean',
        'last_starter': 'mean',
        'replacement': 'mean',
        'vopn': 'mean',
        'dynamic_multiplier': 'mean',
        'team_needs': 'mean',
    })
    # collapse multi-index columns
    param_grading.columns = ['_'.join(col) for col in param_grading.columns.to_flat_index()]
    param_grading['Sharpe'] = param_grading['my_starters_projection_mean'] / param_grading['my_starters_projection_std']
    param_grading = param_grading.sort_values(by='my_starters_projection_mean', ascending=False).reset_index(drop=True)


    best_param_set = param_grading.iloc[0].param_set_id
    best_params = df_params[df_params.index == best_param_set].iloc[0]
    print(f"\nBest param set: {best_param_set}")
    print(f"Best params: {best_params.to_dict()}")
    print(f"Best projection: {param_grading.iloc[0].my_starters_projection:.2f}")

    # save results and param_grading to CSV files
    results_df.to_csv(cfg.CACHE_DIR / "draft_simulation_results.csv", index=False)
    param_grading.to_csv(cfg.CACHE_DIR / "draft_simulation_param_grading.csv", index=False)
    results_df.to_parquet(cfg.CACHE_DIR / "draft_simulation_results.parquet", index=False)
    param_grading.to_parquet(cfg.CACHE_DIR / "draft_simulation_param_grading.parquet", index=False)
    print("\nFull results and param grading saved to cache.")


    # Optional: Save results to a file for later analysis
    # results_output_path = cfg.CACHE_DIR / "draft_simulation_results.csv"
    # results_df.to_csv(results_output_path, index=False)
    # print(f"\nFull results saved to: {results_output_path}")
