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
        PARAMETER SET TO RUN SIMULATIONS ON
    '''
    param_array = static_value_weights_generator(steps=3)
    param_array = add_vopn_param(param_array, param_list=[1, 2])
    param_array = add_dynamic_multiplier_param(param_array, param_list=[0.05, 0.25])
    param_array = add_team_need_param(param_array, param_list=[None])
    df_params = param_array_to_df(param_array)

    if TEST:
        df_params_test = df_params.sample(10).reset_index(drop=True)
        cfg.df_params = df_params_test.copy()  # Use a smaller sample for testing
        N_SIMULATIONS_PER_SET = 10
    else:
        cfg.df_params = df_params.copy()  # Make a copy to avoid modifying the original DataFrame
        N_SIMULATIONS_PER_SET = 30

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

    ''' ====================================================================================
        RUN ALL SIMULATIONS
    ==================================================================================== '''
    print(f"Starting simulations for {len(cfg.df_params)} parameter sets.")
    print(f"Each parameter set will be simulated {N_SIMULATIONS_PER_SET} times.")
    print(f'Total of {len(cfg.df_params) * N_SIMULATIONS_PER_SET} simulations will be run.')
    print(f"This may take about {len(cfg.df_params) * N_SIMULATIONS_PER_SET * 2.0 / 60:.2f} minutes.")

    results_df = run_all_simulations(
        df_params=cfg.df_params,
        base_df_players=raw_df.copy(),
        draft_cfg=cfg,
        n_sims=N_SIMULATIONS_PER_SET,
        df_adp=df_adp
    ).sort_values(by='total_proj', ascending=False).reset_index(drop=True)

    # --- Output ---
    print("\nSimulation complete.")

    '''
        SUMMARIZE RESULTS BY PARAM SET
    '''
    param_grading = results_df.groupby('param_set_id').agg({
        'total_proj': ['mean', 'std', 'min', 'max'],
        'total_value': 'mean',
        'luck_mean': 'mean',
        'rank_proj': 'mean',
        'rank_static': 'mean',
        'worst_proj': 'mean',
        'avg_proj': 'mean',
        'elite': 'mean',
        'last_starter': 'mean',
        'replacement': 'mean',
        'vopn': 'mean',
        'dynamic_multiplier': 'mean',
        'team_needs': 'mean',
    }).round(2)
    # collapse multi-index columns
    param_grading.columns = ['_'.join(col) for col in param_grading.columns.to_flat_index()]
    param_grading['sharpe'] = (param_grading['total_proj_mean'] / param_grading['total_proj_std']).round(2)
    param_grading = param_grading.sort_values(by='total_proj_mean', ascending=False)
    first_cols = ['total_proj_mean', 'sharpe', 'total_proj_std', 'total_proj_min', 'total_proj_max']
    param_grading = param_grading[first_cols + [col for col in param_grading.columns if col not in first_cols]]


    '''
        DESCRIBE BEST PARAM SET
    '''
    param_cols = ['elite_mean', 'last_starter_mean', 'replacement_mean', 'vopn_mean', 'dynamic_multiplier_mean', 'team_needs_mean']
    best_param_set = param_grading.index[0]
    best_params = param_grading[param_grading.index == best_param_set]
    print(f"\nBest param set: {best_param_set}")
    print(f"Best params:")
    for col in param_cols:
        print(f"\t{col}: {best_params[col].values[0]:.2f}")
    print(f"Best projection: {param_grading.loc[best_param_set, 'total_proj_mean']:.2f}")
    print(f"Sharpe: {param_grading.loc[best_param_set, 'sharpe']:.2f}")

    '''
        SAVE RESULTS
    '''
    if not TEST:
        results_df.to_csv(cfg.CACHE_DIR / "draft_simulation_results_new.csv", index=False)
        param_grading.to_csv(cfg.CACHE_DIR / "draft_simulation_param_grading_new.csv", index=False)
        results_df.to_parquet(cfg.CACHE_DIR / "draft_simulation_results_new.parquet", index=False)
        param_grading.to_parquet(cfg.CACHE_DIR / "draft_simulation_param_grading_new.parquet", index=False)
        print("\nFull results and param grading saved to cache.")


    '''
        GET BEST PARAM SET AND SIM TO FIND IDEAL PICKS
    '''
    top_share = 0.10
    top_count = int(len(param_grading) * top_share) + 1
    chosen_params = {
        'elite': param_grading.elite_mean.head(top_count).mean(),
        'last_starter': param_grading.last_starter_mean.head(top_count).mean(),
        'replacement': param_grading.replacement_mean.head(top_count).mean(),
        'vopn': param_grading.vopn_mean.head(top_count).mean(),
        'dynamic_multiplier': param_grading.dynamic_multiplier_mean.head(top_count).mean(),
        'team_needs': param_grading.team_needs_mean.head(top_count).mean(),
    }
    print("\nChosen params:")
    for col in chosen_params.keys():
        print(f"\t{col}: {chosen_params[col]:.2f}")

    best_df = run_all_simulations(
        df_params=pd.DataFrame([chosen_params]),
        base_df_players=raw_df.copy(),
        draft_cfg=cfg,
        n_sims=500,
        df_adp=df_adp
    ).sort_values(by='total_proj', ascending=False).reset_index(drop=True)

    top_8_picks = best_df.my_top_8_picks
    best_picks_df = pd.DataFrame(columns=['pick_1', 'pick_2', 'pick_3', 'pick_4', 'pick_5', 'pick_6', 'pick_7', 'pick_8'])

    for pick in range(1, 9):
        best_picks_df[f'pick_{pick}'] = top_8_picks.apply(lambda x: x[pick - 1] if len(x) >= pick else None)

    for pick in range(1, 9):
        print(f'\nPick {pick}: \n\t{best_picks_df[f"pick_{pick}"].value_counts().head(7)}')

    '''
        SENSE CHECK VALUES
    '''
    elite_param = chosen_params['elite']
    last_starter_param = chosen_params['last_starter']
    replacement_param = chosen_params['replacement'] if chosen_params['replacement'] > 0.0 else min(elite_param, last_starter_param) * 0.33
    sense_check_values = value_players(raw_df,
                                       static_value_weights={'elite': elite_param,
                                                             'last_starter': last_starter_param,
                                                             'replacement': replacement_param},
                                       projection_column_prefix=cfg.proj_col_prefix,
                                       vopn=int(chosen_params['vopn']),
                                       dynamic_multiplier=chosen_params['dynamic_multiplier'])
    sense_check_values = sense_check_values.merge(df_adp.drop(columns=['player', 'team', 'position']), on='id', how='left', validate='one_to_one')

    cheat_sheet = sense_check_values[['player', 'static_value']].copy()
    cheat_sheet['static_value'] = cheat_sheet.static_value.round(0)
    cheat_sheet['player'] = cheat_sheet['player'].apply(lambda x: ' '.join([word.capitalize() for word in x.split()]))
    today = pd.Timestamp.today().strftime("%Y%m%d")
    cheat_sheet.to_csv(cfg.CACHE_DIR / f"{today}_cheat_sheet.csv", index=False)
