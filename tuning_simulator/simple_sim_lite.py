# test_fast_sim.py - Test script for the fast draft simulator

import warnings

warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path to import from main project
sys.path.append(str(Path(__file__).parent.parent))

import config
from gen_values import get_raw_df, value_players
from sim_utils import get_my_picks_list
from pull_adp import pull_adp, clean_adp_df, save_adp_data

# Import our new fast simulator
from fast_draft_sim import (
    FastDraftConfig,
    run_massive_simulation,
    prepare_player_arrays,
    run_single_simulation
)


def test_basic_functionality():
    """Test basic functions with sample data"""
    print("Testing basic functionality...")

    # Create test config
    test_config = FastDraftConfig(
        n_teams=10,
        n_rounds=15,
        draft_position=5,
        n_qb=1,
        n_rb=2,
        n_wr=2,
        n_te=1,
        n_flex=2,
        flex_positions=['RB', 'WR', 'TE']
    )

    print(f"Config created. Total starters: {test_config.n_starters}")
    print(f"Starter filling should end around round: {test_config.n_starters // test_config.n_teams + 2}")

    return test_config


def test_with_real_data():
    """Test with actual fantasy football data"""
    print("\n" + "=" * 50)
    print("Testing with real data...")
    print("=" * 50 + "\n")

    # Load raw data
    print("Loading player data...")
    raw_df = get_raw_df().reset_index(drop=True)
    print(f"Loaded {len(raw_df)} players")

    # Load ADP data
    print("\nLoading ADP data...")
    try:
        import datetime
        today = datetime.date.today().strftime("%Y%m%d")
        output_path = config.OUTPUT_DIR / f"{today}_adp.csv"
        df_adp = pd.read_csv(output_path)
    except FileNotFoundError:
        print("ADP file not found, pulling fresh data...")
        df_adp = clean_adp_df(pull_adp(year=config.SEASON_YEAR, teams=config.N_TEAMS, position='all'))
        save_adp_data(df_adp)

    # Merge ADP data
    input_df = raw_df.merge(
        df_adp.drop(columns=['player', 'team', 'position']),
        how='left',
        on='id',
        validate='1:1'
    )

    print(f"Merged data has {len(input_df)} players")
    print(f"Players with ADP data: {input_df['adp'].notna().sum()}")

    # Fill missing ADP data with reasonable defaults
    input_df['adp'] = input_df['adp'].fillna(method='ffill')
    input_df['stdev'] = input_df['stdev'].fillna(method='ffill')
    input_df['high'] = input_df['high'].fillna(method='ffill')
    input_df['low'] = input_df['low'].fillna(method='ffill')

    # Create config from project settings
    fast_config = FastDraftConfig(
        n_teams=config.N_TEAMS,
        n_rounds=config.N_TEAMS * (config.ROSTER_N_QB + config.ROSTER_N_RB +
                                   config.ROSTER_N_WR + config.ROSTER_N_TE +
                                   config.ROSTER_N_FLEX + config.ROSTER_N_BENCH) // config.N_TEAMS,
        draft_position=config.DRAFT_POSITION,
        n_qb=config.ROSTER_N_QB,
        n_rb=config.ROSTER_N_RB,
        n_wr=config.ROSTER_N_WR,
        n_te=config.ROSTER_N_TE,
        n_flex=config.ROSTER_N_FLEX,
        flex_positions=config.ROSTER_FLEX_ELIGIBLE_POSITIONS
    )

    filled_starter_positions = []  # will populate with e.g. 'QB', 'RB', 'WR', 'TE' as these positions are drafted (by me)
    input_df = value_players(
        df=input_df.copy(),
        static_value_weights={'elite': 0.10,
                              'last_starter': 0.75,
                              'replacement': 0.15},
        vopn=5,
        projection_column_prefix=fast_config.proj_col_prefix,
        dynamic_multiplier=0.05,
        filled_roster_spots=filled_starter_positions,
        team_needs=1.0,
        draft_mode=True)

    print(f"\nDraft Configuration:")
    print(f"  Teams: {fast_config.n_teams}")
    print(f"  Rounds: {fast_config.n_rounds}")
    print(f"  Your pick: #{fast_config.draft_position}")
    print(f"  Starters: {fast_config.n_starters}")

    # Test data preparation
    print("\nTesting data preparation...")
    player_data = prepare_player_arrays(input_df)
    print(f"Player arrays created with {len(player_data['player_ids'])} players")
    print(f"Position breakdown: {pd.Series(player_data['positions']).value_counts().to_dict()}")

    # Test single simulation
    print("\nRunning single simulation test...")
    start_time = time.time()
    draft_results, my_score = run_single_simulation(player_data, fast_config, random_seed=42)
    single_sim_time = time.time() - start_time

    print(f"Single simulation completed in {single_sim_time:.3f} seconds")
    print(f"My team score: {my_score:.1f}")
    print(f"Valid picks made: {np.sum(draft_results >= 0)}/{len(draft_results)}")

    # Estimate time for large simulation
    estimated_time_1m = single_sim_time * 1_000_000 / 60  # Convert to minutes
    print(f"\nEstimated time for 1M simulations: {estimated_time_1m:.1f} minutes")

    # Run small batch test
    print("\nRunning small batch test (1,000 simulations)...")
    start_time = time.time()
    results = run_massive_simulation(input_df, fast_config, n_simulations=1000)
    batch_time = time.time() - start_time

    print(f"\nBatch simulation completed in {batch_time:.2f} seconds")
    print(f"Results summary:")
    print(f"  Mean score: {results['analysis']['mean_score']:.1f}")
    print(f"  Std dev: {results['analysis']['std_score']:.1f}")
    print(f"  Percentiles: {results['analysis']['percentiles']}")
    print(f"  Successful drafts (>90th percentile): {results['paths']['n_successful']}")

    # Show sample successful draft
    if results['paths']['successful_drafts']:
        sample_draft, sample_score = results['paths']['successful_drafts'][0]
        print(f"\nSample successful draft (score: {sample_score:.1f}):")

        # Extract my picks from the draft
        draft_order = np.tile(np.arange(fast_config.n_teams), fast_config.n_rounds)
        for i in range(fast_config.n_rounds):
            if i % 2 == 1:  # Snake draft reversal
                start_idx = i * fast_config.n_teams
                end_idx = (i + 1) * fast_config.n_teams
                draft_order[start_idx:end_idx] = draft_order[start_idx:end_idx][::-1]

        my_pick_indices = np.where(draft_order == fast_config.draft_position - 1)[0]

        print("Your picks:")
        for round_num, pick_idx in enumerate(my_pick_indices[:8], 1):  # Show first 8 rounds
            if pick_idx < len(sample_draft) and sample_draft[pick_idx] >= 0:
                player_idx = sample_draft[pick_idx]
                player_id = player_data['original_ids'][player_idx]
                player_row = input_df[input_df['id'] == player_id].iloc[0]
                print(f"  Round {round_num}: {player_row['player']} ({player_row['position']}) - "
                      f"Proj: {player_row['median_projection']:.1f}")

    return results


def test_performance_scaling():
    """Test performance with different simulation counts"""
    print("\n" + "=" * 50)
    print("Testing performance scaling...")
    print("=" * 50 + "\n")

    # Load data once
    raw_df = get_raw_df().reset_index(drop=True)
    try:
        import datetime
        today = datetime.date.today().strftime("%Y%m%d")
        output_path = config.OUTPUT_DIR / f"{today}_adp.csv"
        df_adp = pd.read_csv(output_path)
    except FileNotFoundError:
        df_adp = clean_adp_df(pull_adp(year=config.SEASON_YEAR, teams=config.N_TEAMS, position='all'))

    input_df = raw_df.merge(df_adp.drop(columns=['player', 'team', 'position']), how='left', on='id', validate='1:1')
    input_df['adp'] = input_df['adp'].fillna(200)
    input_df['stdev'] = input_df['stdev'].fillna(10)
    input_df['high'] = input_df['high'].fillna(input_df['adp'] - 20)
    input_df['low'] = input_df['low'].fillna(input_df['adp'] + 20)

    fast_config = FastDraftConfig(
        n_teams=config.N_TEAMS,
        n_rounds=15,  # Simplified for testing
        draft_position=config.DRAFT_POSITION,
        n_qb=config.ROSTER_N_QB,
        n_rb=config.ROSTER_N_RB,
        n_wr=config.ROSTER_N_WR,
        n_te=config.ROSTER_N_TE,
        n_flex=config.ROSTER_N_FLEX,
        flex_positions=config.ROSTER_FLEX_ELIGIBLE_POSITIONS
    )

    # Test different simulation counts
    test_counts = [100, 1000, 10000]

    for n_sims in test_counts:
        print(f"\nTesting {n_sims:,} simulations...")
        start_time = time.time()

        results = run_massive_simulation(input_df, fast_config, n_simulations=n_sims)

        elapsed = time.time() - start_time
        per_sim = elapsed / n_sims * 1000  # milliseconds per simulation

        print(f"  Total time: {elapsed:.2f} seconds")
        print(f"  Per simulation: {per_sim:.2f} ms")
        print(f"  Simulations/second: {n_sims / elapsed:.0f}")

        # Extrapolate to 1M simulations
        estimated_1m = (per_sim * 1_000_000) / 1000 / 60  # Convert to minutes
        print(f"  Estimated for 1M: {estimated_1m:.1f} minutes")


if __name__ == "__main__":
    # Run all tests
    print("Starting Fast Draft Simulator Tests")
    print("=" * 50)

    # Test 1: Basic functionality
    test_config = test_basic_functionality()

    # Test 2: Real data test
    results = test_with_real_data()

    # Test 3: Performance scaling
    test_performance_scaling()

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)