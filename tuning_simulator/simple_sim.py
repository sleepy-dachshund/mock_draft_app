import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path

from typing import Dict, Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import config
from gen_values import value_players, get_raw_df
from sim_utils import get_my_picks_list

from fast_draft_sim import *

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

@dataclass
class FastDraftConfig:
    """Minimal configuration for fast simulation"""
    n_teams: int
    n_rounds: int
    draft_position: int
    n_qb: int
    n_rb: int
    n_wr: int
    n_te: int
    n_flex: int
    flex_positions: List[str]

    @property
    def n_starters(self) -> int:
        """Total number of starter spots across all positions"""
        return self.n_qb + self.n_rb + self.n_wr + self.n_te + self.n_flex

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
    draft_cfg = DraftFig()
    draft_cfg.other_teams_picks_dict = {i: get_my_picks_list(draft_cfg.n_rounds, draft_cfg.n_teams, i) for i in range(1, draft_cfg.n_teams + 1) if i != draft_cfg.draft_pos}

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
        df_adp = clean_adp_df(pull_adp(year=draft_cfg.YEAR, teams=draft_cfg.n_teams, position='all'))
        save_adp_data(df_adp)

    input_df = raw_df.merge(df_adp.drop(columns=['player', 'team', 'position']), how='left', on='id', validate='1:1')
    input_df[['adp', 'stdev', 'high', 'low']] = input_df[['adp', 'stdev', 'high', 'low']].fillna(method='ffill')

    filled_starter_positions = []  # will populate with e.g. 'QB', 'RB', 'WR', 'TE' as these positions are drafted (by me)
    df = value_players(
            df=input_df.copy(),
            static_value_weights={'elite': 0.10,
                                  'last_starter': 0.75,
                                  'replacement': 0.15},
            vopn=5,
            projection_column_prefix=draft_cfg.proj_col_prefix,
            dynamic_multiplier=0.05,
            filled_roster_spots=filled_starter_positions,
            team_needs=1.0,
            draft_mode=True)

    ''' ==================================================================
    START SIMULATIONS HERE
    ================================================================== '''
    N_SIMS = 50 if TEST else 5_000

    # Create FastDraftConfig from existing DraftFig settings
    fast_config = FastDraftConfig(
        n_teams=draft_cfg.n_teams,
        n_rounds=draft_cfg.n_rounds,
        draft_position=draft_cfg.draft_pos,
        n_qb=draft_cfg.n_qb,
        n_rb=draft_cfg.n_rb,
        n_wr=draft_cfg.n_wr,
        n_te=draft_cfg.n_te,
        n_flex=draft_cfg.n_flex,
        flex_positions=draft_cfg.flex_pos
    )

    print(f"Fast Draft Configuration:")
    print(f"  Teams: {fast_config.n_teams}")
    print(f"  Rounds: {fast_config.n_rounds}")
    print(f"  Your pick: #{fast_config.draft_position}")
    print(f"  Total starters: {fast_config.n_starters}")
    print(f"  Total roster spots: {draft_cfg.n_rounds}")

    # Prepare data for fast simulation
    print(f"\nPreparing {len(df)} players for simulation...")

    # Run single test simulation first
    print("\nRunning single test simulation...")
    player_data = prepare_player_arrays(df)

    import time
    start_time = time.time()
    test_draft, test_score, test_luck_sum, test_luck_avg_early, test_rank = run_single_simulation(player_data, fast_config, random_seed=42)
    single_time = time.time() - start_time

    print(f"Test simulation completed in {single_time:.3f} seconds")
    print(f"Test team score: {test_score:.1f}")
    print(f"Test team rank: {test_rank}")
    print(f"Valid picks in draft: {np.sum(test_draft >= 0)}")

    # Estimate time for full simulation
    if TEST:
        print(f"\nEstimated time for {N_SIMS:,} simulations: {single_time * N_SIMS:.1f} seconds")
    else:
        print(f"\nEstimated time for {N_SIMS:,} simulations: {single_time * N_SIMS / 60:.1f} minutes")

    # Run the actual simulations
    print(f"\nRunning {N_SIMS:,} simulations...")
    start_time = time.time()

    results = run_massive_simulation(df, fast_config, n_simulations=N_SIMS)

    elapsed_time = time.time() - start_time
    print(f"\nCompleted {N_SIMS:,} simulations in {elapsed_time:.1f} seconds")
    print(f"Average time per simulation: {elapsed_time/N_SIMS*1000:.2f} ms")

    # Display results
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)

    analysis = results['analysis']
    print(f"\nTeam Score Statistics:")
    print(f"  Mean: {analysis['mean_score']:.1f}")
    print(f"  Std Dev: {analysis['std_score']:.1f}")
    print(f"  Percentiles:")
    for pct, value in analysis['percentiles'].items():
        print(f"    {pct}th: {value:.1f}")

    # Show successful draft threshold
    paths = results['paths']
    print(f"\nSuccessful Drafts (90th percentile+):")
    print(f"  Threshold score: {paths['threshold_score']:.1f}")
    print(f"  Number of successful drafts: {paths['n_successful']:,} / {N_SIMS:,}")
    print(f"  Success rate: {100 * paths['n_successful'] / N_SIMS:.1f}%")

    # Show sample successful draft if available
    if paths['successful_drafts']:
        print("\n" + "="*50)
        print("SAMPLE SUCCESSFUL DRAFT")
        print("="*50)

        # Get the best draft
        best_draft_idx = np.argmax([d[1] for d in paths['successful_drafts']])
        best_draft, best_score = paths['successful_drafts'][best_draft_idx]

        print(f"\nBest draft score: {best_score:.1f}")

        # Extract my picks from the best draft
        draft_order = initialize_draft_order(fast_config.n_teams, fast_config.n_rounds)
        my_pick_indices = np.where(draft_order == fast_config.draft_position - 1)[0]

        print("\nYour picks in best draft:")
        for round_num, pick_idx in enumerate(my_pick_indices, 1):
            if pick_idx < len(best_draft) and best_draft[pick_idx] >= 0:
                player_idx = best_draft[pick_idx]
                player_id = player_data['original_ids'][player_idx]
                player_row = df[df['id'] == player_id].iloc[0]

                # Show pick number (overall)
                overall_pick = pick_idx + 1
                print(f"  Round {round_num:2d} (Pick #{overall_pick:3d}): "
                      f"{player_row['player']:25s} {player_row['position']:3s} - "
                      f"Proj: {player_row['median_projection']:5.1f}, "
                      f"Value: {player_row['draft_value']:5.1f}, "
                      f"ADP: {player_row['adp']:5.1f}")

        # Show starting lineup
        print("\nOptimal Starting Lineup:")
        position_order = ['QB', 'RB', 'WR', 'TE']
        my_players = []

        for pick_idx in my_pick_indices:
            if pick_idx < len(best_draft) and best_draft[pick_idx] >= 0:
                player_idx = best_draft[pick_idx]
                player_id = player_data['original_ids'][player_idx]
                player_row = df[df['id'] == player_id].iloc[0]
                my_players.append(player_row)

        my_players_df = pd.DataFrame(my_players)

        # Show starters by position
        total_proj = 0
        for pos in position_order:
            pos_players = my_players_df[my_players_df['position'] == pos].sort_values('median_projection', ascending=False)
            n_needed = getattr(fast_config, f'n_{pos.lower()}')

            if len(pos_players) > 0 and n_needed > 0:
                print(f"\n{pos}:")
                for i, (_, player) in enumerate(pos_players.head(n_needed).iterrows()):
                    print(f"  {player['player']:25s} - Proj: {player['median_projection']:5.1f}")
                    total_proj += player['median_projection']

        # Handle FLEX
        flex_players = my_players_df[my_players_df['position'].isin(fast_config.flex_positions)]
        # Remove already used starters
        used_players = []
        for pos in ['RB', 'WR', 'TE']:
            n_needed = getattr(fast_config, f'n_{pos.lower()}')
            used_players.extend(
                my_players_df[my_players_df['position'] == pos]
                .sort_values('median_projection', ascending=False)
                .head(n_needed)
                .index.tolist()
            )

        flex_available = flex_players[~flex_players.index.isin(used_players)].sort_values('median_projection', ascending=False)

        if len(flex_available) > 0 and fast_config.n_flex > 0:
            print(f"\nFLEX:")
            for i, (_, player) in enumerate(flex_available.head(fast_config.n_flex).iterrows()):
                print(f"  {player['player']:25s} ({player['position']}) - Proj: {player['median_projection']:5.1f}")
                total_proj += player['median_projection']

        print(f"\nTotal Projected Points: {total_proj:.1f}")
        print(f"(Reported score from simulation: {best_score:.1f})")

    # Save results if not in test mode
    if not TEST:
        print("\n" + "="*50)
        print("SAVING RESULTS")
        print("="*50)

        # Save summary statistics
        summary_df = pd.DataFrame({
            'metric': ['mean', 'std', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99'],
            'value': [
                analysis['mean_score'],
                analysis['std_score'],
                analysis['percentiles'][10],
                analysis['percentiles'][25],
                analysis['percentiles'][50],
                analysis['percentiles'][75],
                analysis['percentiles'][90],
                analysis['percentiles'][95],
                analysis['percentiles'][99]
            ]
        })

        summary_path = draft_cfg.CACHE_DIR / "fast_sim_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")

        # Save successful drafts for decision tree construction
        import pickle
        paths_path = draft_cfg.CACHE_DIR / "successful_draft_paths.pkl"
        with open(paths_path, 'wb') as f:
            pickle.dump(paths, f)
        print(f"Successful draft paths saved to: {paths_path}")






    print(f"\n{'=' * 50}")
    print("DRAFT DATA INSPECTION")
    print(f"{'=' * 50}")

    # Access all drafts
    all_drafts = results.get('all_drafts', [])
    print(f"\nTotal drafts captured: {len(all_drafts):,}")

    # Inspect structure
    if all_drafts:
        sample_draft, sample_score, sample_luck_sum, sample_luck_avg_early, sample_team_rank = all_drafts[0]
        print(f"Each draft contains:")
        print(f"  - Draft array shape: {sample_draft.shape}")
        print(f"  - Score: {sample_score:.1f}")
        print(f"  - Luck Sum: {sample_luck_sum:.1f}")
        print(f"  - Luck Avg Early: {sample_luck_avg_early:.1f}")

    # Analyze draft distribution
    all_scores = np.array([score for _, score, _, _, _ in all_drafts])
    print(f"\nScore distribution across all drafts:")
    print(f"  Min: {all_scores.min():.1f}")
    print(f"  Max: {all_scores.max():.1f}")
    print(f"  Range: {all_scores.max() - all_scores.min():.1f}")
    print(f"\nLuck distribution across all drafts:")
    all_luck_sums = np.array([luck_sum for _, _, luck_sum, _, _ in all_drafts])
    print(f"  Min: {all_luck_sums.min():.1f}")
    print(f"  Max: {all_luck_sums.max():.1f}")
    print(f"  Range: {all_luck_sums.max() - all_luck_sums.min():.1f}")
    all_luck_avgs = np.array([luck_avg for _, _, _, luck_avg, _ in all_drafts])
    print(f"\nLuck Avg distribution across all drafts:")
    print(f"  Min: {all_luck_avgs.min():.1f}")
    print(f"  Max: {all_luck_avgs.max():.1f}")
    print(f"  Range: {all_luck_avgs.max() - all_luck_avgs.min():.1f}")

    # For decision tree construction, we'll want to extract pick sequences
    print(f"\nPreparing data for decision tree...")

    # Create a matrix where each row is a draft, each column is a pick
    n_drafts = len(all_drafts)
    n_picks = fast_config.n_teams * fast_config.n_rounds
    draft_matrix = np.zeros((n_drafts, n_picks), dtype=np.int32)
    score_array = np.zeros(n_drafts)

    for i, (draft, score, luck_sum, luck_avg_early, team_rank) in enumerate(all_drafts):
        draft_matrix[i] = draft
        score_array[i] = score

    print(f"Draft matrix shape: {draft_matrix.shape}")
    print(f"Score array shape: {score_array.shape}")

    # Extract just my picks for decision tree
    draft_order = initialize_draft_order(fast_config.n_teams, fast_config.n_rounds)
    my_pick_indices = np.where(draft_order == fast_config.draft_position - 1)[0]

    # Create matrix of just my picks across all drafts
    my_picks_matrix = draft_matrix[:, my_pick_indices]
    print(f"\nMy picks matrix shape: {my_picks_matrix.shape}")
    print(f"  ({n_drafts:,} drafts × {len(my_pick_indices)} picks)")

    # Save this data for decision tree construction
    if not TEST:
        import pickle

        tree_data = {
            'my_picks_matrix': my_picks_matrix,
            'all_picks_matrix': draft_matrix,
            'scores': score_array,
            'draft_order': draft_order,
            'my_pick_indices': my_pick_indices,
            'player_data': results['player_data'],
            'config': fast_config
        }

        tree_data_path = draft_cfg.CACHE_DIR / "decision_tree_data.pkl"
        with open(tree_data_path, 'wb') as f:
            pickle.dump(tree_data, f)
        print(f"\nDecision tree data saved to: {tree_data_path}")

    print(f"\n{'=' * 50}")
    print("LUCK ANALYSIS")
    print(f"{'=' * 50}")

    # Extract luck metrics
    all_scores = np.array([r[1] for r in results['all_drafts']])
    all_luck_sum = np.array([r[2] for r in results['all_drafts']])
    all_luck_avg_early = np.array([r[3] for r in results['all_drafts']])
    all_ranks = np.array([r[4] for r in results['all_drafts']])

    # Calculate luck percentiles
    luck_percentiles = np.percentile(all_luck_sum, [10, 25, 50, 75, 90, 95, 99])

    print(f"\nLuck Sum Statistics (sum of luck factors for all starters):")
    print(f"  Mean: {all_luck_sum.mean():.1f}")
    print(f"  Std Dev: {all_luck_sum.std():.1f}")
    print(f"  Percentiles:")
    for i, pct in enumerate([10, 25, 50, 75, 90, 95, 99]):
        print(f"    {pct}th: {luck_percentiles[i]:.1f}")

    print(f"\nLuck Avg Early Statistics (avg luck factor for first 4 picks):")
    early_percentiles = np.percentile(all_luck_avg_early, [10, 25, 50, 75, 90, 95, 99])
    for i, pct in enumerate([10, 25, 50, 75, 90, 95, 99]):
        print(f"    {pct}th: {early_percentiles[i]:.1f}")

    # Identify realistic successful drafts
    print(f"\n{'=' * 50}")
    print("REALISTIC SUCCESSFUL DRAFTS")
    print(f"{'=' * 50}")

    # Filter out top 10% luckiest drafts
    luck_90th_percentile = np.percentile(all_luck_sum, 90)
    not_too_lucky_mask = all_luck_sum <= luck_90th_percentile

    print(f"\nRemoving top 10% luckiest drafts (luck_sum > {luck_90th_percentile:.1f})")
    print(f"Drafts remaining: {not_too_lucky_mask.sum():,} / {len(all_scores):,}")

    # Among non-lucky drafts, find top 10% by score
    realistic_scores = all_scores[not_too_lucky_mask]
    realistic_90th_percentile = np.percentile(realistic_scores, 90)

    print(f"\nAmong realistic drafts, 90th percentile score: {realistic_90th_percentile:.1f}")

    # Find drafts that are both realistic and successful
    realistic_successful_mask = not_too_lucky_mask & (all_scores >= realistic_90th_percentile)
    n_realistic_successful = realistic_successful_mask.sum()

    print(f"Realistic successful drafts: {n_realistic_successful:,}")
    print(f"Success rate: {100 * n_realistic_successful / len(all_scores):.1f}%")

    # Show best realistic draft
    if n_realistic_successful > 0:
        realistic_indices = np.where(realistic_successful_mask)[0]
        best_realistic_idx = realistic_indices[np.argmax(all_scores[realistic_successful_mask])]

        best_draft = results['all_drafts'][best_realistic_idx][0]
        best_score = all_scores[best_realistic_idx]
        best_luck_sum = all_luck_sum[best_realistic_idx]
        best_luck_early = all_luck_avg_early[best_realistic_idx]

        print(f"\n{'=' * 50}")
        print("BEST REALISTIC DRAFT")
        print(f"{'=' * 50}")
        print(f"Score: {best_score:.1f}")
        print(f"Luck Sum: {best_luck_sum:.1f}")
        print(f"Luck Avg Early: {best_luck_early:.1f}")

        # Show picks with luck factor
        draft_order = initialize_draft_order(fast_config.n_teams, fast_config.n_rounds)
        my_pick_indices = np.where(draft_order == fast_config.draft_position - 1)[0]

        print("\nYour picks (with luck factor):")
        for round_num, pick_idx in enumerate(my_pick_indices, 1):
            if pick_idx < len(best_draft) and best_draft[pick_idx] >= 0:
                player_idx = best_draft[pick_idx]
                player_id = player_data['original_ids'][player_idx]
                player_row = df[df['id'] == player_id].iloc[0]

                overall_pick = pick_idx + 1
                luck_factor = overall_pick - player_row['adp']

                print(f"  Round {round_num:2d} (Pick #{overall_pick:3d}): "
                      f"{player_row['player']:25s} {player_row['position']:3s} - "
                      f"ADP: {player_row['adp']:5.1f}, "
                      f"Luck: {luck_factor:+5.1f}")

    # Save realistic successful drafts
    if not TEST:
        realistic_successful_data = {
            'drafts': [results['all_drafts'][i] for i in np.where(realistic_successful_mask)[0]],
            'mask': realistic_successful_mask,
            'luck_threshold': luck_90th_percentile,
            'score_threshold': realistic_90th_percentile
        }

        import pickle

        realistic_path = draft_cfg.CACHE_DIR / "realistic_successful_drafts.pkl"
        with open(realistic_path, 'wb') as f:
            pickle.dump(realistic_successful_data, f)
        print(f"\nRealistic successful drafts saved to: {realistic_path}")

    print(f"\n{'=' * 50}")
    print("CREATING DRAFT RESULTS DATAFRAME")
    print(f"{'=' * 50}")

    # Extract all data from results
    all_drafts = results['all_drafts']
    all_scores = np.array([r[1] for r in all_drafts])
    all_luck_sum = np.array([r[2] for r in all_drafts])
    all_luck_avg_early = np.array([r[3] for r in all_drafts])

    # Calculate percentiles for each metric
    score_pctiles = 100 * stats.rankdata(all_scores, method='average') / len(all_scores)
    luck_sum_pctiles = 100 * stats.rankdata(all_luck_sum, method='average') / len(all_luck_sum)
    luck_avg_early_pctiles = 100 * stats.rankdata(all_luck_avg_early, method='average') / len(all_luck_avg_early)

    # Get draft order and my pick indices
    draft_order = initialize_draft_order(fast_config.n_teams, fast_config.n_rounds)
    my_pick_indices = np.where(draft_order == fast_config.draft_position - 1)[0]

    # Only include rounds up to number of starters
    n_starter_rounds = fast_config.n_starters
    my_starter_pick_indices = my_pick_indices[:n_starter_rounds]

    # Build the dataframe
    draft_data = []

    for i, (draft_array, score, luck_sum, luck_avg_early, my_rank) in enumerate(all_drafts):
        row_data = {
            'draft_rank': 0,  # Will be set after sorting
            'score': score,
            'score_pctile': score_pctiles[i],
            'team_rank': my_rank,
            'luck_sum': luck_sum,
            'luck_sum_pctile': luck_sum_pctiles[i],
            'luck_avg_early': luck_avg_early,
            'luck_avg_early_pctile': luck_avg_early_pctiles[i]
        }

        # Add player names for each starter round
        for round_num, pick_idx in enumerate(my_starter_pick_indices, 1):
            if pick_idx < len(draft_array) and draft_array[pick_idx] >= 0:
                player_idx = draft_array[pick_idx]
                player_id = player_data['original_ids'][player_idx]
                player_row = df[df['id'] == player_id].iloc[0]
                player_name = player_row['player']
            else:
                player_name = 'NONE'

            row_data[f'round_{round_num}'] = player_name

        draft_data.append(row_data)

    # Create DataFrame
    draft_results_df = pd.DataFrame(draft_data)

    # Sort by score descending and assign draft_rank
    draft_results_df = draft_results_df.sort_values('score', ascending=False).reset_index(drop=True)
    draft_results_df['draft_rank'] = range(1, len(draft_results_df) + 1)

    # Reorder columns to put draft_rank first
    cols = ['draft_rank', 'score', 'score_pctile', 'team_rank', 'luck_sum', 'luck_sum_pctile',
            'luck_avg_early', 'luck_avg_early_pctile']
    round_cols = [f'round_{i}' for i in range(1, n_starter_rounds + 1)]
    draft_results_df = draft_results_df[cols + round_cols]

    # Display summary
    print(f"\nDataFrame created with {len(draft_results_df):,} rows")
    print(f"Columns: {list(draft_results_df.columns)}")

    # Show top 10 drafts
    print(f"\nTop 10 drafts by score:")
    display_cols = ['draft_rank', 'score', 'team_rank', 'luck_sum', 'round_1', 'round_2', 'round_3']
    print(draft_results_df[display_cols].head(10).to_string(index=False))

    # Save to CSV (top 10,000 only)
    n_to_save = min(10_000, len(draft_results_df))
    output_filename = draft_cfg.CACHE_DIR / f"draft_results_top_{n_to_save}.csv"

    draft_results_df.head(n_to_save).to_csv(output_filename, index=False)
    print(f"\nSaved top {n_to_save:,} drafts to: {output_filename}")

    # Show filtering example
    print(f"\n{'=' * 50}")
    print("FILTERING EXAMPLE")
    print(f"{'=' * 50}")

    # Get most common round 1 pick
    round_1_counts = draft_results_df['round_1'].value_counts()
    most_common_r1 = round_1_counts.index[0]

    print(f"\nMost common Round 1 pick: {most_common_r1}")
    print(f"Picked in {round_1_counts.iloc[0]:,} / {len(draft_results_df):,} drafts")

    # Filter to drafts with this pick
    filtered_df = draft_results_df[draft_results_df['round_1'] == most_common_r1]
    print(f"\nAfter filtering for '{most_common_r1}' in Round 1:")
    print(f"Drafts remaining: {len(filtered_df):,}")
    print(f"Average score: {filtered_df['score'].mean():.1f}")
    print(f"Score range: {filtered_df['score'].min():.1f} - {filtered_df['score'].max():.1f}")

    # Show most common round 2 picks given round 1
    print(f"\nMost common Round 2 picks after {most_common_r1}:")
    round_2_given_r1 = filtered_df['round_2'].value_counts().head(5)
    for player, count in round_2_given_r1.items():
        pct = 100 * count / len(filtered_df)
        avg_score = filtered_df[filtered_df['round_2'] == player]['score'].mean()
        print(f"  {player:25s} - {pct:5.1f}% (avg score: {avg_score:.1f})")

    # Show realistic vs lucky drafts
    print(f"\n{'=' * 50}")
    print("REALISTIC DRAFTS FILTER")
    print(f"{'=' * 50}")

    # Define realistic as luck_sum_pctile <= 90
    realistic_df = draft_results_df[draft_results_df['luck_sum_pctile'] <= 90]
    print(f"\nRealistic drafts (luck_sum_pctile <= 90): {len(realistic_df):,}")
    print(f"Best realistic score: {realistic_df['score'].iloc[0]:.1f}")
    print(f"Best realistic draft rank: {realistic_df['draft_rank'].iloc[0]}")

    # Save realistic drafts separately if desired
    if not TEST:
        realistic_output = draft_cfg.CACHE_DIR / f"draft_results_realistic_top_{n_to_save}.csv"
        realistic_df.head(n_to_save).to_csv(realistic_output, index=False)
        print(f"Saved realistic drafts to: {realistic_output}")
    else:
        test_realistic_output = draft_cfg.CACHE_DIR / f"test_draft_results_realistic_top_{n_to_save}.csv"
        realistic_df.head(n_to_save).to_csv(test_realistic_output, index=False)
        print(f"Saved realistic drafts to: {test_realistic_output}")