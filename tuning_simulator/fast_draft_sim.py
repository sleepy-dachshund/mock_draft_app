# fast_draft_sim.py - Lightweight draft simulator for million+ simulations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


# ===== 1. DATA STRUCTURE DESIGN =====

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


def prepare_player_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Convert player dataframe to numpy arrays for speed"""
    # Sort by draft_value to make top player selection easier
    df_sorted = df.sort_values('draft_value', ascending=False).reset_index(drop=True)

    arrays = {
        'player_ids': np.arange(len(df_sorted)),  # Simple 0-based indexing
        'original_ids': df_sorted['id'].values,  # Keep original IDs for reference
        'positions': pd.Categorical(df_sorted['position']).codes.astype(np.int8),  # Convert positions to codes
        'position_names': pd.Categorical(df_sorted['position']).categories.values,  # Store position mapping
        'projections': df_sorted['median_projection'].values.astype(np.float32),
        'adp': df_sorted['adp'].values.astype(np.float32),
        'stdev': df_sorted['stdev'].fillna(10).values.astype(np.float32),  # Default stdev of 10
        'high': df_sorted['high'].values.astype(np.float32),
        'low': df_sorted['low'].values.astype(np.float32),
        'draft_value': df_sorted['draft_value'].values.astype(np.float32),
    }

    return arrays


def create_position_masks(player_positions: np.ndarray, position_names: np.ndarray) -> Dict[str, np.ndarray]:
    """Create boolean masks for each position for fast filtering"""
    masks = {}
    for i, pos_name in enumerate(position_names):
        masks[pos_name] = (player_positions == i)
    return masks


def precompute_starter_requirements(config: FastDraftConfig) -> Dict[str, int]:
    """Calculate total starter spots needed per position including flex"""
    requirements = {
        'QB': config.n_qb,
        'RB': config.n_rb,
        'WR': config.n_wr,
        'TE': config.n_te,
    }

    # Don't add flex to specific positions - we'll handle flex separately
    # since it can be filled by multiple positions

    return requirements


# ===== 2. FAST DRAFT SIMULATION ENGINE =====

def initialize_draft_order(n_teams: int, n_rounds: int) -> np.ndarray:
    """Pre-compute snake draft order as array of team numbers"""
    draft_order = np.zeros(n_teams * n_rounds, dtype=np.int8)

    for round_num in range(n_rounds):
        if round_num % 2 == 0:  # Even rounds (0-indexed): go forward
            draft_order[round_num * n_teams:(round_num + 1) * n_teams] = np.arange(n_teams)
        else:  # Odd rounds: go backward (snake)
            draft_order[round_num * n_teams:(round_num + 1) * n_teams] = np.arange(n_teams - 1, -1, -1)

    return draft_order


def create_team_rosters(n_teams: int) -> np.ndarray:
    """Initialize roster tracking arrays for all teams"""
    # Track count of each position for each team
    # Positions: QB=0, RB=1, WR=2, TE=3
    return np.zeros((n_teams, 4), dtype=np.int8)


def get_starter_filling_round(config: FastDraftConfig) -> int:
    """Calculate which round marks the transition from starter to bench drafting"""
    # Most teams will fill starters by this round
    # Add 1-2 rounds buffer since not everyone drafts optimally
    return config.n_starters + 2


# ===== 3. PHASE 1: STARTER-FILLING LOGIC (DEPENDENT PICKS) =====

def update_roster_needs(roster_state: np.ndarray, position: int, team: int) -> None:
    """Update team's position counts after a pick (in-place)"""
    roster_state[team, position] += 1


def calculate_team_needs(roster_state: np.ndarray, team: int, requirements: Dict[str, int]) -> np.ndarray:
    """Determine which positions team still needs for starters"""
    # Returns boolean array: [QB_needed, RB_needed, WR_needed, TE_needed]
    pos_map = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}
    needs = np.zeros(4, dtype=bool)

    for pos, idx in pos_map.items():
        needs[idx] = roster_state[team, idx] < requirements.get(pos, 0)

    return needs


def calculate_pick_probability_with_needs(player_idx: int, pick_num: int, adp: float, stdev: float,
                                          high: float, low: float, is_needed: bool) -> float:
    """Calculate selection probability for a player considering position need"""
    # Base probability from normal CDF
    z_score = (pick_num - adp) / max(stdev, 1.0)  # Avoid division by zero
    base_prob = stats.norm.cdf(z_score)

    # Adjust for pick ranges
    if pick_num < high:
        base_prob = 0.0
    elif pick_num > low:
        base_prob = min(base_prob * 5.0, 1.0)  # Boost overdue players

    # Adjust for position need
    if is_needed:
        base_prob = min(base_prob * 5.0, 1.0)  # Double probability for needed positions
    else:
        base_prob *= 0.1  # Reduce probability for unneeded positions during starter phase

    return base_prob


def simulate_starter_phase_pick(pick_num: int, available_mask: np.ndarray, team_needs: np.ndarray,
                                player_data: Dict[str, np.ndarray], position_masks: Dict[str, np.ndarray]) -> int:
    """Simulate pick during starter-filling phase when needs matter"""
    # Get available players
    available_indices = np.where(available_mask)[0]
    if len(available_indices) == 0:
        return -1

    # Calculate probabilities for each available player
    probabilities = np.zeros(len(available_indices))

    for i, player_idx in enumerate(available_indices):
        # Get player position
        player_pos = player_data['positions'][player_idx]

        # Check if this position is needed
        is_needed = team_needs[player_pos] if player_pos < len(team_needs) else False

        # Calculate probability
        prob = calculate_pick_probability_with_needs(
            player_idx=player_idx,
            pick_num=pick_num,
            adp=player_data['adp'][player_idx],
            stdev=player_data['stdev'][player_idx],
            high=player_data['high'][player_idx],
            low=player_data['low'][player_idx],
            is_needed=is_needed
        )
        probabilities[i] = prob

    # Normalize probabilities
    prob_sum = probabilities.sum()
    if prob_sum > 0:
        probabilities /= prob_sum
    else:
        # Fallback to uniform if all probabilities are 0
        probabilities = np.ones(len(probabilities)) / len(probabilities)

    # Make weighted random choice
    chosen_idx = np.random.choice(len(available_indices), p=probabilities)
    return available_indices[chosen_idx]


# ===== 4. PHASE 2: BENCH-FILLING LOGIC (INDEPENDENT PICKS) =====

def precompute_bench_phase_probabilities(adp: np.ndarray, stdev: np.ndarray, high: np.ndarray,
                                         low: np.ndarray, start_pick: int, end_pick: int) -> np.ndarray:
    """Pre-calculate all pick probabilities for bench phase where picks are independent"""
    n_players = len(adp)
    n_picks = end_pick - start_pick + 1

    # Create matrix: players x picks
    prob_matrix = np.zeros((n_players, n_picks), dtype=np.float32)

    for pick_offset in range(n_picks):
        pick_num = start_pick + pick_offset

        # Vectorized probability calculation for all players
        z_scores = (pick_num - adp) / np.maximum(stdev, 1.0)
        base_probs = stats.norm.cdf(z_scores)

        # Adjust for pick ranges
        base_probs = np.where(pick_num < high, 0.0, base_probs)
        base_probs = np.where(pick_num > low, np.minimum(base_probs * 5.0, 1.0), base_probs)

        prob_matrix[:, pick_offset] = base_probs

    return prob_matrix


def simulate_all_bench_picks(start_pick: int, available_mask: np.ndarray,
                             precomputed_probs: np.ndarray, draft_order: np.ndarray) -> np.ndarray:
    """Simulate all remaining picks at once using pre-computed probabilities"""
    n_remaining_picks = precomputed_probs.shape[1]
    picks = np.full(n_remaining_picks, -1, dtype=np.int32)

    # Copy available mask to track drafted players
    current_available = available_mask.copy()

    for pick_offset in range(n_remaining_picks):
        # Get available players
        available_indices = np.where(current_available)[0]
        if len(available_indices) == 0:
            break

        # Get probabilities for this pick slot
        pick_probs = precomputed_probs[available_indices, pick_offset]

        # Normalize
        prob_sum = pick_probs.sum()
        if prob_sum > 0:
            pick_probs /= prob_sum
        else:
            pick_probs = np.ones(len(pick_probs)) / len(pick_probs)

        # Make selection
        chosen_idx = np.random.choice(len(available_indices), p=pick_probs)
        selected_player = available_indices[chosen_idx]

        picks[pick_offset] = selected_player
        current_available[selected_player] = False

    return picks


# ===== 5. SIMULATION EXECUTION =====

def run_single_simulation(player_data: Dict[str, np.ndarray], config: FastDraftConfig,
                          random_seed: int) -> Tuple[np.ndarray, float, float, float, int]:
    """Execute one complete draft simulation with two phases, return draft + score + luck metrics"""
    np.random.seed(random_seed)

    n_players = len(player_data['player_ids'])
    n_total_picks = config.n_teams * config.n_rounds

    # Initialize tracking arrays
    draft_results = np.full(n_total_picks, -1, dtype=np.int32)
    available_mask = np.ones(n_players, dtype=bool)
    team_rosters = create_team_rosters(config.n_teams)

    # Get draft order and requirements
    draft_order = initialize_draft_order(config.n_teams, config.n_rounds)
    requirements = precompute_starter_requirements(config)
    position_masks = create_position_masks(player_data['positions'], player_data['position_names'])

    # Determine phase transition
    starter_round = get_starter_filling_round(config)
    phase_transition_pick = min(starter_round * config.n_teams, n_total_picks)

    # PHASE 1: Starter-filling phase
    for pick_num in range(phase_transition_pick):
        team_idx = draft_order[pick_num]

        # Calculate team needs
        team_needs = calculate_team_needs(team_rosters, team_idx, requirements)

        # Make pick
        player_idx = simulate_starter_phase_pick(
            pick_num=pick_num + 1,
            available_mask=available_mask,
            team_needs=team_needs,
            player_data=player_data,
            position_masks=position_masks
        )

        if player_idx >= 0:
            draft_results[pick_num] = player_idx
            available_mask[player_idx] = False

            # Update roster
            player_position = player_data['positions'][player_idx]
            if player_position < 4:
                update_roster_needs(team_rosters, player_position, team_idx)

    # PHASE 2: Bench-filling phase
    if phase_transition_pick < n_total_picks:
        bench_probs = precompute_bench_phase_probabilities(
            player_data['adp'],
            player_data['stdev'],
            player_data['high'],
            player_data['low'],
            phase_transition_pick + 1,
            n_total_picks
        )

        bench_picks = simulate_all_bench_picks(
            start_pick=phase_transition_pick + 1,
            available_mask=available_mask,
            precomputed_probs=bench_probs,
            draft_order=draft_order[phase_transition_pick:]
        )

        valid_bench_picks = bench_picks[bench_picks >= 0]
        draft_results[phase_transition_pick:phase_transition_pick + len(valid_bench_picks)] = valid_bench_picks

    # Calculate my team's score with luck metrics
    my_picks = []
    my_pick_numbers = []
    for pick_num in range(n_total_picks):
        if draft_order[pick_num] == config.draft_position - 1:
            if draft_results[pick_num] >= 0:
                my_picks.append(draft_results[pick_num])
                my_pick_numbers.append(pick_num + 1)  # 1-based pick number

    my_score, luck_sum, luck_avg_early = calculate_lineup_score_with_luck(
        team_picks=np.array(my_picks),
        pick_numbers=np.array(my_pick_numbers),
        projections=player_data['projections'],
        positions=player_data['positions'],
        adp_values=player_data['adp'],
        config=config
    )

    # Calculate scores for ALL teams
    team_scores = []
    for team_idx in range(config.n_teams):
        team_picks = []
        team_pick_numbers = []
        for pick_num in range(n_total_picks):
            if draft_order[pick_num] == team_idx:
                if draft_results[pick_num] >= 0:
                    team_picks.append(draft_results[pick_num])
                    team_pick_numbers.append(pick_num + 1)

        score, _, _ = calculate_lineup_score_with_luck(
            team_picks=np.array(team_picks),
            pick_numbers=np.array(team_pick_numbers),
            projections=player_data['projections'],
            positions=player_data['positions'],
            adp_values=player_data['adp'],
            config=config
        )
        team_scores.append(score)

    # Rank teams (1 = best)
    team_scores_array = np.array(team_scores)
    my_rank = (team_scores_array > my_score).sum() + 1

    return draft_results, my_score, luck_sum, luck_avg_early, my_rank


def calculate_lineup_score(team_picks: np.ndarray, projections: np.ndarray, positions: np.ndarray,
                           config: FastDraftConfig) -> float:
    """Calculate optimal starting lineup projection total"""
    if len(team_picks) == 0:
        return 0.0

    # Get projections and positions for team's picks
    team_projections = projections[team_picks]
    team_positions = positions[team_picks]

    total_score = 0.0

    # Fill required positions first
    pos_map = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}
    requirements = [config.n_qb, config.n_rb, config.n_wr, config.n_te]

    used_players = np.zeros(len(team_picks), dtype=bool)

    for pos_idx, n_required in enumerate(requirements):
        # Get players at this position
        pos_mask = team_positions == pos_idx
        pos_players = np.where(pos_mask & ~used_players)[0]

        if len(pos_players) > 0:
            # Sort by projection and take top N
            pos_projections = team_projections[pos_players]
            top_n = min(n_required, len(pos_players))
            top_indices = pos_players[np.argsort(pos_projections)[-top_n:]]

            total_score += team_projections[top_indices].sum()
            used_players[top_indices] = True

    # Fill flex spots with best remaining RB/WR/TE
    flex_positions = [1, 2, 3]  # RB, WR, TE
    flex_mask = np.isin(team_positions, flex_positions) & ~used_players
    flex_players = np.where(flex_mask)[0]

    if len(flex_players) > 0 and config.n_flex > 0:
        flex_projections = team_projections[flex_players]
        top_flex = min(config.n_flex, len(flex_players))
        top_flex_indices = flex_players[np.argsort(flex_projections)[-top_flex:]]

        total_score += team_projections[top_flex_indices].sum()

    return total_score


def calculate_lineup_score_with_luck(team_picks: np.ndarray, pick_numbers: np.ndarray,
                                     projections: np.ndarray, positions: np.ndarray,
                                     adp_values: np.ndarray, config: FastDraftConfig) -> Tuple[float, float, float]:
    """Calculate optimal starting lineup projection total and luck metrics"""
    if len(team_picks) == 0:
        return 0.0, 0.0, 0.0

    # Get projections and positions for team's picks
    team_projections = projections[team_picks]
    team_positions = positions[team_picks]
    team_adp = adp_values[team_picks]

    # Calculate luck factor for each pick
    luck_factors = pick_numbers - team_adp  # Positive = got player later than ADP (lucky)

    total_score = 0.0
    starter_luck_factors = []

    # Fill required positions first
    pos_map = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}
    requirements = [config.n_qb, config.n_rb, config.n_wr, config.n_te]

    used_players = np.zeros(len(team_picks), dtype=bool)

    for pos_idx, n_required in enumerate(requirements):
        # Get players at this position
        pos_mask = team_positions == pos_idx
        pos_players = np.where(pos_mask & ~used_players)[0]

        if len(pos_players) > 0:
            # Sort by projection and take top N
            pos_projections = team_projections[pos_players]
            top_n = min(n_required, len(pos_players))
            top_indices = pos_players[np.argsort(pos_projections)[-top_n:]]

            total_score += team_projections[top_indices].sum()
            used_players[top_indices] = True

            # Track luck for starters
            starter_luck_factors.extend(luck_factors[top_indices])

    # Fill flex spots with best remaining RB/WR/TE
    flex_positions = [1, 2, 3]  # RB, WR, TE
    flex_mask = np.isin(team_positions, flex_positions) & ~used_players
    flex_players = np.where(flex_mask)[0]

    if len(flex_players) > 0 and config.n_flex > 0:
        flex_projections = team_projections[flex_players]
        top_flex = min(config.n_flex, len(flex_players))
        top_flex_indices = flex_players[np.argsort(flex_projections)[-top_flex:]]

        total_score += team_projections[top_flex_indices].sum()
        starter_luck_factors.extend(luck_factors[top_flex_indices])

    # Calculate luck metrics
    luck_sum = sum(starter_luck_factors) if starter_luck_factors else 0.0
    luck_avg_early = np.mean(luck_factors[:7]) if len(luck_factors) >= 7 else 0.0

    return total_score, luck_sum, luck_avg_early


def run_simulations_batch(player_data: Dict[str, np.ndarray], config: FastDraftConfig,
                          n_sims: int, progress_callback=None) -> List[Tuple[np.ndarray, float, float, float, int]]:
    """Run multiple simulations sequentially with optional progress tracking"""
    results = []

    for i in range(n_sims):
        if progress_callback and i % 10 == 0:
            progress_callback(i, n_sims)

        draft_results, score, luck_sum, luck_avg_early, my_rank = run_single_simulation(player_data, config, random_seed=i)
        results.append((draft_results, score, luck_sum, luck_avg_early, my_rank))

    if progress_callback:
        progress_callback(n_sims, n_sims)

    return results


# ===== 6. RESULT STORAGE & ANALYSIS =====

def compress_draft_results(picks: np.ndarray, scores: np.ndarray) -> bytes:
    """Compress draft sequences and scores for efficient storage"""
    # Simple compression: store as int16 for picks (supports up to 32k players)
    # and float32 for scores
    import zlib

    picks_bytes = picks.astype(np.int16).tobytes()
    scores_bytes = scores.astype(np.float32).tobytes()

    combined = picks_bytes + scores_bytes
    compressed = zlib.compress(combined, level=6)

    return compressed


def analyze_simulation_results(results: List[Tuple[np.ndarray, float, float, float, int]], my_team_idx: int) -> Dict:
    """Calculate percentile rankings and success metrics"""
    scores = np.array([r[1] for r in results])

    percentiles = np.percentile(scores, [10, 25, 50, 75, 90, 95, 99])

    return {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'percentiles': {
            10: percentiles[0],
            25: percentiles[1],
            50: percentiles[2],
            75: percentiles[3],
            90: percentiles[4],
            95: percentiles[5],
            99: percentiles[6]
        },
        'score_distribution': scores
    }


def extract_draft_paths(results: List[Tuple[np.ndarray, float, float, float, int]], top_percentile: float = 0.9) -> Dict:
    """Extract successful draft paths for decision tree construction"""
    scores = np.array([r[1] for r in results])
    threshold = np.percentile(scores, top_percentile * 100)

    successful_drafts = [(r[0], r[1]) for r in results if r[1] >= threshold]

    # TODO: Build decision tree structure from successful drafts
    # This will be implemented when we build the decision tree interface

    return {
        'n_successful': len(successful_drafts),
        'threshold_score': threshold,
        'successful_drafts': successful_drafts
    }


# ===== MAIN ENTRY POINT =====

def run_massive_simulation(df: pd.DataFrame, config: FastDraftConfig, n_simulations: int = 1_000_000) -> Dict:
    """Main function to run massive simulation and return analyzed results"""
    print(f"Preparing data structures...")
    player_data = prepare_player_arrays(df)

    print(f"Running {n_simulations:,} simulations...")

    def progress_callback(current, total):
        progress_pct = int(100 * current / total)
        if progress_pct % 10 == 0 and progress_pct > int(100 * (current - 1) / total):
            print(f"Progress: {current:,}/{total:,} ({progress_pct}%)")

    results = run_simulations_batch(player_data, config, n_simulations, progress_callback)

    print("Analyzing results...")
    analysis = analyze_simulation_results(results, config.draft_position - 1)

    print("Extracting successful draft paths...")
    paths = extract_draft_paths(results, top_percentile=0.9)

    return {
        'analysis': analysis,
        'paths': paths,
        'player_data': player_data,
        'config': config,
        'all_drafts': results  # return all draft results
    }