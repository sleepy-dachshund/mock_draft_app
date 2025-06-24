import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def static_value_weights_generator() -> np.ndarray:
    step = 1/3
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

def get_my_picks_list(rounds, teams, draft_pos) -> list:
    my_picks = []
    for round_num in range(1, rounds + 1):
        if round_num % 2 == 1:  # Odd rounds go 1 to N
            pick_num = (round_num - 1) * teams + draft_pos
        else:  # Even rounds go N to 1 (snake draft)
            pick_num = round_num * teams - draft_pos + 1
        my_picks.append(pick_num)
    return my_picks