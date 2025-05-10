import pandas as pd
import numpy as np
import logging
import os
import re
from typing import List, Dict, Tuple, Optional, Union, Any
from pandas import DataFrame, Series

import config
from run_data_gen import data_gen

normalized_df, normalization_factors_df, merged_df, flag_df = data_gen(save_output=False)

# 1. basic value_df prep (assume normalized_df is already available when class is initiated)
value_df = normalized_df.copy()

# identify projection cols to work with
projection_cols = [col for col in value_df.columns if col.startswith(config.PROJECTION_COLUMN_PREFIX)]

# median, high, low projections for all baselines
value_df['median_projection'] = value_df[projection_cols].median(axis=1)
value_df['high_projection'] = value_df[projection_cols].max(axis=1)
value_df['low_projection'] = value_df[projection_cols].min(axis=1)

# 2. begin STATIC value calculation by naming and defining thresholds -- these should be params so we can add multiple static value columns
projection_column = 'median_projection'
value_threshold_name = 'classic'
value_threshold_dict = {
    'QB': 10,  # these are subjective, should be params
    'RB': 24,
    'WR': 36,
    'TE': 10,
}

# 3. sort by median ALWAYS -- not projection column (i.e., we might want to calc value with high/low, but never baseline with high/low, only median)
value_df.sort_values(by='median_projection', ascending=False, inplace=True)
value_df.reset_index(drop=True, inplace=True)

# 4. loop through positions and find the threshold player & their projection

# initialize dict to hold baseline projections (this doesn't need ot be parameterized)
value_baseline_dict = {
    'QB': 0,  # 0s are placeholders for now, to update in loop below
    'RB': 0,
    'WR': 0,
    'TE': 0,
}

for position, threshold in value_threshold_dict.items():
    # trim to only players in position
    position_df = value_df[value_df['position'] == position].copy()

    # find threshold player
    baseline_row = position_df.iloc[threshold - 1]

    # find threshold projection
    baseline_player = baseline_row['player']
    baseline_projection = baseline_row[projection_column]
    print(f'Baseline "{value_threshold_name}" projection for {position}: {baseline_player}, {baseline_projection}')

    # add threshold to dict
    value_baseline_dict[position] = baseline_projection

# 5. add baseline projection to value_df and calculate value as max(0, projection_column - baseline)
value_df[f'baseline_{value_threshold_name}'] = value_df['position'].map(value_baseline_dict)
value_df[f'value_{value_threshold_name}'] = (value_df[projection_column] - value_df[f'baseline_{value_threshold_name}']).clip(lower=0).round(1)

