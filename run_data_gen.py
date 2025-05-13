import pandas as pd
import numpy as np
import logging
import os
import re
from typing import List, Dict, Tuple, Optional, Union, Any
from pandas import DataFrame, Series

import config
from data_loader import DataLoader
from data_cleaner import DataCleaner
from data_merger import DataMerger
from data_normalizer import DataNormalizer

# Set up logging
logger = logging.getLogger(__name__)

def data_gen(trim_output: bool = True, save_output: bool = True) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:

    # load data
    loader = DataLoader(
        season_year=config.SEASON_YEAR,
        historical_year=config.HISTORICAL_YEAR,
        projections_dir=str(config.PROJECTIONS_DIR),
        historical_dir=str(config.HISTORICAL_DIR)
    )
    projection_dfs, source_experts = loader.load_projections()
    historical_df = loader.load_historical()

    # clean data
    cleaner = DataCleaner(config)
    cleaned_projections = cleaner.clean_projection_dataframes(projection_dfs, source_experts)
    cleaned_historical = cleaner.clean_historical_dataframe(historical_df)

    # merge data
    merger = DataMerger(config, cleaned_projections)
    merged_df, flag_df = merger.merge_projections_on_player_id()

    # normalize data
    normalizer = DataNormalizer(config, merged_df)
    normalized_df = normalizer.normalize_projection_data()
    normalization_factors_df = normalizer.get_normalization_factors()

    if trim_output:
        projection_cols = [col for col in normalized_df.columns if col.startswith(config.PROJECTION_COLUMN_PREFIX)]
        normalized_df['median_projection'] = normalized_df[projection_cols].median(axis=1).round(1)
        normalized_df['rank_pos'] = normalized_df.groupby('position')['median_projection'].rank(ascending=False, method='min')

        ls_qb = config.ROSTER_N_QB * config.N_TEAMS
        ls_rb = int(config.ROSTER_N_RB * config.N_TEAMS + (config.ROSTER_N_FLEX * config.N_TEAMS * 1 / 5))
        ls_wr = int(config.ROSTER_N_WR * config.N_TEAMS + (config.ROSTER_N_FLEX * config.N_TEAMS * 4 / 5))
        ls_te = config.ROSTER_N_TE * config.N_TEAMS

        players_to_keep = {'QB': ls_qb * 2.5,
                           'RB': ls_rb * 3,
                           'WR': ls_wr * 3,
                           'TE': ls_te * 2.5}

        normalized_df['keep'] = normalized_df['position'].map(players_to_keep)
        normalized_df = normalized_df[normalized_df['keep'] >= normalized_df['rank_pos']]
        normalized_df.drop(columns=['median_projection', 'rank_pos', 'keep'], inplace=True)

    if save_output:
        # save data
        today = pd.Timestamp.today().strftime("%Y%m%d")

        normalized_df.to_excel(f"{config.OUTPUT_DIR}/{today}_normalized.xlsx", index=False)
        normalization_factors_df.to_excel(f"{config.OUTPUT_DIR}/{today}_normalization_factors.xlsx", index=True)
        merged_df.to_excel(f"{config.OUTPUT_DIR}/{today}_merged.xlsx", index=False)
        flag_df.to_excel(f"{config.OUTPUT_DIR}/{today}_flags.xlsx", index=False)

    return normalized_df, normalization_factors_df, merged_df, flag_df

if __name__ == "__main__":
    normalized_df, normalization_factors_df, merged_df, flag_df = data_gen(trim_output=True, save_output=True)
