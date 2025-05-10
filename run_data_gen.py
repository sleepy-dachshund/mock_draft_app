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

def data_gen():

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

    # save data
    today = pd.Timestamp.today().strftime("%Y%m%d")

    normalized_df.to_excel(f"{config.OUTPUT_DIR}/{today}_normalized.xlsx", index=False)
    normalization_factors_df.to_excel(f"{config.OUTPUT_DIR}/{today}_normalization_factors.xlsx", index=True)
    merged_df.to_excel(f"{config.OUTPUT_DIR}/{today}_merged.xlsx", index=False)
    flag_df.to_excel(f"{config.OUTPUT_DIR}/{today}_flags.xlsx", index=False)

    return normalized_df, normalization_factors_df, merged_df, flag_df

if __name__ == "__main__":
    normalized_df, normalization_factors_df, merged_df, flag_df = data_gen()
