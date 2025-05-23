
import pandas as pd
import numpy as np
import logging
import os
import re
from typing import List, Dict, Tuple, Optional, Union, Any

from pandas import DataFrame, Series

# Set up logging
logger = logging.getLogger(__name__)


class DataMerger:
    def __init__(self, config, cleaned_projections: List[pd.DataFrame]):
        """
        Initialize the DataMerger with configuration settings.

        Parameters
        ----------
        config : module
            Configuration module containing mappings and settings.
        """
        self.config = config
        self.cleaned_projections = cleaned_projections
        self._setup_logging()


    def _setup_logging(self):
        """Set up logging for the DataCleaner."""
        if not logger.handlers:
            handler = logging.FileHandler(self.config.LOG_FILE)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, self.config.LOG_LEVEL))

    def _add_player_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a unique player ID to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to which the player ID will be added.

        Returns
        -------
        pd.DataFrame
            DataFrame with player_id column added.
        """

        # create new column to rank each player by projection_points_ppr desc grouped by team and position and convert to string for player_id
        df['rank_by_team_pos'] = df.groupby(['team_abbr_standardized', 'position_abbr_standardized'])[self.config.PROJECTION_COLUMN_PREFIX].rank(method='dense', ascending=False).astype(str)

        # Create a unique player ID based on cleaned_player_name and team_abbr_standardized, e.g. 'WAS_QB_1_jay'
        df['player_id'] = (df['team_abbr_standardized'] + "_" + df['position_abbr_standardized'] + "_" +  # df['rank_by_team_pos'] + "_" +
                           df['cleaned_player_name'].str.replace(r'\s+','', regex=True).str.replace(r'[^\w\s]','', regex=True).str[:5])

        return df

    def _rename_projection_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename projection columns in the DataFrame to include the source expert name.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with projection columns to be renamed.

        Returns
        -------
        pd.DataFrame
            DataFrame with renamed projection columns.
        """

        # Rename projection columns to include the source expert name
        for col in df.columns:
            if col.startswith(self.config.PROJECTION_COLUMN_PREFIX):
                new_col_name = f"{col}_{df['source_expert'].iloc[0]}"
                df.rename(columns={col: new_col_name}, inplace=True)

        return df

    def _clean_merged_df_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the merged DataFrame by removing all but first columns with the same name, except for projection columns.

        Parameters
        ----------
        df : pd.DataFrame
            Merged DataFrame to be cleaned.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame.
        """

        cols_to_keep = ['player_id', 'cleaned_player_name', 'team_abbr_standardized', 'position_abbr_standardized']
        col_flag_to_keep = self.config.PROJECTION_COLUMN_PREFIX
        cols_to_keep.extend([col for col in df.columns if col.startswith(col_flag_to_keep)])

        df = df[cols_to_keep]

        return df

    def merge_projections_on_player_id(self) -> DataFrame | tuple[Any, Any]:
        """
        Merge all cleaned projections on player_id and create a single DataFrame.

        Parameters
        ----------
        cleaned_projections : List[pd.DataFrame]
            List of cleaned projection DataFrames.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with unique player_id.
        """

        if not self.cleaned_projections:
            return pd.DataFrame()

        # Create initial DataFrame from the first projection
        merged_df = self.cleaned_projections[0].copy()

        # Add player_id to each cleaned projection DataFrame
        merged_df = self._add_player_id(merged_df)

        # Rename projection columns to include the source expert name
        merged_df = self._rename_projection_columns(merged_df)

        # Merge each subsequent projection DataFrame on 'player_id'
        for n, df in enumerate(self.cleaned_projections[1:], start=1):

            # Add player_id to the current DataFrame
            df = self._add_player_id(df)

            # Rename projection columns to include the source expert name
            df = self._rename_projection_columns(df)

            merged_df = merged_df.merge(df, on='player_id', how='outer', suffixes=('', f'_{n}'))

            # fill names, teams, positions in case of missing players
            merged_df['cleaned_player_name'].fillna(merged_df[f'cleaned_player_name_{n}'], inplace=True)
            merged_df['team_abbr_standardized'].fillna(merged_df[f'team_abbr_standardized_{n}'], inplace=True)
            merged_df['position_abbr_standardized'].fillna(merged_df[f'position_abbr_standardized_{n}'], inplace=True)

        # Sort by the sum of projection points columns to keep rows with highest projections
        projection_cols = [col for col in merged_df.columns if col.startswith(self.config.PROJECTION_COLUMN_PREFIX)]
        merged_df['avg_proj'] = merged_df[projection_cols].mean(axis=1)
        merged_df.sort_values('avg_proj', ascending=False, inplace=True)

        # Drop duplicates, keeping the row with the highest sum of projections
        merged_df.drop_duplicates(subset=['player_id'], keep='first', inplace=True)

        # Drop the sum_proj column
        merged_df.drop(columns=['avg_proj'], inplace=True)

        # Reset index
        merged_df.reset_index(drop=True, inplace=True)

        # Drop unnecessary columns
        merged_df = self._clean_merged_df_columns(merged_df)

        # Round projection columns to 1 decimal place
        for col in merged_df.columns:
            if col.startswith(self.config.PROJECTION_COLUMN_PREFIX):
                merged_df[col] = merged_df[col].round(1)

        # Generate Flag DF
        flag_df = merged_df.iloc[:200].copy()
        flag_df['flag'] = np.where(flag_df.isna().any(axis=1), True, False)
        flag_df = flag_df.loc[flag_df['flag'] == True]

        # Remove player_ids from merged df that are in flag df
        merged_df = merged_df[~merged_df['player_id'].isin(flag_df['player_id'])]
        merged_df.reset_index(drop=True, inplace=True)

        return merged_df, flag_df


if __name__ == "__main__":
    import config
    from data_loader import DataLoader
    from data_cleaner import DataCleaner

    # Prep for example usage
    loader = DataLoader(
        season_year=config.SEASON_YEAR,
        historical_year=config.HISTORICAL_YEAR,
        projections_dir=str(config.PROJECTIONS_DIR),
        historical_dir=str(config.HISTORICAL_DIR)
    )
    projection_dfs, source_experts = loader.load_projections()
    historical_df = loader.load_historical()
    cleaner = DataCleaner(config)
    cleaned_projections = cleaner.clean_projection_dataframes(projection_dfs, source_experts)

    # Example usage
    merger = DataMerger(config, cleaned_projections)
    merged_df, flag_df = merger.merge_projections_on_player_id()
