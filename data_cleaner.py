
"""
DataCleaner class outline:
Fantasy Football Projection Aggregator & Draft Tool - Data Cleaner

* __init__: Initialize the DataCleaner with configuration settings.
* clean_projection_dataframes: Clean and standardize a list of projection DataFrames.
* clean_historical_dataframe: Clean and standardize the historical DataFrame.
* _clean_dataframe: Clean and standardize a single DataFrame.
* _standardize_column_names: Standardize column names using the mapping from config.
* _clean_player_names: Clean player names by removing whitespace & special characters and converting to lowercase.
* _standardize_team_abbreviations: Standardize team abbreviations using the mapping from config.
* _standardize_position_abbreviations: Standardize position abbreviations using the mapping from config.
* _clean_data_types: Clean data types for numeric columns.
* _filter_rows: Filter rows based on required fields.
* _select_and_order_columns: Select and order columns for the final DataFrame.

This module is responsible for cleaning and standardizing data from various projection sources.
It handles column name standardization, team and position abbreviation standardization,
and basic data type cleaning.

Expected input objects:
* projection_dfs: List of raw projection DataFrames from various sources.
* source_experts: List of source expert identifiers corresponding to each DataFrame.
* historical_df: Raw historical DataFrame.

Expected output objects:
* cleaned_projections: List of cleaned and standardized projection DataFrames.
* cleaned_historical: Cleaned and standardized historical DataFrame.
"""

import pandas as pd
import numpy as np
import logging
import os
import re
from typing import List, Dict, Tuple, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)


class DataCleaner:
    def __init__(self, config):
        """
        Initialize the DataCleaner with configuration settings.

        Parameters
        ----------
        config : module
            Configuration module containing mappings and settings.
        """
        self.config = config
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging for the DataCleaner."""
        if not logger.handlers:
            handler = logging.FileHandler(self.config.LOG_FILE)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, self.config.LOG_LEVEL))

    def clean_projection_dataframes(self, projection_dfs: List[pd.DataFrame], source_experts: List[str]) -> List[pd.DataFrame]:
        """
        Clean and standardize a list of projection DataFrames.

        Parameters
        ----------
        projection_dfs : List[pd.DataFrame]
            List of raw projection DataFrames from various sources.
        source_experts : List[str]
            List of source expert identifiers corresponding to each DataFrame.

        Returns
        -------
        List[pd.DataFrame]
            List of cleaned and standardized projection DataFrames.
        """
        if len(projection_dfs) != len(source_experts):
            raise ValueError("Length of projection_dfs and source_experts must match")

        cleaned_dfs = []
        for i, df in enumerate(projection_dfs):
            source_expert = source_experts[i]
            logger.info(f"Cleaning data from source: {source_expert}")
            
            try:
                cleaned_df = self._clean_dataframe(df, source_expert)
                if cleaned_df is not None and not cleaned_df.empty:
                    cleaned_dfs.append(cleaned_df)
                    logger.info(f"Successfully cleaned data from {source_expert}")
            except Exception as e:
                logger.error(f"Error cleaning data from {source_expert}: {str(e)}")
                
        return cleaned_dfs

    def clean_historical_dataframe(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the historical DataFrame.

        Parameters
        ----------
        historical_df : pd.DataFrame
            Raw historical DataFrame.

        Returns
        -------
        pd.DataFrame
            Cleaned and standardized historical DataFrame.
        """
        if historical_df is None or historical_df.empty:
            logger.warning("Historical DataFrame is None or empty")
            return pd.DataFrame()
            
        logger.info("Cleaning historical data")
        
        # Add source_expert column for consistency
        historical_df['source_expert'] = 'historical'
        
        try:
            cleaned_df = self._clean_dataframe(historical_df, 'historical')
            if cleaned_df is not None and not cleaned_df.empty:
                logger.info("Successfully cleaned historical data")
                return cleaned_df
            else:
                logger.warning("No valid data remains after cleaning historical data")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error cleaning historical data: {str(e)}")
            return pd.DataFrame()

    def _clean_dataframe(self, df: pd.DataFrame, source_expert: str) -> Optional[pd.DataFrame]:
        """
        Clean and standardize a single DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame to clean.
        source_expert : str
            Identifier for the source expert.

        Returns
        -------
        Optional[pd.DataFrame]
            Cleaned DataFrame or None if critical data is missing.
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # 1. Standardize column names
        df = self._standardize_column_names(df, source_expert)

        if df is None:  # Likely no projection points column found. Check log.
            return None
            
        # 2. Clean player names
        df = self._clean_player_names(df, source_expert)
        
        # 3. Standardize team abbreviations
        df = self._standardize_team_abbreviations(df, source_expert)
        
        # 4. Standardize position abbreviations
        df = self._standardize_position_abbreviations(df, source_expert)
        
        # 5. Clean data types
        df = self._clean_data_types(df, source_expert)
        
        # 6. Filter rows
        df = self._filter_rows(df, source_expert)
        if df is None or df.empty:
            logger.warning(f"No valid data remains after filtering for {source_expert}")
            return None
            
        # 7. Select and order columns
        df = self._select_and_order_columns(df)
        
        return df

    def _standardize_column_names(self, df: pd.DataFrame, source_expert: str) -> Optional[pd.DataFrame]:
        """
        Standardize column names using the mapping from config.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with original column names.
        source_expert : str
            Identifier for the source expert.

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame with standardized column names or None if critical columns are missing.
        """
        # Convert all column names to lowercase for case-insensitive matching
        df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
        
        # Check for player name column
        player_name_found = False
        for col in df.columns:
            if isinstance(col, str) and col in self.config.COLUMN_NAME_MAPPINGS and self.config.COLUMN_NAME_MAPPINGS[col] == 'player_name':
                player_name_found = True
                df['raw_player_name'] = df[col]
                break
        
        if not player_name_found:
            logger.error(f"No player name column found in {source_expert}. Skipping file.")
            return None
            
        # Check for projection points column
        projection_points_found = False
        for col in df.columns:
            if isinstance(col, str) and col in self.config.COLUMN_NAME_MAPPINGS and self.config.COLUMN_NAME_MAPPINGS[col] == self.config.PROJECTION_COLUMN_PREFIX:
                projection_points_found = True
                df[self.config.PROJECTION_COLUMN_PREFIX] = df[col]
                break
                
        if not projection_points_found:
            logger.error(f"No projection points column found in {source_expert}. Skipping file.")
            return None
            
        # Map other columns
        for col in df.columns:
            if isinstance(col, str) and col in self.config.COLUMN_NAME_MAPPINGS:
                standard_name = self.config.COLUMN_NAME_MAPPINGS[col]
                
                # Handle team column
                if standard_name == 'team_abbr':
                    df['raw_team_abbr'] = df[col]
                
                # Handle position column
                elif standard_name == 'position_abbr':
                    df['raw_pos_abbr'] = df[col]
                
                # Handle rank column
                elif standard_name == 'rank':
                    df['rank'] = df[col]
                
                # Handle position rank column
                elif standard_name == 'position_rank':
                    df['position_rank'] = df[col]
                
                # Handle ADP columns
                elif standard_name in ['adp', 'adp_rank']:
                    df[standard_name] = df[col]
                
                # Handle optional metadata columns
                elif any(meta in standard_name for meta in self.config.DESIRED_OPTIONAL_METADATA):
                    df[f'metadata_{standard_name}'] = df[col]
                    
        logger.debug(f"Column standardization completed for {source_expert}")
        return df

    def _clean_player_names(self, df: pd.DataFrame, source_expert: str) -> pd.DataFrame:
        """
        Clean player names by removing whitespace and converting to lowercase.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with raw_player_name column.
        source_expert : str
            Identifier for the source expert.

        Returns
        -------
        pd.DataFrame
            DataFrame with cleaned_player_name column added.
        """
        if 'raw_player_name' in df.columns:
            # Convert to string, strip whitespace, and convert to lowercase
            df['cleaned_player_name'] = df['raw_player_name'].astype(str).str.strip().str.lower()
            df['cleaned_player_name'] = df['cleaned_player_name'].str.replace(r"[^\w\s'.-]", "", regex=True)

            # map player names using PLAYER_NAME_MAPPINGS only if cleaned_player_name is in PLAYER_NAME_MAPPINGS values
            df['cleaned_player_name'] = np.where(df['cleaned_player_name'].isin(self.config.PLAYER_NAME_MAPPINGS.keys()), df['cleaned_player_name'].map(self.config.PLAYER_NAME_MAPPINGS), df['cleaned_player_name'])

            logger.debug(f"Player names cleaned for {source_expert}")
        else:
            logger.warning(f"raw_player_name column not found in {source_expert}")
            df['cleaned_player_name'] = np.nan
            
        return df

    def _standardize_team_abbreviations(self, df: pd.DataFrame, source_expert: str) -> pd.DataFrame:
        """
        Standardize team abbreviations using the mapping from config.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with raw_team_abbr column.
        source_expert : str
            Identifier for the source expert.

        Returns
        -------
        pd.DataFrame
            DataFrame with team_abbr_standardized column added.
        """
        if 'raw_team_abbr' in df.columns:
            # Convert to string, strip whitespace, and convert to lowercase for mapping
            df['team_abbr_for_mapping'] = df['raw_team_abbr'].astype(str).str.strip().str.lower()
            
            # Apply mapping
            df['team_abbr_standardized'] = df['team_abbr_for_mapping'].map(
                lambda x: self.config.TEAM_ABBR_MAPPINGS.get(x, np.nan)
            )
            
            # Log unmapped teams
            unmapped_teams = df[df['team_abbr_standardized'].isna()]['raw_team_abbr'].unique()
            for team in unmapped_teams:
                if pd.notna(team) and team.lower() not in ['nan', 'none', '']:
                    logger.warning(f"Unmapped team abbreviation: '{team}' in {source_expert}")
            
            # Drop temporary mapping column
            df.drop('team_abbr_for_mapping', axis=1, inplace=True)
            
            logger.debug(f"Team abbreviations standardized for {source_expert}")
        else:
            logger.warning(f"raw_team_abbr column not found in {source_expert}")
            df['team_abbr_standardized'] = np.nan
            
        return df

    def _standardize_position_abbreviations(self, df: pd.DataFrame, source_expert: str) -> pd.DataFrame:
        """
        Standardize position abbreviations using the mapping from config.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with raw_pos_abbr column.
        source_expert : str
            Identifier for the source expert.

        Returns
        -------
        pd.DataFrame
            DataFrame with position_abbr_standardized column added.
        """
        if 'raw_pos_abbr' in df.columns:
            # Function to handle position standardization
            def standardize_position(pos):
                if pd.isna(pos):
                    return np.nan
                
                # Convert to string, strip whitespace, and convert to uppercase
                pos_str = str(pos).strip().upper()
                
                # Check if multiple positions are listed
                if any(delim in pos_str for delim in [',', '/', ';']):
                    # Split by common delimiters
                    for delim in [',', '/', ';']:
                        if delim in pos_str:
                            positions = [p.strip() for p in pos_str.split(delim)]
                            primary_pos = positions[0]
                            logger.warning(f"Multiple positions found: '{pos_str}', using primary: '{primary_pos}' in {source_expert}")
                            pos_str = primary_pos
                            break
                
                # Map to standard position
                standardized = self.config.POSITION_ABBR_MAPPINGS.get(pos_str.lower(), np.nan)
                
                if pd.isna(standardized) and pos_str.lower() not in ['nan', 'none', '']:
                    logger.warning(f"Unmapped position abbreviation: '{pos_str}' in {source_expert}")
                
                return standardized
            
            # Apply standardization
            df['position_abbr_standardized'] = df['raw_pos_abbr'].apply(standardize_position)
            
            logger.debug(f"Position abbreviations standardized for {source_expert}")
        else:
            logger.warning(f"raw_pos_abbr column not found in {source_expert}")
            df['position_abbr_standardized'] = np.nan
            
        return df

    def _clean_data_types(self, df: pd.DataFrame, source_expert: str) -> pd.DataFrame:
        """
        Clean data types for numeric columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns to clean.
        source_expert : str
            Identifier for the source expert.

        Returns
        -------
        pd.DataFrame
            DataFrame with cleaned data types.
        """
        # Clean projection_points_ppr
        if self.config.PROJECTION_COLUMN_PREFIX in df.columns:
            # Convert to string first to handle non-string types
            df[self.config.PROJECTION_COLUMN_PREFIX] = df[self.config.PROJECTION_COLUMN_PREFIX].astype(str)
            
            # Remove non-numeric characters except decimal point and minus sign
            df[self.config.PROJECTION_COLUMN_PREFIX] = df[self.config.PROJECTION_COLUMN_PREFIX].str.replace(r'[^\d.-]', '', regex=True)
            
            # Convert to numeric, errors become NaN
            df[self.config.PROJECTION_COLUMN_PREFIX] = pd.to_numeric(df[self.config.PROJECTION_COLUMN_PREFIX], errors='coerce')
            
            logger.debug(f"{self.config.PROJECTION_COLUMN_PREFIX} cleaned for {source_expert}")
        
        # Clean rank if present
        if 'rank' in df.columns:
            df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
            logger.debug(f"rank cleaned for {source_expert}")
            
        # Clean position_rank if present
        if 'position_rank' in df.columns:
            df['position_rank'] = pd.to_numeric(df['position_rank'], errors='coerce')
            logger.debug(f"position_rank cleaned for {source_expert}")
            
        # Clean ADP if present
        if 'adp' in df.columns:
            df['adp'] = pd.to_numeric(df['adp'], errors='coerce')
            logger.debug(f"adp cleaned for {source_expert}")
            
        if 'adp_rank' in df.columns:
            df['adp_rank'] = pd.to_numeric(df['adp_rank'], errors='coerce')
            logger.debug(f"adp_rank cleaned for {source_expert}")
            
        # Clean metadata columns
        for col in df.columns:
            if col.startswith('metadata_'):
                # If it's a numeric metadata (like bye_week or age)
                if any(meta in col for meta in ['bye_week', 'age']):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.debug(f"{col} cleaned for {source_expert}")
                    
        return df

    def _filter_rows(self, df: pd.DataFrame, source_expert: str) -> pd.DataFrame:
        """
        Filter rows based on required fields.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to filter.
        source_expert : str
            Identifier for the source expert.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame.
        """
        # Count initial rows
        initial_count = len(df)
        
        # Drop rows where cleaned_player_name is NaN or empty
        name_mask = df['cleaned_player_name'].isna() | (df['cleaned_player_name'] == '')
        name_drop_count = name_mask.sum()
        if name_drop_count > 0:
            logger.warning(f"Dropping {name_drop_count} rows with missing player names from {source_expert}")
            df = df[~name_mask]
            
        # Drop rows where projection_points_ppr is NaN
        proj_mask = df[self.config.PROJECTION_COLUMN_PREFIX].isna()
        proj_drop_count = proj_mask.sum()
        if proj_drop_count > 0:
            logger.warning(f"Dropping {proj_drop_count} rows with missing projections from {source_expert}")
            df = df[~proj_mask]
            
        # Log total rows dropped
        final_count = len(df)
        total_dropped = initial_count - final_count
        if total_dropped > 0:
            logger.info(f"Total rows dropped from {source_expert}: {total_dropped} ({total_dropped/initial_count:.1%})")
            
        return df

    def _select_and_order_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and order columns for the final DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with all columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with selected and ordered columns.
        """
        # Define core columns that should always be included
        core_columns = [
            'source_expert',
            'cleaned_player_name',
            'team_abbr_standardized',
            'position_abbr_standardized',
            self.config.PROJECTION_COLUMN_PREFIX
        ]
        
        # Add optional columns if they exist
        optional_columns = ['rank', 'position_rank', 'adp', 'adp_rank']
        existing_optional_columns = [col for col in optional_columns if col in df.columns]
        
        # Add metadata columns if they exist
        metadata_columns = [col for col in df.columns if col.startswith('metadata_')]
        
        # Combine all columns in the desired order
        all_columns = core_columns  # + existing_optional_columns + metadata_columns
        
        # Select only columns that exist in the DataFrame
        existing_columns = [col for col in all_columns if col in df.columns]
        
        return df[existing_columns]


# Example usage
if __name__ == "__main__":
    import config
    from data_loader import DataLoader
    
    # Prep for Example Usage
    loader = DataLoader(
        season_year=config.SEASON_YEAR,
        historical_year=config.HISTORICAL_YEAR,
        projections_dir=str(config.PROJECTIONS_DIR),
        historical_dir=str(config.HISTORICAL_DIR)
    )
    projection_dfs, source_experts = loader.load_projections()
    historical_df = loader.load_historical()

    # Example Usage
    cleaner = DataCleaner(config)
    cleaned_projections = cleaner.clean_projection_dataframes(projection_dfs, source_experts)
    cleaned_historical = cleaner.clean_historical_dataframe(historical_df)
    print(f"Cleaned {len(cleaned_projections)} projection datasets")
    if not cleaned_historical.empty:
        print(f"Cleaned historical data with {len(cleaned_historical)} rows")
