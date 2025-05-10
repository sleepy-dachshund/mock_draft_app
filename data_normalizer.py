import pandas as pd
import numpy as np
import logging
import os
import re
from typing import List, Dict, Tuple, Optional, Union, Any
from pandas import DataFrame, Series

# Set up logging
logger = logging.getLogger(__name__)


class DataNormalizer:
    def __init__(self, config: Any, projections: pd.DataFrame):
        """
        Initialize the DataNormalizer.

        Parameters
        ----------
        config : module or object
            Configuration module/object. Expected to have attributes:
            - BENCHMARK_PLAYER_COUNTS: Dict[str, int] (e.g., {'QB': 12, 'RB': 24})
            - POSITION_COLUMN: str (column name for player positions, e.g., 'position_abbr_standardized')
            - PROJECTION_COLUMN_PREFIX: str (prefix for projection columns, e.g., 'projection_points_ppr')
            - LOG_FILE: str (for logging setup)
            - LOG_LEVEL: str (for logging setup)
        projections : pd.DataFrame
            DataFrame containing player projections. Must include the POSITION_COLUMN
            and one or more projection columns identified by PROJECTION_COLUMN_PREFIX.
        """
        self.config = config
        self.projections_df = projections  # Store original, operate on copies if needed
        self.benchmark_player_counts = self.config.BENCHMARK_PLAYER_COUNTS
        self.position_col = self.config.POSITION_COLUMN
        self.projection_prefix = self.config.PROJECTION_COLUMN_PREFIX
        self.avg_benchmark_sums = {}
        self.last_normalization_factors = {}
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging for the DataCleaner."""
        if not logger.handlers:  # Check if handlers are already configured
            try:
                log_file = self.config.LOG_FILE
                log_level_str = self.config.LOG_LEVEL.upper()
                log_level = getattr(logging, log_level_str, logging.INFO)

                handler = logging.FileHandler(log_file)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(log_level)
            except AttributeError as e:
                logger.error(f"Logging configuration attribute missing: {e}")
            except Exception as e:  # Catch other potential errors during logging setup
                logger.error(f"Error setting up logging: {e}")

    def _get_projection_columns(self) -> List[str]:
        """Identifies projection columns in the DataFrame."""
        return [col for col in self.projections_df.columns if col.startswith(self.projection_prefix)]

    def _calculate_projection_benchmark_sums(self, projection_cols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculates the sum of projected points for benchmark players for each
        projection source and position.
        """
        projection_benchmark_sums: Dict[str, Dict[str, float]] = {}
        for proj_col in projection_cols:
            projection_benchmark_sums[proj_col] = {}
            for position, benchmark_count in self.benchmark_player_counts.items():
                # Filter DataFrame for the current position
                position_df = self.projections_df[self.projections_df[self.position_col] == position]
                if position_df.empty or proj_col not in position_df.columns:
                    benchmark_sum = 0.0
                else:
                    benchmark_sum = position_df[proj_col].nlargest(benchmark_count).sum()

                projection_benchmark_sums[proj_col][position] = benchmark_sum
        return projection_benchmark_sums

    def _calculate_average_benchmark_sums_by_position(self, proj_benchmark_sums: Dict[str, Dict[str, float]]) -> Dict[
        str, float]:
        """
        Calculates the average of benchmark sums for each position across all
        projection sources.
        """
        position_aggregates: Dict[str, List[float]] = {}
        for proj_source_sums in proj_benchmark_sums.values():
            for position, benchmark_sum in proj_source_sums.items():
                if position not in position_aggregates:
                    position_aggregates[position] = []
                position_aggregates[position].append(benchmark_sum)

        average_benchmark_sums: Dict[str, float] = {}
        for position, sums_list in position_aggregates.items():
            if sums_list:
                average_benchmark_sums[position] = sum(sums_list) / len(sums_list)
            else:
                average_benchmark_sums[position] = 0.0  # Should not happen if proj_benchmark_sums is populated
        return average_benchmark_sums

    def _calculate_normalization_factors(self,
                                         proj_benchmark_sums: Dict[str, Dict[str, float]],
                                         avg_benchmark_sums: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Calculates normalization factors for each projection source and position.
        Factor = target_average_sum / actual_projection_sum.
        """
        normalization_factors: Dict[str, Dict[str, float]] = {}
        for proj_col, position_sums in proj_benchmark_sums.items():
            normalization_factors[proj_col] = {}
            for position, actual_sum in position_sums.items():
                target_sum = avg_benchmark_sums.get(position)
                if target_sum is not None:
                    if actual_sum != 0:
                        factor = target_sum / actual_sum
                    elif target_sum == 0:  # actual_sum is 0 and target_sum is 0
                        factor = 1.0  # No change needed
                    else:  # actual_sum is 0 but target_sum is > 0
                        # Cannot scale 0 to a positive sum via multiplication if underlying values are 0.
                        # Default to 1.0 to not alter these zero projections.
                        factor = 1.0
                        logger.warning(
                            f"Actual sum for {proj_col}, position {position} is 0, "
                            f"but target average sum is {target_sum}. Factor set to 1.0."
                        )
                else:  # Position not in average_benchmark_sums (shouldn't occur if logic is sound)
                    factor = 1.0
                    logger.warning(
                        f"No average benchmark sum for position {position}. Factor set to 1.0 for {proj_col}.")
                normalization_factors[proj_col][position] = factor
        return normalization_factors

    def _apply_normalization(self,
                             df_to_normalize: pd.DataFrame,
                             projection_cols: List[str],
                             normalization_factors: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Applies normalization factors to the projection data.
        """
        # Iterate through relevant positions present in normalization_factors
        # This ensures we only try to normalize positions for which factors were calculated.
        # All positions from benchmark_player_counts should be in normalization_factors keys (even if nested dict is empty).

        positions_to_normalize = set()
        for proj_factors in normalization_factors.values():
            positions_to_normalize.update(proj_factors.keys())

        for proj_col in projection_cols:
            for position in positions_to_normalize:
                factor = normalization_factors.get(proj_col, {}).get(position, 1.0)
                if factor != 1.0:  # Apply only if factor is not trivial
                    position_mask = df_to_normalize[self.position_col] == position
                    df_to_normalize.loc[position_mask, proj_col] = \
                        df_to_normalize.loc[position_mask, proj_col] * factor
        return df_to_normalize

    def get_normalization_factors(self) -> pd.DataFrame:
        """
        Returns a DataFrame of normalization factors with projection column names as index,
        position abbreviations as columns, and normalization factors as values.
        """
        factors_dict = getattr(self, 'last_normalization_factors', {})
        return pd.DataFrame.from_dict(factors_dict, orient='index')

    def normalize_projection_data(self) -> pd.DataFrame:
        """
        Normalizes projection data based on average benchmark sums across projection sources.

        The process:
        1. Identify projection columns.
        2. For each projection source & position, sum points of benchmark players.
        3. For each position, average these sums across sources (this is the target sum).
        4. For each source & position, find factor = target_sum / actual_sum.
        5. Apply factors to individual player projections in a copy of the input DataFrame.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with normalized projection values.
        """
        logger.info("Starting projection data normalization.")

        projection_cols = self._get_projection_columns()
        if not projection_cols:
            logger.warning(
                f"No projection columns found with prefix '{self.projection_prefix}'. Returning original DataFrame.")
            return self.projections_df.copy()  # Return a copy of the original
        logger.info(f"Found projection columns: {projection_cols}")

        proj_benchmark_sums = self._calculate_projection_benchmark_sums(projection_cols)
        logger.debug(f"Calculated projection benchmark sums: {proj_benchmark_sums}")

        if not proj_benchmark_sums or all(not v for v in proj_benchmark_sums.values()):  # Check if truly empty
            logger.warning("No benchmark sums could be calculated. Returning original DataFrame.")
            return self.projections_df.copy()

        self.avg_benchmark_sums = self._calculate_average_benchmark_sums_by_position(proj_benchmark_sums)
        logger.debug(f"Calculated average benchmark sums by position: {self.avg_benchmark_sums}")

        if not self.avg_benchmark_sums:
            logger.warning("No average benchmark sums could be calculated. Normalization factors will default to 1.0.")
            # Factors will become 1.0 due to logic in _calculate_normalization_factors if target_sum is None

        normalization_factors = self._calculate_normalization_factors(proj_benchmark_sums, self.avg_benchmark_sums)
        logger.debug(f"Calculated normalization factors: {normalization_factors}")

        normalized_df = self.projections_df.copy()  # Work on a copy of the input DataFrame
        normalized_df = self._apply_normalization(normalized_df, projection_cols, normalization_factors)

        # convert float cols in normalized_df to float with 1 decimal
        float_cols = normalized_df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            normalized_df[col] = normalized_df[col].round(1)

        normalized_df.rename(columns={'player_id': 'id',
                                      'cleaned_player_name': 'player',
                                      'position_abbr_standardized': 'position',
                                      'team_abbr_standardized': 'team'}, inplace=True)

        self.last_normalization_factors = normalization_factors

        logger.info("Projection data normalization complete.")
        return normalized_df


if __name__ == "__main__":
    import config
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    from data_merger import DataMerger

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
    cleaned_historical = cleaner.clean_historical_dataframe(historical_df)

    merger = DataMerger(config, cleaned_projections)
    merged_df, flag_df = merger.merge_projections_on_player_id()

    # Example usage
    normalizer = DataNormalizer(config, merged_df)
    normalized_df = normalizer.normalize_projection_data()
    normalization_factors_df = normalizer.get_normalization_factors()

    #
    # # Verification step
    # print("\nVerifying normalization (conceptual):")
    # for proj_col in normalizer._get_projection_columns():
    #     if proj_col in normalized_df.columns:
    #         print(f"\nVerification for projection source: {proj_col}")
    #         for pos, bench_count in normalizer.benchmark_player_counts.items():
    #             pos_df = normalized_df[normalized_df[normalizer.position_col] == pos]
    #             if not pos_df.empty:
    #                 norm_sum = pos_df[proj_col].nlargest(bench_count).sum()
    #                 target = normalizer.avg_benchmark_sums.get(pos, "N/A")
    #                 print(f"  Position {pos} (Top {bench_count}): Normalized Sum = {norm_sum:.2f} (Target Avg: {target})")
    #




