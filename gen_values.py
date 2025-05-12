import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from pandas import DataFrame

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProjectionValueCalculator:
    """
    A class to calculate player values based on position-specific thresholds.

    This calculator takes projection data and applies value calculations by comparing
    player projections to baseline thresholds for each position.
    """

    def __init__(self, df: DataFrame, projection_column_prefix: str = 'projection_'):
        """
        Initialize the calculator with a dataframe containing player projections.

        Parameters
        ----------
        df : DataFrame
            DataFrame containing player projection data
        projection_column_prefix : str, optional
            Prefix used to identify projection columns, by default 'projection_'
        """
        self.df = df.copy()
        self.projection_column_prefix = projection_column_prefix

        # Identify projection columns
        self.projection_cols = [col for col in self.df.columns
                                if col.startswith(self.projection_column_prefix)]

        if not self.projection_cols:
            logger.warning(f"No projection columns found with prefix '{projection_column_prefix}'")

        # Calculate aggregate projections
        self._calculate_aggregate_projections()

    def _calculate_aggregate_projections(self) -> None:
        """Calculate median, high, and low projections across all baseline projections."""
        self.df['median_projection'] = self.df[self.projection_cols].median(axis=1).round(1)
        self.df['high_projection'] = self.df[self.projection_cols].max(axis=1).round(1)
        self.df['low_projection'] = self.df[self.projection_cols].min(axis=1).round(1)

        # Sort by median projection
        self.df.sort_values(by='median_projection', ascending=False, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def add_value_column(self,
                         projection_column: str,
                         value_threshold_name: str,
                         value_threshold_dict: Dict[str, int]) -> DataFrame:
        """
        Add a new value column to the dataframe based on specified thresholds.

        Parameters
        ----------
        projection_column : str
            Column name to use for projection values
        value_threshold_name : str
            Name to use for the new value column
        value_threshold_dict : Dict[str, int]
            Dictionary mapping positions to threshold ranks

        Returns
        -------
        DataFrame
            DataFrame with new value columns added
        """
        if projection_column not in self.df.columns:
            logger.error(f"Projection column '{projection_column}' not found in dataframe")
            return self.df

        # Initialize baseline dictionary
        value_baseline_dict = {pos: 0 for pos in value_threshold_dict.keys()}

        # Calculate baseline projections for each position
        for position, threshold in value_threshold_dict.items():
            position_df = self.df[self.df['position'] == position].copy()

            if len(position_df) < threshold:
                logger.warning(f"Not enough {position} players for threshold {threshold}")
                continue

            # Find threshold player
            baseline_row = position_df.iloc[threshold - 1]
            baseline_player = baseline_row['player']
            baseline_projection = baseline_row[projection_column]

            logger.info(f"Baseline '{value_threshold_name}' for {position}: "
                        f"{baseline_player}, {baseline_projection}")

            # Update baseline dictionary
            value_baseline_dict[position] = baseline_projection

        # Add baseline and value columns
        baseline_col = f'baseline_{value_threshold_name}'
        value_col = f'value_{value_threshold_name}'

        self.df[baseline_col] = self.df['position'].map(value_baseline_dict)
        self.df[value_col] = (self.df[projection_column] - self.df[baseline_col]).clip(lower=0).round(1)

        return self.df

    def calc_static_value(self, value_col: str) -> None:
        """
        Calculate static value based on the specified value column.

        Parameters
        ----------
        value_col : str
            The name of the value column to use for calculation
        """
        # Combine value columns
        self.df['static_value'] = self.df[[col for col in self.df.columns if col.startswith('value_')]].sum(axis=1)
        self.df['static_value'] = (self.df['static_value'] / self.df['static_value'].max() * 100).round(1)

    def order_columns(self) -> None:
        """
        Reorder the columns in the dataframe for better readability.
        """
        id_cols = ['id', 'player', 'team', 'position']
        projection_cols = [col for col in self.df.columns if col.startswith(self.projection_column_prefix)]
        agg_projection_cols = [col for col in self.df.columns if col.endswith('_projection')]
        baseline_cols = [col for col in self.df.columns if col.startswith('baseline_')]
        value_cols = [col for col in self.df.columns if col.startswith('value_')]

        all_cols = id_cols + ['static_value'] + agg_projection_cols + value_cols + projection_cols + baseline_cols
        self.df = self.df[all_cols]

    def get_dataframe(self) -> DataFrame:
        """
        Return the current state of the dataframe with all calculated values.

        Returns
        -------
        DataFrame
            The processed dataframe
        """
        return self.df


if __name__ == "__main__":
    import config
    from run_data_gen import data_gen

    # Generate data
    normalized_df, _, _, _ = data_gen(save_output=False)

    # Initialize calculator
    calculator = ProjectionValueCalculator(normalized_df, config.PROJECTION_COLUMN_PREFIX)

    for threshold_name in ['elite', 'classic', 'replacement']:
        # Add value column using median projections
        calculator.add_value_column(
            projection_column='median_projection',
            value_threshold_name=threshold_name,
            value_threshold_dict={
                'QB': 6 if threshold_name == 'elite' else 10 if threshold_name == 'classic' else 17,
                'RB': 8 if threshold_name == 'elite' else 24 if threshold_name == 'classic' else 55,
                'WR': 15 if threshold_name == 'elite' else 36 if threshold_name == 'classic' else 60,
                'TE': 3 if threshold_name == 'elite' else 10 if threshold_name == 'classic' else 17,
            }
        )

    # todo: implement this
    # for n in range(1,10):
    #     # Add VONA column
    #     calculator.add_vona_column(projection_column=f'projection_{n}',value_threshold_name=f'vona_{n}')

    # Get the resulting dataframe
    calculator.calc_static_value(value_col='static_value')
    calculator.order_columns()
    result_df = calculator.get_dataframe().sort_values(by=['static_value'], ascending=False).reset_index(drop=True)
