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

            logger.debug(f"Baseline '{value_threshold_name}' for {position}: "
                        f"{baseline_player}, {baseline_projection}")

            # Update baseline dictionary
            value_baseline_dict[position] = baseline_projection

        # Add baseline and value columns
        baseline_col = f'baseline_{value_threshold_name}'
        value_col = f'value_{value_threshold_name}'

        self.df[baseline_col] = self.df['position'].map(value_baseline_dict)
        self.df[value_col] = (self.df[projection_column] - self.df[baseline_col]).clip(lower=0).round(1)

        return self.df

    def add_vop_columns(self,
                        projection_column: str = 'median_projection',
                        n: int = 5) -> DataFrame:
        """
        Add VOP (Value Over Player) columns from 1 to N.

        VOP1 = Only for top player, value over 2nd best player
        VOP2 = For top 2 players, incremental value of player 2 over player 3
        VOP3 = For top 3 players, incremental value of player 3 over player 4
        VOPn = For top n players, incremental value of player n over player n+1

        Parameters
        ----------
        projection_column : str, optional
            Column to use for projections, default is 'median_projection'
        n : int, optional
            Maximum gap to calculate, default is 5

        Returns
        -------
        DataFrame
            DataFrame with new VOP columns
        """
        if projection_column not in self.df.columns:
            logger.error(f"Projection column '{projection_column}' not found in dataframe")
            return self.df

        # Sort by projection column within each position
        positions = self.df['position'].unique()
        vop_cols = []

        # Initialize VOP columns in the main dataframe
        for gap in range(1, n + 1):
            vop_col = f'vop{gap}'
            vop_cols.append(vop_col)
            self.df[vop_col] = 0.0

        # Calculate VOP values for each position
        for position in positions:
            # Filter players by position and sort by projection (descending)
            pos_df = self.df[self.df['position'] == position].sort_values(
                by=projection_column, ascending=False
            ).reset_index(drop=True)

            # Get list of player IDs in sorted order
            sorted_ids = pos_df['id'].tolist()

            # Calculate VOP values for each gap
            for gap in range(1, n + 1):
                # Need at least gap+1 players to calculate this VOP
                if len(pos_df) < gap + 1:
                    continue

                # Calculate the incremental value at this gap
                current_projection = pos_df.loc[gap - 1, projection_column]
                next_projection = pos_df.loc[gap, projection_column]
                vop_value = max(0, current_projection - next_projection)

                # Assign this VOP value to all players ranked 1 to gap
                vop_col = f'vop{gap}'
                for i in range(gap):
                    if i < len(sorted_ids):
                        player_id = sorted_ids[i]
                        self.df.loc[self.df['id'] == player_id, vop_col] = vop_value

        for col in vop_cols:
            # Round values
            self.df[col] = self.df[col].round(1)

        return self.df

    def calc_dynamic_value(self) -> DataFrame:
        """
        Calculate dynamic value based on the vona columns.

        Parameters
        ----------

        """
        # Calculate dynamic_value as the sum of all VONAs normalized to 100
        self.df['dynamic_value'] = self.df[[col for col in self.df.columns if col.startswith('vop')]].sum(axis=1)
        self.df['dynamic_value'] = (self.df['dynamic_value'] / self.df['dynamic_value'].max() * 100).round(1)

    def calc_static_value(self) -> None:
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

    def add_rank_cols(self) -> None:
        """
        Add rank & market share columns for each player

        Parameters
        ----------


        """
        self.df = self.df.sort_values(by=['static_value'], ascending=False).reset_index(drop=True)

        self.df['rank'] = self.df.static_value.rank(ascending=False, method='min')
        self.df['rank_pos'] = self.df.groupby('position')['median_projection'].rank(ascending=False, method='min')
        self.df['rank_pos_team'] = self.df.groupby(['team', 'position'])['median_projection'].rank(ascending=False, method='min')
        self.df['mkt_share'] = (self.df['median_projection'] / self.df.groupby(['team', 'position'])['median_projection'].transform('sum') * 100).round(1)

    def combine_static_and_dynamic_value(self, draft_mode: bool = True) -> None:
        """
        Combine static and dynamic values into a single column.
        """
        if draft_mode:
            self.df['dynamic_value'] = np.where(self.df['static_value'] == 0, 0, self.df['dynamic_value'])
            self.df['draft_value'] = np.where(self.df['dynamic_value'] > self.df['static_value'], (self.df['static_value'] + self.df['dynamic_value'] / 10), self.df['static_value'])
        else:
            self.df['draft_value'] = self.df['static_value']
        self.df['draft_value'] = (self.df['draft_value'] / self.df['draft_value'].max() * 100).round(1)

    def order_columns(self) -> None:
        """
        Reorder the columns in the dataframe for better readability.
        """
        id_cols = ['id', 'player', 'team', 'position']
        projection_cols = [col for col in self.df.columns if col.startswith(self.projection_column_prefix)]
        agg_projection_cols = [col for col in self.df.columns if col.endswith('_projection')]
        baseline_cols = [col for col in self.df.columns if col.startswith('baseline_')]
        value_cols = [col for col in self.df.columns if col.startswith('value_')]
        vop_cols = [col for col in self.df.columns if col.startswith('vop')]
        rank_cols = ['rank', 'rank_pos', 'rank_pos_team', 'mkt_share']

        all_cols = id_cols + rank_cols + ['draft_value', 'static_value', 'dynamic_value'] + agg_projection_cols + value_cols + vop_cols + projection_cols + baseline_cols
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


def value_players(df: DataFrame,
                  projection_column_prefix: str = 'projection_',
                  vopn: int = 5,
                  draft_mode: bool = True) -> DataFrame:
    """
    Calculate player values based on projections and thresholds.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing player projection data
    projection_column_prefix : str, optional
        Prefix used to identify projection columns, by default 'projection_'
    draft_mode : bool, optional
        Whether to adjust for scarcity during draft, by default True

    Returns
    -------
    DataFrame
        DataFrame with calculated values
    """
    calculator = ProjectionValueCalculator(df, projection_column_prefix)  # Initialize calculator

    # Add traditional static value columns
    for threshold_name in ['elite', 'last_starter', 'replacement']:
        calculator.add_value_column(
            projection_column='median_projection',
            value_threshold_name=threshold_name,
            value_threshold_dict={
                'QB': 6 if threshold_name == 'elite' else 10 if threshold_name == 'last_starter' else 17,
                'RB': 8 if threshold_name == 'elite' else 24 if threshold_name == 'last_starter' else 55,
                'WR': 15 if threshold_name == 'elite' else 36 if threshold_name == 'last_starter' else 60,
                'TE': 3 if threshold_name == 'elite' else 10 if threshold_name == 'last_starter' else 17,
            }
        )

    calculator.calc_static_value()  # Calculate static value
    calculator.add_rank_cols()  # Calculate Position Rank, Market Share
    calculator.add_vop_columns(n=vopn)  # Add VOPn columns
    calculator.calc_dynamic_value()  # Calculate dynamic value
    calculator.combine_static_and_dynamic_value(draft_mode=draft_mode)  # Combine static and dynamic values to get draft_value
    calculator.order_columns()  # Order columns for better readability

    return calculator.get_dataframe().sort_values(by=['draft_value'], ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    import config
    from run_data_gen import data_gen

    # Generate data
    normalized_df, _, _, _ = data_gen(trim_output=True, save_output=False)

    # Calculate player values
    result_df = value_players(normalized_df, projection_column_prefix=config.PROJECTION_COLUMN_PREFIX, vopn=5, draft_mode=True)
