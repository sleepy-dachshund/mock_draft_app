import pandas as pd
import numpy as np
import os
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

    def __init__(self, df: DataFrame,
                 projection_column_prefix: str = 'projection_',
                 filled_roster_spots: list = None,
                 team_needs_adjustment: float = 1.0):
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
        self._calculate_aggregate_projections(filled_roster_spots=filled_roster_spots, team_needs_adjustment=team_needs_adjustment)

    def _calculate_aggregate_projections(self, filled_roster_spots: list = None, team_needs_adjustment: float = 1.0) -> None:
        """Calculate median, high, and low projections across all baseline projections."""
        self.df['median_projection'] = self.df[self.projection_cols].median(axis=1).round(1)
        self.df['high_projection'] = self.df[self.projection_cols].max(axis=1).round(1)
        self.df['low_projection'] = self.df[self.projection_cols].min(axis=1).round(1)

        # Create available_PPR for dynamic_value and draft_value
        self.df['available_pts'] = np.where(self.df['drafted'] >= 1, 0, self.df['median_projection'])

        # Adjust available_pts based on filled roster spots (lower available_pts for positions that have been filled already)
        if filled_roster_spots is not None:
            for position in filled_roster_spots:
                self.df['available_pts'] = np.where(self.df['position'] == position,
                                                    self.df['available_pts'] * team_needs_adjustment,
                                                    self.df['available_pts'])

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
                        projection_column: str = 'available_pts',  # note available_pts is used for live draft_value calcs since it zeroes out when a player has been drafted
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
            # Filter players by position and sort by (available) projection (descending)
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

    def calc_static_value(self,
                          weight_elite: float = 0.10,
                          weight_starter: float = 0.75,
                          weight_replacement: float = 0.15) -> None:
        """
        Calculate static value based on the specified value column.

        Parameters
        ----------
        value_col : str
            The name of the value column to use for calculation
        """
        # Combine value columns
        # self.df['static_value'] = self.df[[col for col in self.df.columns if col.startswith('value_')]].sum(axis=1)
        self.df['static_value'] = (self.df['value_elite'] * weight_elite +
                                   self.df['value_last_starter'] * weight_starter +
                                   self.df['value_replacement'] * weight_replacement)

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

    def combine_static_and_dynamic_value(self, draft_mode: bool = True, dynamic_multiplier: float = 0.05) -> None:
        """
        Combine static and dynamic values into a single column.
        """
        if draft_mode:

            # No dynamic value for players with no static value
            self.df['dynamic_value'] = np.where(self.df['static_value'] == 0, 0, self.df['dynamic_value'])

            # If draft value is greater than dynamic value, this is a flag, and we want to reflect that in draft_value
            self.df['draft_value'] = np.where(self.df['dynamic_value'] > self.df['static_value'], (self.df['static_value'] + self.df['dynamic_value'] * dynamic_multiplier), self.df['static_value'])

            # if drafting, only available players have value
            self.df['draft_value'] = np.where(self.df['drafted'] > 0, 0, self.df['draft_value'])

        else:

            # If not in draft_mode, just use the static value (create a standard cheat sheet
            self.df['draft_value'] = self.df['static_value']

        # Lastly, normalize draft_value
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

        all_cols = (id_cols
                    + ['draft_value', 'static_value', 'dynamic_value']
                    + ['drafted', 'available_pts']
                    + rank_cols
                    + agg_projection_cols
                    + value_cols
                    # + vop_cols + projection_cols
                    )

        for adp_col in ['adp', 'stdev', 'high', 'low']:
            if adp_col in self.df.columns:
                all_cols.append(adp_col)

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

def get_raw_df():
    import config
    from run_data_gen import data_gen
    raw_df, _, _, _ = data_gen(trim_output=True, save_output=True)
    raw_df['drafted'] = 0

    output_cols = ['id', 'player', 'team', 'position', 'drafted']
    proj_cols = [col for col in raw_df.columns if col.startswith(config.PROJECTION_COLUMN_PREFIX)]
    output_cols.extend(proj_cols)

    raw_df = raw_df[output_cols].copy()

    # quick sort by VORP here.
    raw_df['median_projection'] = raw_df[proj_cols].median(axis=1)
    projection_column = 'median_projection'

    # ============= START add values ============= #
    import config

    ls_qb = config.ROSTER_N_QB * config.N_TEAMS
    ls_rb = int(config.ROSTER_N_RB * config.N_TEAMS + (config.ROSTER_N_FLEX * config.N_TEAMS * 1 / 5))
    ls_wr = int(config.ROSTER_N_WR * config.N_TEAMS + (config.ROSTER_N_FLEX * config.N_TEAMS * 4 / 5))
    ls_te = config.ROSTER_N_TE * config.N_TEAMS

    value_threshold_name1 = 'vorp'
    value_threshold_dict1 = {'QB': ls_qb * 2, 'RB': ls_rb * 2, 'WR': ls_wr * 2, 'TE': ls_te * 2}

    value_threshold_name2 = 'vols'
    value_threshold_dict2 = {'QB': ls_qb, 'RB': ls_rb, 'WR': ls_wr, 'TE': ls_te}

    for col_name, col_dict in zip([value_threshold_name1, value_threshold_name2], [value_threshold_dict1, value_threshold_dict2]):
        # Initialize baseline dictionary
        value_baseline_dict = {pos: 0 for pos in col_dict.keys()}

        # Calculate baseline projections for each position
        for position, threshold in col_dict.items():
            position_df = raw_df[raw_df['position'] == position].copy()

            if len(position_df) < threshold:
                continue

            # Find threshold player
            baseline_row = position_df.iloc[threshold - 1]
            baseline_projection = baseline_row[projection_column]

            # Update baseline dictionary
            value_baseline_dict[position] = baseline_projection

        # Add baseline and value columns
        baseline_col = f'baseline_{col_name}'
        value_col = f'value_{col_name}'

        raw_df[baseline_col] = raw_df['position'].map(value_baseline_dict)
        raw_df[value_col] = (raw_df[projection_column] - raw_df[baseline_col]).clip(lower=0).round(1)
    # ============= END add values ============= #

    # Combine values and sort to help with draft board ordering
    raw_df['combined_value'] = raw_df[[col for col in raw_df.columns if col.startswith('value_')]].sum(axis=1)
    raw_df.sort_values(by='combined_value', ascending=False, inplace=True)
    raw_df.reset_index(drop=True, inplace=True)

    return raw_df[output_cols]

def value_players(df: DataFrame,
                  static_value_weights: Optional[Dict[str, float]] = {'elite': 0.10, 'last_starter': 0.75, 'replacement': 0.15},
                  projection_column_prefix: str = 'projection_',
                  vopn: int = 4,
                  dynamic_multiplier: float = 0.05,
                  filled_roster_spots: list = None,
                  team_needs: float = 1.0,
                  draft_mode: bool = True) -> DataFrame:
    """
    Calculate player values based on projections and thresholds.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing player projection data
    static_value_weights : Dict[str, float], optional
        Weights for averaging of static value columns, by default {'elite': 0.10, 'last_starter': 0.75, 'replacement': 0.15}
    projection_column_prefix : str, optional
        Prefix used to identify projection columns, by default 'projection_'
    vopn : int, optional
        Number of VOP columns to calculate, by default 5
    dynamic_multiplier : float, optional
        Multiplier for dynamic value, by default 0.05
    filled_roster_spots : list, optional
        List of filled roster spots, by default None
    team_needs : float, optional
        Team needs multiplier, by default 1.0
    draft_mode : bool, optional
        Whether to adjust for scarcity during draft, by default True

    Returns
    -------
    DataFrame
        DataFrame with calculated values
    """
    import config
    calculator = ProjectionValueCalculator(df, projection_column_prefix,
                                           filled_roster_spots=filled_roster_spots,
                                           team_needs_adjustment=team_needs)  # Initialize calculator

    ls_qb = config.ROSTER_N_QB * config.N_TEAMS
    ls_rb = int(config.ROSTER_N_RB * config.N_TEAMS + (config.ROSTER_N_FLEX * config.N_TEAMS * 1/5))
    ls_wr = int(config.ROSTER_N_WR * config.N_TEAMS + (config.ROSTER_N_FLEX * config.N_TEAMS * 4/5))
    ls_te = config.ROSTER_N_TE * config.N_TEAMS

    # Add traditional static value columns
    for threshold_name in ['elite', 'last_starter', 'replacement']:
        calculator.add_value_column(
            projection_column='median_projection',
            value_threshold_name=threshold_name,
            value_threshold_dict={
                'QB': 6 if threshold_name == 'elite' else ls_qb if threshold_name == 'last_starter' else 17,
                # todo: confirm elite thresholds, these are very subjective
                'RB': 8 if threshold_name == 'elite' else ls_rb if threshold_name == 'last_starter' else 55,
                'WR': 15 if threshold_name == 'elite' else ls_wr if threshold_name == 'last_starter' else 60,
                'TE': 3 if threshold_name == 'elite' else ls_te if threshold_name == 'last_starter' else 17,
            }
        )

    calculator.calc_static_value(weight_elite=static_value_weights['elite'],
                                 weight_starter=static_value_weights['last_starter'],
                                 weight_replacement=static_value_weights['replacement'])  # Calculate static value
    calculator.add_rank_cols()  # Calculate Position Rank, Market Share
    calculator.add_vop_columns(n=vopn, projection_column='available_pts')  # Add VOPn columns
    calculator.calc_dynamic_value()  # Calculate dynamic value
    calculator.combine_static_and_dynamic_value(draft_mode=draft_mode, dynamic_multiplier=dynamic_multiplier)  # Combine static and dynamic values to get draft_value
    # todo: add ADP column
    calculator.order_columns()  # Order columns for better readability

    # cache simple cheat sheet
    df_cache = (calculator
            .get_dataframe()
            .sort_values(by=['static_value', 'value_last_starter', 'value_replacement', 'median_projection'],
                         ascending=[False, False, False, False])
            .reset_index(drop=True))
    today_str = pd.Timestamp.now().strftime('%Y%m%d')
    file_directory = config.PROCESSED_DIR
    file_path = f'{today_str}_cheat_sheet.xlsx'
    file_to_check = file_directory / file_path
    if not os.path.exists(file_to_check):
        df_cache.to_excel(file_to_check, index=False)

    return (calculator
            .get_dataframe()
            .sort_values(by=['draft_value', 'static_value', 'value_elite', 'value_last_starter', 'value_replacement', 'median_projection'],
                         ascending=[False, False, False, False, False, False])
            .reset_index(drop=True))

if __name__ == "__main__":
    import config
    from run_data_gen import data_gen

    # Generate data
    raw_df = get_raw_df()
    input_df = raw_df.copy()

    # ========= PARAMS TO OPTIMIZE ========= #
    static_value_weights = {
        'elite': 1 / 3,
        'last_starter': 1 / 3,
        'replacement': 1 / 3,
    }
    vopn = 3
    dynamic_multiplier = 0.05
    # ====================================== #

    # Calculate player values
    result_df = value_players(input_df,
                              static_value_weights=static_value_weights,
                              projection_column_prefix=config.PROJECTION_COLUMN_PREFIX,
                              vopn=vopn,
                              dynamic_multiplier=dynamic_multiplier,
                              draft_mode=True)

    print(result_df.columns)
