import requests
import json
import pandas as pd
import numpy as np

def pull_adp(year: int = 2025, teams: int = 10, position: str = 'all') -> pd.DataFrame:
    """
    Pulls ADP data from Fantasy Football Calculator API and returns it as a DataFrame.

    Parameters:
    - year (int): The year for which to pull ADP data.
    - teams (int): The number of teams in the league.
    - position (str): The position to filter by ('all', 'qb', 'rb', 'wr', 'te', 'k', 'dst').

    Returns:
    - pd.DataFrame: DataFrame containing the ADP data.
    """
    url = f"https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams={teams}&year={year}&position={position}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")

    data = response.json()['players']

    # Convert JSON data to DataFrame
    df = (pd.DataFrame(data)
          .drop(columns=['player_id', 'bye', 'adp_formatted', 'times_drafted'])
          .rename(columns={'name': 'player'})
    )

    return df

def _clean_player_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean player names by removing whitespace and converting to lowercase.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw_player_name column.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned_player_name column added.
    """
    if 'player' in df.columns:
        # Convert to string, strip whitespace, and convert to lowercase
        df['cleaned_player_name'] = df['player'].astype(str).str.strip().str.lower()
        df['cleaned_player_name'] = df['cleaned_player_name'].str.replace(r"[^\w\s'.-]", "", regex=True)

        # map player names using PLAYER_NAME_MAPPINGS only if cleaned_player_name is in PLAYER_NAME_MAPPINGS values
        df['cleaned_player_name'] = np.where(
            df['cleaned_player_name'].isin(config.PLAYER_NAME_MAPPINGS.keys()),
            df['cleaned_player_name'].map(config.PLAYER_NAME_MAPPINGS),
            df['cleaned_player_name'])

        df['player'] = df['cleaned_player_name']
        df.drop(columns=['cleaned_player_name'], inplace=True)

    return df

def _standardize_team_abbreviations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize team abbreviations using the mapping from config.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw_team_abbr column.

    Returns
    -------
    pd.DataFrame
        DataFrame with team_abbr_standardized column added.
    """
    if 'team' in df.columns:
        # Convert to string, strip whitespace, and convert to lowercase for mapping
        df['team_abbr_for_mapping'] = df['team'].astype(str).str.strip().str.lower()

        # Apply mapping
        df['team_abbr_standardized'] = df['team_abbr_for_mapping'].map(
            lambda x: config.TEAM_ABBR_MAPPINGS.get(x, np.nan)
        )

        # Drop temporary mapping column
        df.drop('team_abbr_for_mapping', axis=1, inplace=True)

        df['team'] = df['team_abbr_standardized']  # Update team column with standardized values
        df.drop(columns=['team_abbr_standardized'], inplace=True)

    return df

def _standardize_position_abbreviations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize position abbreviations using the mapping from config.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw_pos_abbr column.

    Returns
    -------
    pd.DataFrame
        DataFrame with position_abbr_standardized column added.
    """
    if 'position' in df.columns:
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
                        pos_str = primary_pos
                        break

            # Map to standard position
            standardized = config.POSITION_ABBR_MAPPINGS.get(pos_str.lower(), np.nan)

            return standardized

        # Apply standardization
        df['position_abbr_standardized'] = df['position'].apply(standardize_position)

        df['position'] = df['position_abbr_standardized']  # Update position column with standardized values
        df.drop(columns=['position_abbr_standardized'], inplace=True)

    return df

def _make_numeric_cols(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Convert specified columns to numeric, coercing errors to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to convert.
    cols : list
        List of column names to convert to numeric.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns converted to numeric.
    """
    if cols is None:
        cols = ['adp', 'high', 'low', 'stdev']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def _add_id_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a unique ID column to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to add the ID column to.

    Returns
    -------
    pd.DataFrame
        DataFrame with the ID column added.
    """
    df['player_id'] = (df['team'] + "_" +
                       df['position'] + "_" +  # df['rank_by_team_pos'] + "_" +
                       df['player'].str.replace(r'\s+', '', regex=True).str.replace(r'[^\w\s]', '',regex=True).str[:5])
    return df

def clean_adp_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the ADP DataFrame by applying various transformations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ADP data.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardized player names, team abbreviations, and position abbreviations.
    """
    df = _clean_player_names(df)
    df = _standardize_team_abbreviations(df)
    df = _standardize_position_abbreviations(df)
    df = _make_numeric_cols(df)
    df = _add_id_col(df)
    df = df[['player_id', 'player', 'team', 'position', 'adp', 'high', 'low', 'stdev']].copy()

    return df

def save_adp_data(df: pd.DataFrame) -> None:
    """
    Save ADP data to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ADP data.
    """
    import datetime
    today = datetime.date.today().strftime("%Y%m%d")
    output_path = config.OUTPUT_DIR / f"{today}_adp.csv"
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    import config

    df = clean_adp_df(pull_adp(year=config.SEASON_YEAR, teams=config.N_TEAMS, position='all'))
    save_adp_data(df)
    print(df.head(10))
    print(df.columns)