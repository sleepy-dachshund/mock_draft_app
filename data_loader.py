import pandas as pd
import os

import pandas as pd
import os

from pandas import DataFrame


class DataLoader:
    def __init__(self, season_year: str, historical_year: str, projections_dir: str, historical_dir: str) -> None:
        """
        Initialize the DataLoader with specified directories.

        Parameters
        ----------
        season_year : str
            The year for which the data is being loaded.
        historical_year : str
            The year for which historical data is being loaded.
        projections_dir : str
            The directory path where projection CSV files are located.
        historical_dir : str
            The directory path where historical CSV files are located.
        """
        self.season_year = season_year
        self.historical_year = historical_year
        self.projections_dir = projections_dir
        self.historical_dir = historical_dir

    def load_projections(self) -> tuple[list[DataFrame], list[str]]:
        """
        Load all projection CSV files from the projections directory.

        Returns
        -------
        list
            A list of pandas DataFrames, each containing data from a projection CSV file.
        """
        projections = []
        source_experts = []
        for filename in os.listdir(self.projections_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.projections_dir, filename)
                try:
                    df = pd.read_csv(file_path)

                    # Extract the expert name from the filename
                    expert_name = filename.split('.')[-2].split('_')[-1]
                    df['source_expert'] = expert_name

                    projections.append(df)
                    source_experts.append(expert_name)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        return projections, source_experts

    def load_historical(self) -> pd.DataFrame | None:
        """
        Load historical data for the specified season year.

        Returns
        -------
        pandas.DataFrame or None
            A DataFrame containing historical data if successful, otherwise None.
        """
        historical_file = os.path.join(self.historical_dir, f'{self.historical_year}_actuals.csv')
        try:
            historical_data = pd.read_csv(historical_file)
            return historical_data
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return None

if __name__ == '__main__':

    # Example usage
    loader = DataLoader(season_year='2025', historical_year='2024', projections_dir='data/input/2025/projections/', historical_dir='data/input/2025/historical/')
    projections, source_experts = loader.load_projections()
    historical = loader.load_historical()
