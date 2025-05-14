import pandas as pd
from pandas import DataFrame  # Explicit import

class IngestData:
    def __init__(self) -> None:
        self.data_path = None

    def get_data(self, data_path: str) -> DataFrame:  # Corrected type hint
        self.data_path = data_path
        df = pd.read_csv(self.data_path)
        return df
