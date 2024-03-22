import logging

import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting data from data_path
    """
    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        pass
    def get_data(self) -> pd.DataFrame:
        df = pd.read_csv("./data/olist_customers_dataset.csv")
        return df

@step
def ingest_df() -> pd.DataFrame:
    """
    Ingesting data from path

    Args:
        data_path: Path to the data

    Returns:
        pd.DataFrame: Ingested data
    """
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f'Error while ingesting data: {e}')
        raise e