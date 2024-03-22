import logging

import pandas as pd
from zenml import step
from model.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Cleaning data

    Args:
        df: Raw data

    Returns:
       X_train: Training data
       X_test: Testing data
       y_train: Training labels
       y_test: Testing labels
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        process_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(process_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaned successfully")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise e