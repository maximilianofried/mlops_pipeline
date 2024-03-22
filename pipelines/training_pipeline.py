from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

"""
Trains a machine learning model on the provided data.

The pipeline ingests the data, cleans it, trains a model, and evaluates model performance.
It returns the trained model and model evaluation metrics.
"""
@pipeline(enable_cache=False)
def train_pipeline() -> None:
    """
    Training pipeline
    """
    df = ingest_df()
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2, rmse =  evaluate_model(model, X_test, y_test)
