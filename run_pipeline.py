from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == '__main__':
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path='./data/olist_customers_dataset.csv')
"""
Trains a machine learning pipeline on the provided data path.

Args:
    data_path: Path to input data file.
"""
