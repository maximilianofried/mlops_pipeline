# Anticipating customer sentiment about a product prior to purchase

**Problem statement**: Given a customer's historical data, the objective is to forecast the review score for the upcoming order or purchase. We will utilize the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), which comprises information on 100,000 orders from 2016 to 2018 across various marketplaces in Brazil. The dataset encompasses diverse dimensions such as order status, pricing, payment, freight performance, customer location, product attributes, and customer reviews. Our goal is to predict customer satisfaction scores for orders based on features like order status, pricing, and payment. To accomplish this in a real-world scenario, we will leverage [ZenML](https://zenml.io/) to construct a production-ready pipeline for predicting customer satisfaction scores for upcoming orders or purchases.

The aim of this repository is to showcase how [ZenML](https://github.com/zenml-io/zenml) enables businesses to develop and deploy machine learning pipelines through various means:

- Providing a framework and template for building custom solutions.
- Integrating seamlessly with tools like [MLflow](https://mlflow.org/) for deployment, tracking, and more.
- Facilitating the easy construction and deployment of machine learning pipelines.

## Python Requirements

```bash
pip install zenml["server"]
zenml up
```

For executing the run_deployment.py script, additional integrations need to be installed using ZenML:

```bash
zenml integration install mlflow -y
```

The project requires a ZenML stack with an MLflow experiment tracker and model deployer as components. Setting up a new stack with these components involves the following steps:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

Launch the MLflow backend UI with the experiment tracker URI:
```bash
mlflow ui --backend-store-uri "file:.."
```

## The Solution

Building a real-world workflow for predicting customer satisfaction scores for upcoming orders or purchases, which aids in better decision-making, necessitates more than just training the model once.

Instead, we develop an end-to-end pipeline for continuously predicting and deploying the machine learning model, alongside a data application that leverages the latest deployed model for business consumption.

This pipeline can be deployed to the cloud, scaled according to requirements, and ensures comprehensive tracking of parameters and data flowing through each running pipeline. It encompasses raw data input, feature engineering, model training, model parameters, evaluation results, and prediction outputs. ZenML simplifies the construction of such pipelines in a straightforward yet robust manner.

This project places special emphasis on the MLflow integration of ZenML. Specifically, we utilize MLflow tracking to monitor metrics and parameters, and MLflow deployment to deploy our model. Additionally, we utilize Streamlit to demonstrate how this model can be utilized in a real-world setting.

### Training Pipeline

Our standard training pipeline comprises several key steps:

- `ingest_data`: Ingests the data and generates a DataFrame.
- `clean_data`: Cleans the data and removes unwanted columns.
- `model_train`: Trains the model and saves it using MLflow autologging.
- `evaluation`: Evaluates the model and saves the metrics, leveraging MLflow autologging, into the artifact store.

### Deployment Pipeline

We have another pipeline, deployment_pipeline.py, that extends the training pipeline and implements a continuous deployment workflow. It ingests and processes input data, trains a model, and then deploys the prediction server that serves the model if it meets our evaluation criteria. The criteria chosen is a configurable threshold on the Mean Squared Error (MSE) of the training. The initial four steps of the pipeline remain the same as above, but we've added the following additional steps:

- `deployment_trigger`: Checks whether the newly trained model meets the deployment criteria.
- `model_deployer`: Deploys the model as a service using MLflow (if deployment criteria are met).

In the deployment pipeline, ZenML's MLflow tracking integration is utilized for logging hyperparameter values, the trained model itself, and model evaluation metrics as MLflow experiment tracking artifacts into the local MLflow backend. This pipeline also initiates a local MLflow deployment server to serve the latest MLflow model if its accuracy surpasses a configured threshold.

The MLflow deployment server runs locally as a daemon process, persisting in the background after the example execution completes. Upon running a new pipeline that produces a model meeting the accuracy threshold validation, the pipeline automatically updates the currently running MLflow deployment server to serve the new model instead of the old one.

To wrap it up, we deploy a Streamlit application that asynchronously consumes the latest model service from the pipeline logic. Achieving this with ZenML within the Streamlit code is straightforward:

```python
service = prediction_service_loader(
   pipeline_name="continuous_deployment_pipeline",
   pipeline_step_name="mlflow_model_deployer_step",
   running=False,
)
...
service.predict(...)  # Predict on incoming data from the application
```

While this ZenML Project trains and deploys a model locally, other ZenML integrations such as the Seldon deployer can also be employed similarly to deploy the model in a more production-oriented setting (e.g., on a Kubernetes cluster). We opt for MLflow here for the convenience of its local deployment.

## Exploring the code

You can run two pipelines as follows:

- Training pipeline:

```bash
python run_pipeline.py
```

- Continuous deployment pipeline:

```bash
python run_deployment.py
```

## Demo Streamlit App

 To run this Streamlit app on your local system, execute the following command:

```bash
streamlit run streamlit_app.py
```
