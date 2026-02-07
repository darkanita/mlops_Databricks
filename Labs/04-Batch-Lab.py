# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Lab: Deploying a Model in Batch
# MAGIC Deploying a model via batch is the preferred solution for most machine learning applications. In this lab, you will scale the deployment of a single-node model via Spark UDFs and MLflow's **`pyfunc`**. 
# MAGIC
# MAGIC
# MAGIC ## In this lab you:<br>
# MAGIC  - Develop and register an MLflow model
# MAGIC  - Deploy the model as a Spark UDF
# MAGIC  - Optimize the predictions for reading using Delta

# COMMAND ----------

import pandas as pd
from sklearn import datasets

# Load data
housing = datasets.fetch_california_housing()

# COMMAND ----------

# MAGIC %md <i18n value="4bceb653-c1e7-44c5-acb4-0cce746fbad2"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Develop and Register an MLflow Model
# MAGIC
# MAGIC In this exercise, you will build, log, and register an XGBoost model using Scikit-learn and MLflow.
# MAGIC
# MAGIC This model will predict the **`price`** variable using **`bathrooms`**, **`bedrooms`**, and **`number_of_reviews`** as features.

# COMMAND ----------

import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_squared_error

# Start run
with mlflow.start_run(run_name="xgboost_model") as run:
    X_train = pd.DataFrame(housing.data, columns=housing.feature_names)
    y_train = housing.target

    # Train model
    n_estimators = 10
    max_depth = 5
    regressor = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth)
    regressor.fit(X_train, y_train)

    # Evaluate model
    predictions = regressor.predict(X_train)
    rmse = mean_squared_error(predictions, y_train, squared=False)

    # Log params and metric
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("rmse", rmse)

    # Log model
    mlflow.xgboost.log_model(regressor, "xgboost-model")
    
# Register model
suffix = "amls4"
model_name = f"xgboost-model_{suffix}"
model_uri = f"runs:/{run.info.run_id}/xgboost-model"
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md <i18n value="922f29ae-e6a6-4e8a-8793-acbddfb2e22e"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Deploy Model as a Spark UDF
# MAGIC
# MAGIC Next, you will compute predictions for your model using a Spark UDF.

# COMMAND ----------

# TODO
# Create the prediction UDF
predict = #FILL_IN

# Compute the predictions
prediction_df = df.withColumn("prediction", #FILL_IN)
             

# View the results
display(prediction_df)