# Databricks notebook source
# MAGIC %md <i18n value="2b9aa913-4058-48fd-9d2a-cf99c3171893"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Lab: Adding Post-Processing Logic
# MAGIC
# MAGIC ## In this lab you:<br>
# MAGIC  - Import data and train a random forest model
# MAGIC  - Adding post-processing steps

# COMMAND ----------

# MAGIC %md <i18n value="9afdeb4a-5436-4775-b091-c20451ab9229"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Import Data

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Load data
housing = datasets.fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train.head()

# COMMAND ----------

# MAGIC %md <i18n value="b0f36204-cc7e-4bdd-a856-e8e78ba4673c"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ##Train Random Forest

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor

# Fit and evaluate a random forest model
rf_model = RandomForestRegressor(n_estimators=15, max_depth=5)

rf_model.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md <i18n value="17863f12-50a2-42d5-bb2f-47d7e647ab2e"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Create Pyfunc with Post-Processing Steps
# MAGIC In the demo notebook, we built a custom **`RFWithPreprocess`** model class that uses a **`preprocess_result(self, results)`** helper function to automatically pre-processes the raw input it receives before passing that input into the trained model's **`.predict()`** function.
# MAGIC
# MAGIC Now suppose we are not as interested in a numerical prediction as we are in a categorical label of **`Expensive`** and **`Not Expensive`** where the cut-off is above a price of $100. Instead of retraining an entirely new classification model, we can simply add on a post-processing step to the model trained above so it returns the predicted label instead of numerical price.
# MAGIC
# MAGIC Complete the following model class with **a new `postprocess_result(self, result)`** function such that passing in **`X_test`** into our model will return an **`Expensive`** or **`Not Expensive`** label for each row.

# COMMAND ----------

# TODO
import mlflow

# Define the model class
class RFWithPostprocess(mlflow.pyfunc.PythonModel):

    def __init__(self, trained_rf):
        self.rf = trained_rf
      
    def postprocess_result(self, results):
        """return post-processed results
        Expensive: predicted price > 100
        Not Expensive: predicted price <= 100"""
        # FILL_IN
        return 
    
    def predict(self, context, model_input):
        # FILL_IN
        return 

# COMMAND ----------

# MAGIC %md <i18n value="25109107-4520-4146-9435-6841fd514c16"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC Create, save, and apply the model to **`X_test`**.

# COMMAND ----------

# Construct model
rf_postprocess_model = RFWithPostprocess(trained_rf=rf_model)

# log model
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model("rf_postprocess_model", python_model=rf_postprocess_model)

# Load the model in `python_function` format
model_path = f"runs:/{run.info.run_id}/rf_postprocess_model"
loaded_postprocess_model = mlflow.pyfunc.load_model(model_path)

# Apply the model
loaded_postprocess_model.predict(X_test)