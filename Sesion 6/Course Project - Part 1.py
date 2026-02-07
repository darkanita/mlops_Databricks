# Databricks notebook source
# MAGIC %md
# MAGIC # Course Project
# MAGIC
# MAGIC ## Objective
# MAGIC The goal of this project is to apply everything we have learned in this course to build an end-to-end machine learning project.
# MAGIC
# MAGIC ## Problem statement
# MAGIC For the project, we will ask you to build an end-to-end ML project. We will be working with the dataset Home Credit Default Risk, this a problem of  binary Classification task: where we want to predict whether the person applying for a home credit will be able to repay their debt or not. Our model will have to predict a 1 indicating the client will have payment difficulties: he/she will have late payment of more than X days on at least one of the first Y installments of the loan in our sample, 0 in all other cases.
# MAGIC
# MAGIC We will use [Area Under the ROC Curve](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=es_419) as the evaluation metric, so our models will have to return the probabilities that a loan is not paid for each row.
# MAGIC
# MAGIC For that, you will need:
# MAGIC - Dataset:
# MAGIC   - [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/)
# MAGIC - Train a model on that dataset tracking your experiments
# MAGIC - Create a model training pipeline
# MAGIC - Deploy the model in batch, web service or streaming
# MAGIC - Monitor the performance of your model
# MAGIC - Follow the best practices
# MAGIC
# MAGIC ## Technologies
# MAGIC
# MAGIC - Cloud: Databricks
# MAGIC - Experiment tracking tools: MLFlow
# MAGIC - CI/CD: Azure DevOps
# MAGIC
# MAGIC ## The data
# MAGIC
# MAGIC - `application_train_aai.csv`: Training dataset you must use to train and find the best hyperparameters on your model.
# MAGIC
# MAGIC - `HomeCredit_columns_description.csv`: This file contains descriptions for the columns in train and test datasets.

# COMMAND ----------

import os
home_credit_df = spark.read.option("header", True).option("inferSchema", True).csv("file:/Workspace/Users/alopezmoreno@mngenvmcap358061.onmicrosoft.com/databricks-training/Sesion 6/dataset/application_train_aai.csv")
display(home_credit_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Feature Table
# MAGIC
# MAGIC Next, we can use the DataFrame **`home_credit_df`** to create a feature table using Feature Store.
# MAGIC
# MAGIC **In order to write our features out as a feature table we will perform the following steps:**
# MAGIC 1. Create a Database that will store any feature table. In our case let that be `home_credit_analysis`
# MAGIC 1. Write the Python functions to compute the features. The output of each function should be an Apache Spark DataFrame with a unique primary key. The primary key can consist of one or more columns.
# MAGIC 1. Create a feature table by instantiating a FeatureStoreClient and using create_table (Databricks Runtime 10.2 ML or above) or create_feature_table (Databricks Runtime 10.1 ML or below).
# MAGIC 1. Populate the feature table using write_table.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Defining a database to store feature tables.

# COMMAND ----------

DATABASE_NAME = "home_credit_analysis"
#setup database that will hold our Feature tables in Delta format.
spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Defining a feature engineering function that will return a Spark dataframe with a unique primary key. 
# MAGIC In our case it is the `SK_ID_CURR`.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The `home_credit_df` DataFrame is not clean, you must execute the next steps:
# MAGIC
# MAGIC - Preprocessing:
# MAGIC   - Feature selection
# MAGIC   - Outlier detection/handling
# MAGIC   - Feature Imputation/Normalization/Standardization
# MAGIC   - Categorical feature encoding
# MAGIC
# MAGIC ### Create `compute_features` Function
# MAGIC
# MAGIC Complete the function with your preprocessing defined.

# COMMAND ----------

import pyspark.pandas as ps
import numpy as np

def compute_features(spark_df):
    # https://spark.apache.org/docs/latest/api/python/migration_guide/koalas_to_pyspark.html?highlight=dataframe%20pandas_api
    # Convert to pyspark.pandas DataFrame
    ps_df = spark_df.pandas_api()
    
    
    # Your code
    
    return ps_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute Features
# MAGIC
# MAGIC Next, we can use our featurization function `compute_features` to create create a DataFrame of our features.

# COMMAND ----------

# Your Code

# COMMAND ----------

# MAGIC %md
# MAGIC ##3. Create the Feature Table
# MAGIC
# MAGIC Next, we can use the `feature_table` operation to register the DataFrame as a Feature Store table.
# MAGIC
# MAGIC In order to do this, we'll want the following details:
# MAGIC
# MAGIC 1. The `name` of the database and table where we want to store the feature table
# MAGIC 1. The `keys` for the table
# MAGIC 1. The `schema` of the table
# MAGIC 1. A `description` of the contents of the feature table
# MAGIC 1. `partition_columns`- Column(s) used to partition the feature table.
# MAGIC 1. `features_df`(optional) - Data to insert into this feature table. The schema of features_df will be used as the feature table schema.
# MAGIC
# MAGIC **Note:** 
# MAGIC 1. This creates our feature table, but we still need to write our values in the DataFrame to the table. 

# COMMAND ----------

#Our first step is to instantiate the feature store client using `FeatureStoreClient()`.
from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()

# COMMAND ----------

#  Your code

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Populate the feature table using write_table.

# COMMAND ----------

# Your Code