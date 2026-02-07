# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Real Time Deployment
# MAGIC
# MAGIC While real time deployment represents a smaller share of the deployment landscape, many of these deployments represent high value tasks.  This lesson surveys real-time deployment options ranging from proofs of concept to both custom and managed solutions.
# MAGIC
# MAGIC ## We will learn:<br>
# MAGIC  - Survey the landscape of real-time deployment options
# MAGIC  - Prototype a RESTful service using MLflow
# MAGIC  - Deploy registered models using MLflow Model Serving
# MAGIC  - Query an MLflow Model Serving endpoint for inference using individual records and batch requests
# MAGIC  
# MAGIC *You need <a href="https://docs.databricks.com/applications/mlflow/model-serving.html#requirements" target="_blank">cluster creation</a> permissions to create a model serving endpoint. The instructor will either demo this notebook or enable cluster creation permission for the students from the Admin console.*

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### The Why and How of Real Time Deployment
# MAGIC
# MAGIC Real time inference is...<br><br>
# MAGIC
# MAGIC * Generating predictions for a small number of records with fast results (e.g. results in milliseconds)
# MAGIC * The first question to ask when considering real time deployment is: do I need it?  
# MAGIC   - It represents a minority of machine learning inference use cases &mdash; it's necessary when features are only available at the time of serving
# MAGIC   - Is one of the more complicated ways of deploying models
# MAGIC   - That being said, domains where real time deployment is often needed are often of great business value.  
# MAGIC   
# MAGIC Domains needing real time deployment include...<br><br>
# MAGIC
# MAGIC  - Financial services (especially with fraud detection)
# MAGIC  - Mobile
# MAGIC  - Ad tech

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC There are a number of ways of deploying models...<br><br>
# MAGIC
# MAGIC * Many use REST
# MAGIC * For basic prototypes, MLflow can act as a development deployment server
# MAGIC   - The MLflow implementation is backed by the Python library Flask
# MAGIC   - *This is not intended to for production environments*
# MAGIC
# MAGIC In addition, Databricks offers a managed **MLflow Model Serving** solution. This solution allows you to host machine learning models from Model Registry as REST endpoints that are automatically updated based on the availability of model versions and their stages.
# MAGIC
# MAGIC For production RESTful deployment, there are two main options...<br><br>
# MAGIC
# MAGIC * A managed solution 
# MAGIC   - Azure ML
# MAGIC   - SageMaker (AWS)
# MAGIC   - VertexAI (GCP)
# MAGIC * A custom solution  
# MAGIC   - Involve deployments using a range of tools
# MAGIC   - Often using Docker or Kubernetes
# MAGIC * One of the crucial elements of deployment in containerization
# MAGIC   - Software is packaged and isolated with its own application, tools, and libraries
# MAGIC   - Containers are a more lightweight alternative to virtual machines
# MAGIC
# MAGIC Finally, embedded solutions are another way of deploying machine learning models, such as storing a model on IoT devices for inference.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Prototyping with MLflow
# MAGIC
# MAGIC MLflow offers <a href="https://www.mlflow.org/docs/latest/models.html#pyfunc-deployment" target="_blank">a Flask-backed deployment server for development purposes only.</a>
# MAGIC
# MAGIC Let's build a simple model below. This model will always predict 5.

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import pandas as pd

class TestModel(mlflow.pyfunc.PythonModel):
  
    def predict(self, context, input_df):
        return 5

model_run_name="pyfunc-model"

with mlflow.start_run() as run:
    model = TestModel()
    mlflow.pyfunc.log_model(artifact_path=model_run_name, python_model=model)
    model_uri = f"runs:/{run.info.run_id}/{model_run_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC There are a few ways to send requests to the development server for testing purpose:
# MAGIC * using **`click`** library 
# MAGIC * using MLflow Model Serving API
# MAGIC * through CLI using **`mlflow models serve`**
# MAGIC
# MAGIC In this lesson, we are going to demonstrate how to use both the **`click`** library and MLflow Model Serving API. 
# MAGIC
# MAGIC This is just to demonstrate how a basic development server works. This design pattern (which hosts a server on the driver of your Spark cluster) is not recommended for production.<br>
# MAGIC Models can be served in this way in other languages as well.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Method 2: MLflow Model Serving
# MAGIC Now, let's use MLflow Model Serving. 
# MAGIC
# MAGIC Step 1: We first need to register the model in MLflow Model Registry and load the model. At this step, we don't specify the model stage, so that the stage version would be **`None`**. 
# MAGIC
# MAGIC You can refer to the MLflow documentation <a href="https://www.mlflow.org/docs/latest/model-registry.html#api-workflow" target="_blank">here</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC Train a model.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_auc_score

suffix="amls4"
model_name = f"demo-model_{suffix}"

# Data prep
white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# Remove spaces from column names
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

data.dropna(inplace=True)
data.reset_index(drop=True,inplace=True)
    
train, test = train_test_split(data, random_state=123)
X_train = train.drop(["quality"], axis=1)
X_test = test.drop(["quality"], axis=1)
y_train = train.quality
y_test = test.quality

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

input_example = X_train.head(3)
signature = infer_signature(X_train, pd.DataFrame(y_train))

with mlflow.start_run(run_name="RF Model") as run:
    mlflow.sklearn.log_model(rf, 
                             "model", 
                             input_example=input_example, 
                             signature=signature, 
                             registered_model_name=model_name, 
                             extra_pip_requirements=["mlflow==1.*"]
                            )

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Step 2: Run Tests Against Registered Model in order to Promote To Staging

# COMMAND ----------

time.sleep(10) # to wait for registration to complete

model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Here, visit the MLflow Model Registry to enable Model Serving. 
# MAGIC

# COMMAND ----------

import mlflow 
# We need both a token for the API, which we can get from the notebook.
# Recall that we discuss the method below to retrieve tokens is not the best practice. We recommend you create your personal access token and save it in a secret scope. 
token = mlflow.utils.databricks_utils._get_command_context().apiToken().get()

# With the token, we can create our authorization header for our subsequent REST calls
headers = {"Authorization": f"Bearer {token}"}

# Next we need an endpoint at which to execute our request which we can get from the Notebook's context
api_url = mlflow.utils.databricks_utils.get_webapp_url()
print(api_url)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Enable the endpoint

# COMMAND ----------

import requests

url = f"{api_url}/api/2.0/mlflow/endpoints/enable"

r = requests.post(url, headers=headers, json={"registered_model_name": model_name})
assert r.status_code == 200, f"Expected an HTTP 200 response, received {r.status_code}"

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC It will take a couple of minutes for the endpoint and model to become ready.
# MAGIC
# MAGIC Define a **wait_for_endpoint()** and **wait_for_model()** function.

# COMMAND ----------

def wait_for_endpoint():
    import time
    while True:
        url = f"{api_url}/api/2.0/preview/mlflow/endpoints/get-status?registered_model_name={model_name}"
        response = requests.get(url, headers=headers)
        assert response.status_code == 200, f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"

        status = response.json().get("endpoint_status", {}).get("state", "UNKNOWN")
        if status == "ENDPOINT_STATE_READY": print(status); print("-"*80); return
        else: print(f"Endpoint not ready ({status}), waiting 10 seconds"); time.sleep(10) # Wait 10 seconds

# COMMAND ----------

def wait_for_version():
    import time
    while True:    
        url = f"{api_url}/api/2.0/preview/mlflow/endpoints/list-versions?registered_model_name={model_name}"
        response = requests.get(url, headers=headers)
        assert response.status_code == 200, f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"

        state = response.json().get("endpoint_versions")[0].get("state")
        if state == "VERSION_STATE_READY": print(state); print("-"*80); return
        else: print(f"Version not ready ({state}), waiting 10 seconds"); time.sleep(10) # Wait 10 seconds

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC Define the **`score_model()`** function.

# COMMAND ----------

def score_model(dataset: pd.DataFrame, timeout_sec=300):
    import time
    start = int(time.time())
    print(f"Scoring {model_name}")
    
    url = f"{api_url}/model/{model_name}/1/invocations"
    ds_dict = dataset.to_dict(orient="split")
    
    while True:
        response = requests.request(method="POST", headers=headers, url=url, json=ds_dict)
        elapsed = int(time.time()) - start
        
        if response.status_code == 200: return response.json()
        elif elapsed > timeout_sec: raise Exception(f"Endpoint was not ready after {timeout_sec} seconds")
        elif response.status_code == 503: 
            print("Temporarily unavailable, retr in 5")
            time.sleep(5)
        else: raise Exception(f"Request failed with status {response.status_code}, {response.text}")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC After the model serving cluster is in the **`ready`** state, you can now send requests to the REST endpoint.

# COMMAND ----------

wait_for_endpoint()
wait_for_version()

# Give the system just a couple
# extra seconds to transition
time.sleep(5)

# COMMAND ----------

score_model(X_test)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC You can also optionally transition the model to the **`Staging`** or **`Production`** stage, using <a href="https://www.mlflow.org/docs/latest/model-registry.html#transitioning-an-mlflow-models-stage" target="_blank">MLflow Model Registry</a>. 
# MAGIC
# MAGIC Sample code is below:
# MAGIC ```
# MAGIC client.transition_model_version_stage(
# MAGIC     name=model_name,
# MAGIC     version=model_version,
# MAGIC     stage="Staging"
# MAGIC )
# MAGIC ```