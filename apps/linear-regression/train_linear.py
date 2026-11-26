# Copyright 2018 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging 
from typing import any

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def validate_columns(df: pd.DataFrame, required_cols: list[str], dataset_name: str) -> None:
    """Ensure all required feature and label columns exist in the DataFrame."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {dataset_name}: {missing}")
    testLabels = test_pandas_data[label_col].values
    testFeatures = test_pandas_data[feat_cols].values

    #We will use a linear Elastic Net model.
    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    # Here we train the model.
    en.fit(trainingFeatures, trainingLabels)

    # Calculating the scores of the model.
    test_rmse = mean_squared_error(testLabels, en.predict(testFeatures))**0.5
    r2_score_training = en.score(trainingFeatures, trainingLabels)
    r2_score_test = en.score(testFeatures, testLabels)
    print("Test RMSE:", test_rmse)
    print("Training set score:", r2_score_training)
    print("Test set score:", r2_score_test)

    #Logging the RMSE and r2 scores.
    mlflow.log_metric("Test RMSE", test_rmse)
    mlflow.log_metric("Train R2", r2_score_training)
    mlflow.log_metric("Test R2", r2_score_test)

    #Saving the model as an artifact.
    sklearn.log_model(en, "model")

    run_id = mlflow.active_run().info.run_uuid
    print("Run with id %s finished" % run_id)
