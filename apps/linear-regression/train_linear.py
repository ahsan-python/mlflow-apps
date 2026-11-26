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

def evaluate_model(
    model: ElasticNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute standard regression evaluation metrics for train and test splits."""
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    return {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_preds))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
        "test_mae": float(mean_absolute_error(y_test, test_preds)),
        "train_r2": float(r2_score(y_train, train_preds)),
        "test_r2": float(r2_score(y_test, test_preds)),
    }


    #Saving the model as an artifact.
    sklearn.log_model(en, "model")

    run_id = mlflow.active_run().info.run_uuid
    print("Run with id %s finished" % run_id)
