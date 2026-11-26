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
def train(
    training_pandas_data: pd.DataFrame,
    test_pandas_data: pd.DataFrame,
    label_col: str,
    feat_cols: list[str],
    alpha: float = 0.5,
    l1_ratio: float = 0.5,
    training_data_path: str = "",
    test_data_path: str = "",
    random_state: int = 42,
) -> dict[str, Any]:
    """Train an ElasticNet regressor, log parameters/metrics/model to MLflow, and return run summary."""
    required_cols = [*feat_cols, label_col]
    validate_columns(training_pandas_data, required_cols, "training_data")
    validate_columns(test_pandas_data, required_cols, "test_data")

    X_train = training_pandas_data[feat_cols]
    y_train = training_pandas_data[label_col].to_numpy()

    X_test = test_pandas_data[feat_cols]
    y_test = test_pandas_data[label_col].to_numpy()

    # Ensure an active MLflow run exists whether called inside a parent run or standalone
    active_run = mlflow.active_run()
    run_context = mlflow.start_run() if active_run is None else active_run

    with run_context as run:
        logger.info("Starting ElasticNet training | run_id=%s", run.info.run_id)
        logger.info("Train path: %s | Test path: %s", training_data_path, test_data_path)

        # Log hyperparameters and dataset metadata
        params = {
            "model_type": "ElasticNet",
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "random_state": random_state,
            "label_col": label_col,
            "num_features": len(feat_cols),
            "features": ",".join(feat_cols),
            "training_data_path": training_data_path,
            "test_data_path": test_data_path,
        }
        mlflow.log_params(params)

        # Train model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        model.fit(X_train, y_train)

        # Evaluate and log metrics
        metrics = evaluate_model(
            model,
            X_train.to_numpy(),
            y_train,
            X_test.to_numpy(),
            y_test,
        )
        mlflow.log_metrics(metrics)

        for metric_name, value in metrics.items():
            logger.info("%s: %.4f", metric_name, value)

        # Infer model signature and log artifact
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
        )

        logger.info("Training finished successfully | run_id=%s", run.info.run_id)

        return {
            "run_id": run.info.run_id,
            "metrics": metrics,
            "model": model,
        }

   
