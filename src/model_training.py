# Copyright 2025 Alireza Aghamohammadi

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module provides functionality for the entire model training pipeline.
"""

from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from src.logger import get_logger

logger = get_logger(__name__)


class ModelTraining:

    def __init__(self, config):
        self.model_training_config = config["model_training"]
        artifact_dir = Path(config["data_ingestion"]["artifact_dir"])
        self.processed_dir = artifact_dir / "processed"
        self.model_output_dir = artifact_dir / "models"
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load training and validation data from processed files.

        Returns
        -------
        tuple
            Tuple containing:
                train_data : pd.DataFrame
                    Training data
                val_data : pd.DataFrame
                    Validation data
        """
        self.train_path = self.processed_dir / "train.csv"
        self.val_path = self.processed_dir / "validation.csv"
        train_data = pd.read_csv(self.train_path)
        val_data = pd.read_csv(self.val_path)

        return train_data, val_data

    def build_model(self):
        """Build a Random Forest Regressor model using configuration parameters.

        Returns
        -------
        RandomForestRegressor
            Configured Random Forest Regressor model
        """
        n_estimators = self.model_training_config["n_estimators"]
        max_samples = self.model_training_config["max_samples"]
        n_jobs = self.model_training_config["n_jobs"]
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            oob_score=root_mean_squared_error,
            max_samples=max_samples,
            n_jobs=n_jobs,
        )
        return model

    def train(self, model, train_data):
        """Train the model using training data.

        Parameters
        ----------
        model : RandomForestRegressor
            Model to be trained
        train_data : pd.DataFrame
            Training data containing features and demand
        """
        X_train, y_train = train_data.drop(columns=["demand"]), train_data["demand"]
        model.fit(X_train, y_train)

    def evaluate(self, model, val_data):
        """Evaluate the model using validation data and log the results.

        Parameters
        ----------
        model : RandomForestRegressor
            Trained model to be evaluated
        val_data : pd.DataFrame
            Validation data containing features and demand
        """
        X_val, y_val = val_data.drop(columns=["demand"]), val_data["demand"]
        y_pred = model.predict(X_val)
        y_pred = [round(x) for x in y_pred]
        self.rmse = root_mean_squared_error(y_val, y_pred)
        self.oob_score = model.oob_score_

        logger.info(f"Out-of-Bag Score: {self.oob_score}")
        logger.info(f"Root Mean Squared Error for validation data: {self.rmse}")

    def save(self, model):
        """Save the trained model to disk using joblib.

        Parameters
        ----------
        model : RandomForestRegressor
            The trained model to be saved

        Notes
        -----
        The model is saved as a compressed .joblib file in the model_output_dir directory
        """
        logger.info("Start saving the model")
        self.model_output_path = self.model_output_dir / "rf.joblib"
        joblib.dump(model, self.model_output_path, compress=("lzma", 3))
        logger.info(
            f"Model saved successfully to {self.model_output_dir / 'rf.joblib'}"
        )

    def run(self):
        """Run the entire model training.

        Example
        -------
        >>> from src.model_training import ModelTraining
        >>> config = read_config("config/config.yaml")
        >>> model_training = ModelTraining(config)
        >>> model_training.run()
        """
        mlflow.set_experiment("tap30_ride_demand_mlops")

        with mlflow.start_run():
            logger.info("Model Training started")
            logger.info("MLflow started")
            mlflow.set_tag("model_type", "random_forest")

            train_data, val_data = self.load_data()
            mlflow.log_artifact(self.train_path, "datasets")
            mlflow.log_artifact(self.val_path, "datasets")

            model = self.build_model()
            self.train(model, train_data)
            self.evaluate(model, val_data)

            mlflow.log_metric("rmse", self.rmse)
            mlflow.log_metric("oob_score", self.oob_score)

            self.save(model)

            mlflow.log_artifact(self.model_output_path, "models")

            params = model.get_params()
            mlflow.log_params(params)

            logger.info("MLflow completed successfully")
            logger.info("Model Training completed successfully")
