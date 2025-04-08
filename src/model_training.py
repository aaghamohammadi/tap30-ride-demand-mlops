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
        train_data = pd.read_csv(self.processed_dir / "train.csv")
        val_data = pd.read_csv(self.processed_dir / "validation.csv")

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
        rmse = root_mean_squared_error(y_val, y_pred)
        logger.info(f"Out-of-Bag Score: {model.oob_score_}")
        logger.info(f"Root Mean Squared Error for validation data: {rmse}")

    def run(self):
        """Run the entire model training.

        Example
        -------
        >>> from src.model_training import ModelTraining
        >>> config = read_config("config/config.yaml")
        >>> model_training = ModelTraining(config)
        >>> model_training.run()
        """
        logger.info("Model Training started")
        train_data, val_data = self.load_data()
        model = self.build_model()
        self.train(model, train_data)
        self.evaluate(model, val_data)
        logger.info("Model Training completed successfully")
