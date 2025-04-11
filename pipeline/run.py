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

"""
orchestrates the end-to-end machine learning pipeline for ride demand prediction.

It sequentially executes the following stages:
    1. Data ingestion
    2. Data processing
    3. Model training
"""

from src.config_reader import read_config
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training import ModelTraining

if __name__ == "__main__":
    config_path = "config/config.yaml"
    config = read_config(config_path)

    data_ingestion = DataIngestion(config)
    data_ingestion.run()

    data_processing = DataProcessing(config)
    data_processing.run()

    model_training = ModelTraining(config)
    model_training.run()
