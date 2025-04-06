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
This module handles downloading raw demand data from a specified URL, splitting it into
train/validation/test sets, and saving the data as CSV files.
"""


import random
from pathlib import Path
from urllib.request import urlopen

from src.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:

    DEMAND_VALUE_WITHHELD = -1

    def __init__(self, config):
        self.data_ingestion_config = config["data_ingestion"]
        self.bucket_name = self.data_ingestion_config["bucket_name"]
        self.object_name = self.data_ingestion_config["object_name"]
        self.storage_path = self.data_ingestion_config["storage_path"]
        self.train_ratio = self.data_ingestion_config["train_ratio"]
        self.url = f"https://{self.bucket_name}.{self.storage_path}/{self.object_name}"

        artifact_dir = Path(self.data_ingestion_config["artifact_dir"])
        self.raw_dir = artifact_dir / "raw"

        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_raw_data(self):
        """Download raw demand data from the configured URL.

        Returns
        -------
        str
            Raw data as a string containing space-separated integers.

        Raises
        ------
        Exception
            If the download fails for any reason.
        """
        try:
            with urlopen(self.url) as response:
                raw_data = response.read().decode("utf-8")
            return raw_data
        except Exception as e:
            logger.error(f"Failed to download data from {self.url}: {str(e)}")
            raise

    def split_data(self, raw_data):
        """Split the raw data into train, validation, and test sets.

        Parameters
        ----------
        raw_data : str
            Raw data string containing space-separated integers.

        Returns
        -------
        tuple
            Contains three lists:
            - train_data: list
                Training data points as [time, row, col, demand]
            - val_data: list
                Validation data points as [time, row, col, demand]
            - test_data: list
                Test data points as [time, row, col, demand]
        """
        numbers = [int(x) for x in raw_data.strip().split()]

        total_periods, n_cols, n_rows = numbers[:3]
        data = numbers[3:]

        logger.info(f"Data format: {total_periods} periods, {n_rows}x{n_cols} grid")
        test_data = []
        train_val_data = []

        for t in range(total_periods):
            offset = t * n_rows * n_cols
            for row in range(n_rows):
                for col in range(n_cols):
                    demand = data[offset + row * n_cols + col]
                    if demand == self.DEMAND_VALUE_WITHHELD:
                        test_data.append([t, row, col, demand])
                    else:
                        train_val_data.append([t, row, col, demand])

        random.shuffle(train_val_data)

        train_size = int(len(train_val_data) * self.train_ratio)
        train_data = train_val_data[:train_size]
        val_data = train_val_data[train_size:]

        return train_data, val_data, test_data

    def save_to_csv_files(self, train_data, val_data, test_data):
        """Save the split data into CSV files.

        Parameters
        ----------
        train_data : list
            Training data points as [time, row, col, demand]
        val_data : list
            Validation data points as [time, row, col, demand]
        test_data : list
            Test data points as [time, row, col, demand]
        """
        header = "time,row,col,demand\n"
        data_files = [
            ("train", train_data),
            ("validation", val_data),
            ("test", test_data),
        ]

        for name, data in data_files:
            output_file = self.raw_dir / f"{name}.csv"
            with open(output_file, "w") as f:
                f.write(header)
                for row in data:
                    f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")
            logger.info(f"Saved {name} data with {len(data)} records to {output_file}")

        logger.info(f"Split summary:")
        logger.info(f"Train set: {len(train_data)} records")
        logger.info(f"Validation set: {len(val_data)} records")
        logger.info(f"Test set (demand=-1): {len(test_data)} records")

    def run(self):
        """Execute the complete data ingestion pipeline.

        This method orchestrates the entire data ingestion process:
        1. Downloads raw data
        2. Splits it into train/validation/test sets
        3. Saves the processed data as CSV files

        Examples
        --------
        >>> data_ingestion = DataIngestion(read_config("config/config.yaml"))
        >>> data_ingestion.run()
        """
        logger.info(f"Data Ingestion started for {self.url}")

        raw_data = self.download_raw_data()
        train_data, val_data, test_data = self.split_data(raw_data)
        self.save_to_csv_files(train_data, val_data, test_data)

        logger.info(f"Data Ingestion completed successfully")
