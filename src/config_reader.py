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
Provides functionality for reading and parsing YAML configuration files.

Import `read_config` from this module to read YAML configuration files that
follow the project's configuration structure.
"""


from pathlib import Path

import yaml

from logger import get_logger

logger = get_logger(__name__)


def read_config(config_path):
    """Read and parse a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Dictionary containing the parsed YAML configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the specified path.
    yaml.YAMLError
        If there is an error parsing the YAML file.

    Examples
    --------
    >>> config = read_config("config/config.yaml")
    >>> bucket_name = config["data_ingestion"]["bucket_name"]
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Config file not found at {config_path}")

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_file}")
        raise e
