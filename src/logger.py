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
Provides a pre-configured logger for consistent logging across the application.

Import `get_logger` from this module in your own modules to obtain a
logger instance that automatically writes timestamped messages (INFO level
and above) to a daily log file located in the 'logs/' directory.
"""

import logging
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / f"log_{datetime.now().strftime('%Y-%m-%d')}.log"


logging.basicConfig(
    filename=LOG_FILE,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def get_logger(name):
    """Gets a logger configured to write to the application's log file.

    This function utilizes the root logging configuration set up
    by this module.

    Parameters
    ----------
    name : str
        The name of the logger to retrieve. Typically __name__ for the
        calling module.

    Returns
    -------
    logging.Logger
        A logger instance configured according to the module's settings.

    Examples
    --------
    >>> from logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("This is an informational message.")
    """
    return logging.getLogger(name)
