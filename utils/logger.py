"""
A simple logger for printing messages and ensuring the log directory exists.
"""

import os
from config import LOG_PATH

class Logger:
    """
    Logger class ensures the log directory exists and provides a method to log messages.
    """

    def __init__(self) -> None:
        """
        Initializes the Logger and creates the log directory if it does not exist.
        """
        os.makedirs(LOG_PATH, exist_ok=True)

    def log(self, message: str) -> None:
        """
        Logs a message by printing it to the console.

        Args:
            message (str): The message to log.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
