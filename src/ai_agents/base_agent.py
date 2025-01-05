import logging
from typing import Any
import torch

class BaseAgent:
    """
    A robust base class for AI agents, providing foundational methods and structure 
    for agents that process data and interact with various systems.
    """

    def __init__(self, name: str):
        """
        Initialize the base agent with a name.

        Args:
            name (str): Name of the agent.
        """
        self.name = name
        self.logger = logging.getLogger(self.name)
        self._initialize_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def _initialize_logger(self):
        """
        Initialize a logger for the agent to handle debugging and error reporting.
        """
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def process(self, input_data: Any) -> Any:
        """
        Abstract method to process input data. Must be implemented by subclasses.

        Args:
            input_data (Any): Input data for the agent to process.

        Returns:
            Any: Processed output data.
        """
        raise NotImplementedError("Subclasses must implement the process method.")

    def validate_input(self, input_data: Any) -> bool:
        """
        Validate the input data. Can be overridden by subclasses for specific validations.

        Args:
            input_data (Any): Input data to validate.

        Returns:
            bool: True if input is valid, False otherwise.
        """
        if input_data is None:
            self.logger.error("Input data is None.")
            return False
        self.logger.info("Input data validation passed.")
        return True

    def log_info(self, message: str):
        """
        Log an informational message.

        Args:
            message (str): Message to log.
        """
        self.logger.info(message)

    def log_error(self, message: str):
        """
        Log an error message.

        Args:
            message (str): Message to log.
        """
        self.logger.error(message)

    def execute(self, input_data: Any) -> Any:
        """
        Execute the processing pipeline, including validation and processing.

        Args:
            input_data (Any): Input data for processing.

        Returns:
            Any: Processed output data or None if validation fails.
        """
        self.log_info("Execution started.")
        if not self.validate_input(input_data):
            self.log_error("Execution aborted due to invalid input.")
            return None

        try:
            output_data = self.process(input_data)
            self.log_info("Execution completed successfully.")
            return output_data
        except Exception as e:
            self.log_error(f"An error occurred during processing: {e}")
            return None
