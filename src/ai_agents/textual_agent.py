from .base_agent import BaseAgent
from src.data_processing.feature_extractor import FeatureExtractor
import torch

class TextualAgent(BaseAgent):
    """
    An AI agent specialized for processing text data and extracting textual features.
    """

    def __init__(self, name: str = "TextualAgent", device=None):
        """
        Initialize the TextualAgent with a name and an instance of the FeatureExtractor.

        Args:
            name (str): Name of the agent. Defaults to 'TextualAgent'.
            device (torch.device, optional): The device to use (e.g., 'cuda' or 'cpu'). Defaults to auto-detection.
        """
        super().__init__(name)
        # Set the device (use provided or default to auto-detection)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"TextualAgent initialized on device: {self.device}")

        # Initialize the feature extractor
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.visual_model.to(self.device)  # Move only the visual model to the device

    def process(self, text_data: str):
        """
        Process text data to extract relevant features.

        Args:
            text_data (str): The input text data to process.

        Returns:
            dict: Extracted textual features or an error message if processing fails.
        """
        if not text_data or not isinstance(text_data, str):
            self.log_error("Invalid text data provided.")
            return {"error": "Invalid text data. Please provide a non-empty string."}

        self.log_info(f"Processing text data: {text_data[:50]}...")

        try:
            # Extract features using FeatureExtractor
            features = self.feature_extractor.extract_from_text(text_data)
            self.log_info("Text feature extraction completed successfully.")
            return {
                "original_text": text_data,
                "extracted_features": features
            }
        except Exception as e:
            self.log_error(f"Error during text feature extraction: {e}")
            return {"error": str(e)}
