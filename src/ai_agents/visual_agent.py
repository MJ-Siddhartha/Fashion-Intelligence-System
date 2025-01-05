import requests
from PIL import Image
import io
import torch
from .base_agent import BaseAgent
from src.data_processing.feature_extractor import FeatureExtractor


class VisualAgent(BaseAgent):
    """
    An AI agent specialized for processing image data and extracting visual features.
    """

    def __init__(self, name: str = "VisualAgent", device: torch.device = None):
        """
        Initialize the VisualAgent with a name and an instance of the FeatureExtractor.

        Args:
            name (str): Name of the agent. Defaults to 'VisualAgent'.
            device (torch.device): Torch device (CPU or GPU). Defaults to None.
        """
        super().__init__(name)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.visual_model.to(self.device)

    def process(self, image_url: str):
        """
        Process an image by downloading it, validating it, and extracting visual features.

        Args:
            image_url (str): URL of the image to process.

        Returns:
            dict: Extracted visual features or an error message if processing fails.
        """
        self.log_info(f"Processing image from URL: {image_url}")

        # Download and validate the image
        image = self._download_image(image_url)
        if image is None:
            self.log_error("Image download failed.")
            return {"error": "Failed to download or process the image."}

        # Extract features using FeatureExtractor
        try:
            features = self.feature_extractor.extract_from_image(image)
            self.log_info("Feature extraction completed successfully.")
            return features
        except Exception as e:
            self.log_error(f"Error during feature extraction: {e}")
            return {"error": str(e)}

    def _download_image(self, image_url: str):
        """
        Download an image from a URL and return it as a PIL Image object.

        Args:
            image_url (str): URL of the image.

        Returns:
            PIL.Image.Image: Downloaded image object, or None if download fails.
        """
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            self.log_info("Image downloaded and loaded successfully.")
            return image
        except Exception as e:
            self.log_error(f"Failed to download image: {e}")
            return None
