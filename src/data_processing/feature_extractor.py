import re
from typing import List, Dict
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import torch
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms as transforms


class FeatureExtractor:
    def __init__(self):
        """
        Initialize components for feature extraction.
        """
        # Initialize visual feature extraction components
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match the model's input size
            transforms.ToTensor(),          # Convert image to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pre-trained models
        ])
        
        # Initialize the image processor and model for feature extraction
        self.visual_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')  # Image processor
        self.visual_model = AutoModel.from_pretrained('facebook/dinov2-base')  # Model to extract features

        # Set device to CUDA if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_model.to(self.device)

    def extract_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract features from text, such as keywords, hashtags, and alphanumeric tokens.

        Args:
            text (str): Input text.

        Returns:
            dict: A dictionary containing extracted features.
        """
        text_features = {
            "keywords": re.findall(r'\b\w{4,}\b', text.lower()),  # Extract keywords with 4+ characters
            "hashtags": re.findall(r'#\w+', text),                # Extract hashtags
            "alphanumeric": re.findall(r'\w+', text)              # Extract all alphanumeric tokens
        }
        return text_features

    def extract_from_image(self, image_path: str) -> Dict[str, any]:
        """
        Extract visual features such as dominant colors, color palette, and visual embeddings.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: A dictionary containing extracted image features.
        """
        try:
            # Load and process the image
            image = Image.open(image_path).convert('RGB')
            transformed_image = self.image_transform(image)  # Apply transformations

            # Extract feature embeddings using the pretrained model
            with torch.no_grad():
                inputs = self.visual_extractor(images=[transformed_image], return_tensors='pt')  # Extract image features
                outputs = self.visual_model(**inputs)  # Get output embeddings from the model
                feature_vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Take the mean of the embeddings

            # Extract color-related features
            dominant_colors = self._extract_dominant_colors(image)
            color_palette = self._extract_color_palette(image)

            return {
                "feature_vector": feature_vector.tolist(),
                "dominant_colors": dominant_colors,
                "color_palette": color_palette,
                "image_size": image.size
            }
        except Exception as e:
            return {"error": str(e)}  # Return error message if anything goes wrong

    def _extract_dominant_colors(self, image, n_colors: int = 3) -> List[str]:
        """
        Extract dominant colors using KMeans clustering.

        Args:
            image: PIL image object.
            n_colors (int): Number of dominant colors to extract.

        Returns:
            list: A list of dominant color names or hex values.
        """
        image_data = np.array(image.resize((100, 100)))  # Downscale image for faster processing
        image_data = image_data.reshape((-1, 3))  # Flatten image data to RGB values

        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(image_data)

        # Convert RGB to HEX for easier representation
        dominant_colors = ['#{:02x}{:02x}{:02x}'.format(*center.astype(int)) for center in kmeans.cluster_centers_]
        return dominant_colors

    def _extract_color_palette(self, image) -> List[str]:
        """
        Extract a broader color palette from the image.

        Args:
            image: PIL image object.

        Returns:
            list: A list of hex color values representing the palette.
        """
        image_data = np.array(image.resize((50, 50)))  # Downscale for faster processing
        unique_colors = np.unique(image_data.reshape((-1, 3)), axis=0)

        # Limit to top 10 unique colors and convert to HEX
        color_palette = ['#{:02x}{:02x}{:02x}'.format(*color) for color in unique_colors[:10]]
        return color_palette
