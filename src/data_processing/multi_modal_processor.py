import requests
import torch
import torchvision.transforms as transforms
from transformers import AutoModel
import spacy
from PIL import Image
from typing import Dict, Any
import pandas as pd  # Ensure pandas is imported for text processing

class MultiModalProcessor:
    def __init__(self, device=None):
        """
        Initialize multi-modal feature extraction components with an optional device argument.
        """
        # Set device (use provided or default to CUDA/CPU)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MultiModalProcessor initialized on device: {self.device}")

        # Visual Model (DINOV2)
        self.visual_model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)

        # Textual Feature Extractor
        self.nlp = spacy.load('en_core_web_sm')

        # Image Transformation Pipeline
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_text(self, text: str) -> str:
        """
        Process text by converting it to lowercase and stripping any extra spaces.
        Handles non-string values gracefully by converting them to strings.
        """
        try:
            if isinstance(text, str):  # If it's already a string
                return text.lower().strip()
            elif pd.isna(text):  # Handle NaN or missing values
                return ""  # Return an empty string for missing descriptions
            else:  # Convert other types (e.g., float, int) to string
                return str(text).lower().strip()
        except Exception as e:
            print(f"Error processing text: {text}. Error: {e}")
            return ""  # Default to an empty string if an error occurs

    def process_image(self, image_url: str) -> str:
        """
        Basic image accessibility check.

        Args:
            image_url (str): URL of the image.

        Returns:
            str: Status message about image processing.
        """
        try:
            response = requests.get(image_url, timeout=5)
            if response.status_code == 200:
                return f"Image processed successfully from {image_url}"
            else:
                return "Error: Image not accessible"
        except Exception as e:
            return f"Error: {str(e)}"

    def extract_visual_features(self, image_path: str) -> Dict[str, Any]:
        """
        Extract visual features from an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            Dict[str, Any]: Extracted visual features.
        """
        try:
            # Load and transform the image
            image = Image.open(image_path).convert('RGB')
            transformed_image = self.image_transform(image)

            # Extract features using the model
            with torch.no_grad():
                inputs = transformed_image.unsqueeze(0).to(self.device)  # Add batch dimension and move to device
                outputs = self.visual_model(inputs)
                features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            return {
                'feature_vector': features.tolist(),
                'image_size': image.size
            }
        except Exception as e:
            return {'error': str(e)}

    def extract_textual_features(self, text: str) -> Dict[str, Any]:
        """
        Extract semantic features from text.

        Args:
            text (str): Input text description.

        Returns:
            Dict[str, Any]: Extracted textual features.
        """
        doc = self.nlp(text)

        return {
            'named_entities': [(ent.text, ent.label_) for ent in doc.ents],
            'pos_tags': [(token.text, token.pos_) for token in doc],
            'noun_chunks': [chunk.text for chunk in doc.noun_chunks]
        }

    def merge_features(self, visual_features: Dict[str, Any], textual_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge visual and textual features.

        Args:
            visual_features (Dict[str, Any]): Extracted visual features.
            textual_features (Dict[str, Any]): Extracted textual features.

        Returns:
            Dict[str, Any]: Merged feature set.
        """
        merged_features = {
            'feature_vector': visual_features.get('feature_vector', []),
            'named_entities': textual_features.get('named_entities', []),
        }

        return merged_features


if __name__ == "__main__":
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MultiModalProcessor(device=device)

    # Process text
    text = "This is a stunning floral summer dress made of cotton."
    print("Processed Text:", processor.process_text(text))
    print("Textual Features:", processor.extract_textual_features(text))

    # Process image (using a local file path)
    image_path = r"c:\Users\mjsid\Downloads\shirt.jpg"  # Replace with the path to your image
    print("Visual Features:", processor.extract_visual_features(image_path))
