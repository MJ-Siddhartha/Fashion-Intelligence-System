import urllib.parse
import torch


class TrendAdapter:
    def __init__(self, ontology, use_cuda: bool = False):
        """
        Initialize the TrendAdapter with the given ontology instance and device configuration.

        Args:
            ontology (FashionOntology): Instance of the ontology to be updated.
            use_cuda (bool): Whether to enable CUDA for computations.
        """
        self.ontology = ontology
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    def _sanitize_uri(self, name: str) -> str:
        """
        Sanitize a string to create a valid URI.

        Args:
            name (str): The string to be sanitized.

        Returns:
            str: A valid URI-compliant string.
        """
        return f"fashion:{urllib.parse.quote(name)}"

    def update_for_trend(self, category: str, new_feature: dict):
        """
        Update the ontology with a new trend under a specific category.

        Args:
            category (str): The category in which the new trend will be added.
            new_feature (dict): Details of the new feature/trend.
        """
        # Sanitize the feature name for use in the URI
        new_feature['name'] = self._sanitize_uri(new_feature['name'])
        self.ontology.add_feature(category, new_feature)
        print(f"Trend added: {urllib.parse.unquote(new_feature['name'])} to category: {category}")

    def batch_update_for_trends(self, category: str, features: list):
        """
        Add multiple trends to the ontology under a specific category.

        Args:
            category (str): The category in which the trends will be added.
            features (list): List of feature dictionaries to be added.
        """
        for feature in features:
            self.update_for_trend(category, feature)

    def display_trends(self, category: str):
        """
        Display all trends (features) under a specific category.

        Args:
            category (str): The category to display trends for.
        """
        relationships = self.ontology.get_feature_relationships(category)
        print(f"Trends under '{category}':")
        for feature in relationships.get('related_features', []):
            metadata = self.ontology.get_feature_relationships(feature).get('metadata', {})
            print(f"  - {urllib.parse.unquote(feature)}: {metadata}")

    def process_trends_with_cuda(self, trends: list):
        """
        Hypothetical method to process trends using CUDA. Assigns scores and sorts them.

        Args:
            trends (list): List of trend dictionaries to process.

        Example:
            trends = [
                {'name': 'Ripped Jeans', 'popularity': 80},
                {'name': 'Leather Jacket', 'popularity': 95}
            ]
        """
        # Convert trends' popularity scores to a CUDA-compatible tensor
        scores = torch.tensor([trend.get('popularity', 0) for trend in trends], device=self.device)
        
        # Sort the trends by popularity using CUDA (if available)
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_trends = [trends[i] for i in sorted_indices.tolist()]

        print("\nProcessed Trends (sorted by popularity):")
        for trend in sorted_trends:
            print(f"  - {trend['name']}: Popularity = {trend.get('popularity', 0)}")


# Usage Example
if __name__ == "__main__":
    from base_ontology import FashionOntology  # Ensure this path is accurate.

    # Initialize the Fashion Ontology
    ontology = FashionOntology()

    # Initialize the TrendAdapter with the ontology instance and CUDA support
    trend_adapter = TrendAdapter(ontology, use_cuda=True)

    # Add a single new trend
    trend_adapter.update_for_trend("Apparel", {
        'name': 'Oversized Hoodie',
        'material': 'Cotton',
        'style': 'Casual',
        'season': 'Winter'
    })

    # Add multiple trends in a batch
    trends = [
        {'name': 'Ripped Jeans', 'style': 'Casual', 'material': 'Denim', 'popularity': 80},
        {'name': 'Leather Jacket', 'style': 'Modern', 'material': 'Leather', 'popularity': 95},
        {'name': 'Summer Dress', 'style': 'Casual', 'material': 'Cotton', 'popularity': 75}
    ]
    trend_adapter.batch_update_for_trends("Apparel", trends)

    # Display all trends under "Apparel"
    print("\nAll Trends in 'Apparel':")
    trend_adapter.display_trends("Apparel")

    # Process trends using CUDA
    print("\nProcessing Trends with CUDA:")
    trend_adapter.process_trends_with_cuda(trends)
