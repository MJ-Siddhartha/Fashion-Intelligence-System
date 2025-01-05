import torch
from typing import List, Dict


class FeatureTaxonomy:
    def __init__(self, use_cuda: bool = False):
        """
        Initialize the Feature Taxonomy as a dictionary to store parent-child relationships.
        
        Args:
            use_cuda (bool): Whether to use CUDA for computations.
        """
        self.taxonomy = {}
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    def add_taxonomy(self, parent: str, child: str):
        """
        Add a relationship between a parent and a child in the taxonomy.

        Args:
            parent (str): Parent node
            child (str): Child node
        """
        if parent not in self.taxonomy:
            self.taxonomy[parent] = []
        self.taxonomy[parent].append(child)

    def get_children(self, parent: str) -> List[str]:
        """
        Retrieve the list of children for a given parent.

        Args:
            parent (str): Parent node

        Returns:
            List[str]: List of child nodes
        """
        return self.taxonomy.get(parent, [])

    def get_all_taxonomy(self) -> Dict[str, List[str]]:
        """
        Retrieve the entire taxonomy structure as a dictionary.

        Returns:
            dict: Taxonomy dictionary
        """
        return self.taxonomy

    def display_taxonomy(self):
        """
        Display the taxonomy in a readable format.
        """
        print("Feature Taxonomy:")
        for parent, children in self.taxonomy.items():
            print(f"{parent}: {', '.join(children)}")

    def add_bulk_taxonomy(self, relationships: Dict[str, List[str]]):
        """
        Add multiple parent-child relationships in bulk.

        Args:
            relationships (dict): Dictionary of {parent: [children]}
        """
        for parent, children in relationships.items():
            for child in children:
                self.add_taxonomy(parent, child)

    def process_data_on_device(self, data: List[int]) -> torch.Tensor:
        """
        Example method to process numerical data on the selected device (CPU/GPU).

        Args:
            data (List[int]): Input data to process.

        Returns:
            torch.Tensor: Processed data on the specified device.
        """
        tensor_data = torch.tensor(data, device=self.device)
        processed_data = tensor_data * 2  # Example computation
        return processed_data


# Usage Example
if __name__ == "__main__":
    # Create an instance of FeatureTaxonomy with CUDA support
    taxonomy = FeatureTaxonomy(use_cuda=True)

    # Add individual taxonomy relationships
    taxonomy.add_taxonomy("Apparel", "Dress")
    taxonomy.add_taxonomy("Apparel", "Top")
    taxonomy.add_taxonomy("Dress", "Floral Summer Dress")
    taxonomy.add_taxonomy("Top", "T-Shirt")
    taxonomy.add_taxonomy("Accessories", "Jewelry")
    taxonomy.add_taxonomy("Jewelry", "Earrings")

    # Display taxonomy
    print("\nInitial Taxonomy:")
    taxonomy.display_taxonomy()

    # Add bulk taxonomy relationships
    bulk_relationships = {
        "Materials": ["Cotton", "Silk", "Polyester"],
        "Styles": ["Casual", "Formal", "Vintage"]
    }
    taxonomy.add_bulk_taxonomy(bulk_relationships)

    # Display updated taxonomy
    print("\nUpdated Taxonomy:")
    taxonomy.display_taxonomy()

    # Retrieve and print children of a specific parent
    print("\nChildren of 'Apparel':")
    print(taxonomy.get_children("Apparel"))

    # Retrieve the entire taxonomy as a dictionary
    print("\nComplete Taxonomy Structure:")
    print(taxonomy.get_all_taxonomy())

    # Example of processing data on the CUDA/CPU device
    data = [1, 2, 3, 4, 5]
    processed_data = taxonomy.process_data_on_device(data)
    print("\nProcessed Data on Device:")
    print(processed_data)
