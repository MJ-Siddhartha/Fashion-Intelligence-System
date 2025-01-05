import networkx as nx
import json
from typing import Dict, Any, List
import rdflib


class FashionOntology:
    def __init__(self):
        """
        Initialize Fashion Ontology with a graph-based structure
        """
        self.graph = nx.DiGraph()
        self.rdf_graph = rdflib.Graph()
        self._initialize_base_ontology()

    def _initialize_base_ontology(self):
        """
        Create base taxonomy of fashion features
        """
        base_categories = {
            'Apparel': ['Top', 'Bottom', 'Dress', 'Outerwear'],
            'Accessories': ['Jewelry', 'Bags', 'Footwear', 'Headwear'],
            'Materials': ['Cotton', 'Silk', 'Polyester', 'Wool'],
            'Styles': ['Casual', 'Formal', 'Vintage', 'Modern']
        }

        for category, subcategories in base_categories.items():
            # Add main category nodes
            self.graph.add_node(category, type='main_category')

            # Add subcategories and connect them to main categories
            for subcategory in subcategories:
                self.graph.add_node(subcategory, type='subcategory')
                self.graph.add_edge(category, subcategory)

    def add_feature(self, category: str, feature: Dict[str, Any]):
        """
        Add a new feature to the ontology
        
        Args:
            category (str): Product category
            feature (Dict): Feature details
        """
        feature_name = feature.get('name')
        if not feature_name:
            raise ValueError("Feature must include a 'name' field.")

        # Add the feature node and connect it to the category
        self.graph.add_node(feature_name, **feature)
        self.graph.add_edge(category, feature_name)

        # RDF representation
        subject = rdflib.URIRef(f"fashion:{feature_name}")
        for key, value in feature.items():
            predicate = rdflib.URIRef(f"fashion:{key}")
            object_value = rdflib.Literal(value)
            self.rdf_graph.add((subject, predicate, object_value))

    def get_feature_relationships(self, feature: str) -> Dict[str, List[str]]:
        """
        Get relationships and context for a feature
        
        Args:
            feature (str): Feature to analyze
        
        Returns:
            Dict of feature relationships
        """
        if feature not in self.graph:
            raise ValueError(f"Feature '{feature}' not found in ontology.")

        predecessors = list(self.graph.predecessors(feature))
        successors = list(self.graph.successors(feature))
        
        return {
            'category': predecessors,
            'related_features': successors,
            'metadata': dict(self.graph.nodes[feature])
        }

    def display_ontology(self):
        """
        Display the full ontology hierarchy
        """
        for node in self.graph.nodes(data=True):
            print(f"Node: {node[0]}, Metadata: {node[1]}")
        for edge in self.graph.edges():
            print(f"Edge: {edge}")

    def export_ontology(self, format='json'):
        """
        Export ontology in specified format
        
        Args:
            format (str): Export format (json/rdf)
        
        Returns:
            Exported ontology representation
        """
        if format == 'json':
            return nx.node_link_data(self.graph)
        elif format == 'rdf':
            return self.rdf_graph.serialize(format='turtle')
        else:
            raise ValueError("Unsupported format. Choose 'json' or 'rdf'.")

    def merge_ontologies(self, other_ontology):
        """
        Merge another ontology with current ontology
        
        Args:
            other_ontology (FashionOntology): Ontology to merge
        """
        self.graph = nx.compose(self.graph, other_ontology.graph)
        self.rdf_graph += other_ontology.rdf_graph


# Usage example
if __name__ == "__main__":
    # Create an instance of the ontology
    ontology = FashionOntology()

    # Add a feature to the ontology
    ontology.add_feature('Dress', {
        'name': 'Floral Summer Dress',
        'pattern': 'Floral',
        'length': 'Knee',
        'sleeve_type': 'Sleeveless',
        'season': 'Summer'
    })

    # Display the ontology hierarchy
    print("Ontology Hierarchy:")
    ontology.display_ontology()

    # Get relationships for a specific feature
    feature_relationships = ontology.get_feature_relationships('Floral Summer Dress')
    print("\nFeature Relationships:")
    print(json.dumps(feature_relationships, indent=2))

    # Export ontology to JSON
    print("\nOntology Export (JSON):")
    print(json.dumps(ontology.export_ontology(format='json'), indent=2))
