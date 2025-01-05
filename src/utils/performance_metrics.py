import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch


class PerformanceMetrics:
    def __init__(self, use_cuda: bool = False):
        """
        Initialize PerformanceMetrics with optional CUDA support.

        Args:
            use_cuda (bool): Whether to use CUDA for computations.
        """
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    def calculate_accuracy(self, predictions, labels):
        """
        Calculate accuracy as percentage of correct predictions using CUDA if available.

        Args:
            predictions (list): Model predictions
            labels (list): Ground truth labels

        Returns:
            float: Accuracy percentage
        """
        predictions_tensor = torch.tensor(predictions, device=self.device)
        labels_tensor = torch.tensor(labels, device=self.device)
        correct = (predictions_tensor == labels_tensor).sum().item()
        accuracy = (correct / len(labels)) * 100
        return accuracy

    def calculate_loss(self, predictions, labels):
        """
        Calculate mean squared error loss using CUDA if available.

        Args:
            predictions (list): Model predictions
            labels (list): Ground truth labels

        Returns:
            float: Mean squared error loss
        """
        predictions_tensor = torch.tensor(predictions, device=self.device, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, device=self.device, dtype=torch.float32)
        loss_tensor = ((predictions_tensor - labels_tensor) ** 2).mean()
        return loss_tensor.item()

    @staticmethod
    def save_metrics_to_file(metrics, filename='reports/performance_metrics.json'):
        """
        Save metrics dictionary to a JSON file.

        Args:
            metrics (dict): Dictionary containing performance metrics
            filename (str): Path to save the metrics JSON file
        """
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {filename}")


class PerformanceVisualizer:
    @staticmethod
    def generate_category_performance_plot(metrics_file='reports/performance_metrics.json'):
        """
        Generate comprehensive performance visualization from category performance metrics.

        Args:
            metrics_file (str): Path to performance metrics JSON
        """
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Create DataFrame from feature extraction rates
        df = pd.DataFrame.from_dict(
            metrics.get('feature_extraction_rates', {}),
            orient='index',
            columns=['Extraction Rate']
        )

        # Set up the plot
        plt.figure(figsize=(12, 6))
        sns.set(style="whitegrid")

        # Create bar plot
        ax = sns.barplot(
            x=df.index,
            y='Extraction Rate',
            data=df,
            hue=None,  # Explicitly set hue to None
            palette='viridis',
            legend=False  # No legend since there is no hue
        )


        # Customize plot
        plt.title('Feature Extraction Performance by Category', fontsize=16)
        plt.xlabel('Product Categories', fontsize=12)
        plt.ylabel('Extraction Rate (%)', fontsize=12)
        plt.xticks(rotation=45)

        # Add value labels
        for i, v in enumerate(df['Extraction Rate']):
            ax.text(i, v, f'{v:.2f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('reports/category_performance.png')
        plt.close()
        print("Category performance plot saved as 'category_performance.png'.")

    @staticmethod
    def generate_trend_analysis_plot(metrics_file='reports/performance_metrics.json'):
        """
        Generate trend analysis visualization from trend metrics.

        Args:
            metrics_file (str): Path to trend metrics JSON
        """
        # Load trend metrics (if applicable)
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            # Assuming trend analysis data structure contains 'accuracy' and 'epoch'
            df = pd.DataFrame(metrics['trend_analysis'])

            # Set up the plot for trend analysis (e.g., accuracy over epochs)
            plt.figure(figsize=(12, 6))
            sns.set(style="whitegrid")

            sns.lineplot(x='epoch', y='accuracy', data=df, marker='o')
            plt.title('Trend Analysis: Accuracy over Epochs', fontsize=16)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)

            plt.tight_layout()
            plt.savefig('reports/trend_analysis.png')
            plt.close()
            print("Trend analysis plot saved as 'trend_analysis.png'.")
        except Exception as e:
            print(f"Error generating trend analysis plot: {e}")


# Usage Example
if __name__ == "__main__":
    # Initialize PerformanceMetrics with CUDA support
    metrics_calculator = PerformanceMetrics(use_cuda=True)

    # Example predictions and labels
    predictions = [1, 0, 1, 1, 0]
    labels = [1, 0, 1, 0, 0]

    # Calculate accuracy and loss
    accuracy = metrics_calculator.calculate_accuracy(predictions, labels)
    loss = metrics_calculator.calculate_loss(predictions, labels)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Loss: {loss:.4f}")

    # Save metrics to file
    metrics = {
        'accuracy': accuracy,
        'loss': loss,
        'feature_extraction_rates': {
            'Category1': 85.2,
            'Category2': 92.5,
            'Category3': 78.3
        }
    }
    PerformanceMetrics.save_metrics_to_file(metrics)

    # Generate performance plots
    PerformanceVisualizer.generate_category_performance_plot()
