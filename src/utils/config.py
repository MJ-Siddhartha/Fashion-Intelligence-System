import os
import torch


class Config:
    # Base Paths
    BASE_PATH = os.path.abspath(os.path.dirname(__file__))
    DATA_PATH = os.path.join(BASE_PATH, r"C:\Users\mjsid\Reverse Coding\Hack\project\Reduced Dataset")
    PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed/")
    RAW_DATA_PATH = os.path.join(DATA_PATH, "raw/")
    MODELS_PATH = os.path.join(BASE_PATH, "models/")
    LOGS_PATH = os.path.join(BASE_PATH, "logs/")

    # Experiment Settings
    EXPERIMENT_NAME = "fashion_feature_extraction"
    MLFLOW_TRACKING_URI = "http://localhost:5000"  # MLflow tracking URI (could be local or remote server)

    # Model Hyperparameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 10
    DROPOUT_RATE = 0.3

    # Model Paths for Pretrained Models (if any)
    PRETRAINED_MODEL_PATH = os.path.join(MODELS_PATH, "pretrained/dinov2-base")

    # Log & Experiment Settings
    LOGGING_INTERVAL = 50  # Log after every N iterations
    CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs

    # Image and Text Processing Configurations
    IMAGE_SIZE = (224, 224)  # For resizing images
    TEXT_MAX_LENGTH = 512    # Max length for text tokenization

    # Hardware Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_CUDA = torch.cuda.is_available()

    @staticmethod
    def print_config():
        """
        Utility function to print all configurations in a readable format.
        """
        print("Configurations:")
        for attr, value in Config.__dict__.items():
            if not attr.startswith('__') and not callable(value):
                print(f"{attr}: {value}")
        print(f"Using CUDA: {'Yes' if Config.USE_CUDA else 'No'}")
        if Config.USE_CUDA:
            print(f"CUDA Device: {torch.cuda.get_device_name(Config.DEVICE)}")

    @staticmethod
    def ensure_directories():
        """
        Ensure all necessary directories exist.
        """
        dirs = [
            Config.DATA_PATH,
            Config.PROCESSED_DATA_PATH,
            Config.RAW_DATA_PATH,
            Config.MODELS_PATH,
            Config.LOGS_PATH
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)


# Example usage of Config class
if __name__ == "__main__":
    # Print configurations
    Config.print_config()
    
    # Ensure necessary directories exist
    Config.ensure_directories()
