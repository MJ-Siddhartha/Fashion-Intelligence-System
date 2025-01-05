import os
import pandas as pd
from sklearn.utils import shuffle

# Paths
INPUT_DIR = "NXT Hackathon Data"  # Original dataset directory
OUTPUT_DIR = "Reduced Dataset"  # Directory for reduced dataset
MAX_ITEMS = 100  # Number of items per file

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def reduce_csv(file_path, output_path):
    """
    Reduce a CSV file to a maximum number of items, ensuring diversity.
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Shuffle the data to ensure randomness
    df = shuffle(df, random_state=42)
    
    # Check for a meaningful grouping column (e.g., 'category_name')
    if "category_name" in df.columns:
        # Stratified sampling by category_name
        reduced_df = df.groupby("category_name", group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), MAX_ITEMS // len(df["category_name"].unique())), random_state=42
            )
        )
    else:
        # No grouping column available, just take a random sample
        reduced_df = df.head(MAX_ITEMS)
    
    # Save the reduced file
    reduced_df.to_csv(output_path, index=False)
    print(f"Reduced {os.path.basename(file_path)} to {len(reduced_df)} rows.")

# Process each CSV file in the input directory
for file_name in os.listdir(INPUT_DIR):
    if file_name.endswith(".csv"):
        input_path = os.path.join(INPUT_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name)
        reduce_csv(input_path, output_path)

print(f"All files reduced and saved in '{OUTPUT_DIR}'.")
