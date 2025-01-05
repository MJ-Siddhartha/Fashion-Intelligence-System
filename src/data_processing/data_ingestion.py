import os
import pandas as pd

class DataIngestion:
    """
    Handles loading, previewing, and saving of dataset files.
    """

    @staticmethod
    def load_all_data(folder_path: str) -> pd.DataFrame:
        """
        Load all CSV files from the given folder and merge them into a single DataFrame.

        Args:
            folder_path (str): Path to the folder containing CSV files.

        Returns:
            pd.DataFrame: Combined DataFrame of all loaded CSV files.
        """
        dataframes = []
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                print(f"Loading file: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                    df['source_file'] = file  # Add a column to identify the source file
                    dataframes.append(df)
                except Exception as e:
                    print(f"Error loading file {file}: {e}")
        if not dataframes:
            raise ValueError("No valid CSV files found in the folder.")
        # Concatenate all DataFrames and return a single combined DataFrame
        return pd.concat(dataframes, ignore_index=True)

    @staticmethod
    def preview_data(dataframe: pd.DataFrame, rows: int = 5) -> None:
        """
        Print a preview of the first few rows of the DataFrame.

        Args:
            dataframe (pd.DataFrame): DataFrame to preview.
            rows (int): Number of rows to preview. Defaults to 5.
        """
        if dataframe.empty:
            print("The DataFrame is empty.")
        else:
            print(f"Previewing the first {rows} rows of the data:")
            print(dataframe.head(rows))

    @staticmethod
    def save_combined_data(dataframe: pd.DataFrame, output_file: str) -> None:
        """
        Save the combined DataFrame to a CSV file.

        Args:
            dataframe (pd.DataFrame): DataFrame to save.
            output_file (str): Path to the output file.
        """
        try:
            dataframe.to_csv(output_file, index=False)
            print(f"Combined data saved to {output_file}")
        except Exception as e:
            print(f"Error saving combined data: {e}")
