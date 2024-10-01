import os
from pipeline import run_pipeline

def main(dataset_path: str) -> None:
    """
    Main entry point of the application.

    Args:
        dataset_path (str): Path to the dataset containing high-resolution images.
    """
    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"The specified dataset path does not exist: {dataset_path}")

    # Run the entire pipeline
    run_pipeline(dataset_path)

if __name__ == "__main__":
    # Example usage; replace with your dataset path
    dataset_path = "/path/to/your/dataset"
    main(dataset_path)
