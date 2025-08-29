
import os
from pathlib import Path
from deepinv.datasets import SimpleFastMRISliceDataset

def get_fastmri_data():
    """
    Downloads the FastMRI dataset from the deepinverse huggingface repo.
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    print("Downloading knee dataset...")
    SimpleFastMRISliceDataset(root_dir=data_dir, anatomy="knee", download=True)
    print("Knee dataset downloaded.")

    print("Downloading brain dataset...")
    SimpleFastMRISliceDataset(root_dir=data_dir, anatomy="brain", download=True)
    print("Brain dataset downloaded.")

    print("All datasets downloaded successfully.")

if __name__ == "__main__":
    get_fastmri_data()
