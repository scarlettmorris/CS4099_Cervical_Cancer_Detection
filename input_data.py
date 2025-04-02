import os
import zarr
import numpy as np
import pandas as pd
import json


def get_label_from_filename(filename):
    """
    Extract metadata (e.g., level, labels) from a given filename.
    """
    # Example: Parse filename like
    # 'IC-CX-00001-01.patches/normal/IC-CX-00001-01-00000001-x8960-y15872-l1-s256-normal_inflammationLnormal_inflammationLtrainLnormal.png'
    full_label = filename.split("-")[-1].split(".png")[0]
    # Split by L delimiter 
    label = full_label.split("L")[0]
    return {
        "filename": filename,
        "label": label,
    }


def process_zarr_file(file_path, label_mapping):
    """
    Process a single Zarr file to extract features and labels.
    """
    zarr_file = zarr.open(file_path, mode='r')
    
    # Extract features and filenames
    features = zarr_file['features'][:]
    filenames = zarr_file['filenames'][:]

    # Collect data
    data = []
    for feature, filename in zip(features, filenames):
        label_data = get_label_from_filename(filename)
        # Return the number corresponding to each string label (from labels.json)
        label = label_data["label"]
        label_numeric = get_label_numeric(label)
        data.append({
            "filename": label_data["filename"],
            "feature": feature,
            "label": label_data["label"],
            "label_numeric": label_numeric,
        })
    
    return data

def get_label_numeric(label):

    # From labels.json
    label_mapping = {
        "normal_inflammation": 0,
        "low_grade": 1,
        "high_grade": 2,
        "malignant": 3
    }
    
    return label_mapping.get(label, -1)  # Default to -1 if the label is not found


# Process embeddings .zarr files and export data to numpy arrays
def main(embeddings_directory, json_label_path, output_file):
    """
    Main function to process all Zarr files and export data.
    """
    # Load label mapping
    with open(json_label_path, "r") as f:
        label_mapping = json.load(f)
    
    # Process all Zarr files
    zarr_files = sorted(
        [f for f in os.listdir(embeddings_directory) if f.endswith(".zarr")]
    )
    all_data = []

    for zarr_file in zarr_files:
        zarr_path = os.path.join(embeddings_directory, zarr_file)
        print(f"Processing file: {zarr_path}...")
        data = process_zarr_file(zarr_path, label_mapping)
        all_data.extend(data)
    
    # Export structured data
    print(f"Exporting data to {output_file}...")
    np.savez_compressed(
        output_file,
        filenames = np.array([d["filename"] for d in all_data]),
        features=np.array([d["feature"] for d in all_data]),
        labels=np.array([d["label"] for d in all_data]),
        labels_numeric=np.array([d["label_numeric"] for d in all_data]),
    )
    print("Export complete")

