import pandas as pd
import json
import zarr

def read_index():
    index_df = pd.read_csv("data/index.csv")
    print(index_df.head())

def read_json():
    with open("data/labels.json", "r") as f:
        label_mapping = json.load(f)
        print(label_mapping)

# Lables stored in an array where the index number of the filename relates to the index number of the feature
def get_labels_for_filename():
    file = "data/embeddings/IC-CX-00001-01.zarr"
    data = zarr.open(file, mode='r')
    filenames = data['filenames'][:]

    # Extract label from filename
    labels = []
    for fname in filenames:
        # Extract label from filename
        full_label = fname.split("-")[-1].split(".png")[0]
        # Split by L delimiter
        label = full_label.split("L")[0]
        labels.append(label)

    return labels