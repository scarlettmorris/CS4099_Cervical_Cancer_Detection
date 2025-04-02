# Class to load the processed_data.npz data into arrays and inspect these arrays
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_npz_data(filepath):
    # Load the .npz file
    data = np.load(filepath)
    # Inspect the keys (names of arrays saved in the file)
    #print(data.files)
    return data

def get_arrays(filepath):
    # Return the arrays in the .npz file
    data = load_npz_data(filepath)
    filenames = data['filenames']
    features = data['features']
    labels = data['labels']
    labels_numeric = data['labels_numeric']
    return filenames, features, labels, labels_numeric

def inspect_arrays(filenames, features, labels, labels_numeric):

    # Inspect the shapes
    print("Filenames shape:", filenames.shape)
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
    print("Labels numeric shape:", labels_numeric.shape)

    # explore the data (EDA - exploratory data analysis)
    # Count occurrences of each label
    label_counts = Counter(labels)

    # Separate the keys (label names) and values (counts)
    label_names = list(label_counts.keys())
    counts = list(label_counts.values())

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.bar(label_names, counts, color='skyblue')
    plt.xlabel("Labels", fontsize=14)
    plt.ylabel("Number of Data Points", fontsize=14)
    plt.title("Number of Data Points per Label", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.show()

def test_arrays_contents(filenames, features, labels, labels_numeric):
    for i in range(5):  # Inspect the first 5 samples
        print(f"Filenames {i}: {filenames[i]}")
        print(f"Feature {i}: {features[i]}")
        print(f"Text Label {i}: {labels[i]}")
        print(f"Numeric Label {i}: {labels_numeric[i]}")
        print()

# split the arrays
def get_index_for_splits(filenames):
    # get index of array where we split the .zarr filename for T/V/T sets
    # split train <= 1884
    # split validate > 1884 && <= 2512
    # split test > 2512 && 3140
    patches = []

    # get the index of the first occurrence of patch number 1885 and store as start_of_validation
    # get the index of the first occurrence of patch number 2513 and store as start_of_train
    start_of_validation_index = 0
    start_of_test_index = 0

    for i in range(len(filenames)):
        # Filenames 0: IC-CX-00001-01.patches/normal/IC-CX-00001-01-00000000-x8704-y15872-l1-s256-normal_inflammationLnormal_inflammationLtrainLnormal.png
        # split the filename to subtract patch number
        patch_number = filenames[i].split("-")[2].split("-")[0]
        if patch_number == '01885' and start_of_validation_index == 0:
            start_of_validation_index = i
        if patch_number == '02513' and start_of_test_index == 0:
            start_of_test_index = i
        patches.append(patch_number)
    
    print("Finished looping through patches")
    print("Start of train set index: 0")
    print('Start of validation set index: ', start_of_validation_index)
    print('Start of test index: ', start_of_test_index)


# create separate train/validate/test arrays
def split_arrays(filenames, features, labels, labels_numeric):
    
    # training & validation set index splits
    train_split = 2049574
    validate_split = 2818834
    # split data into train, validate, and test sets

    train_data = {
        "filenames": filenames[:train_split],
        "features": features[:train_split],
        "labels": labels[:train_split],
        "labels_numeric": labels_numeric[:train_split],
    }
    validate_data = {
        "filenames": filenames[train_split:validate_split],
        "features": features[train_split:validate_split],
        "labels": labels[train_split:validate_split],
        "labels_numeric": labels_numeric[train_split:validate_split],
    }
    test_data = {
        "filenames": filenames[validate_split:],
        "features": features[validate_split:],
        "labels": labels[validate_split:],
        "labels_numeric": labels_numeric[validate_split:],
    }

    # save arrays as .npz files
    np.savez_compressed("train_data.npz", **train_data)
    np.savez_compressed("validate_data.npz", **validate_data)
    np.savez_compressed("test_data.npz", **test_data)

    print("Data saved to train_data.npz, validate_data.npz, and test_data.npz")
    
def get_tvt_arrays(filenames):
    tvt = []
    
    for filename in filenames:
        # Filenames of format 0: IC-CX-00001-01.patches/normal/IC-CX-00001-01-00000000-x8704-y15872-l1-s256-normal_inflammationLnormal_inflammationLtrainLnormal.png
        # split the filename to subtract patch number
        full_label = filename.split("-")[-1].split(".png")[0]
        # gives us normal_inflammationLnormal_inflammationLtrainLnormal
        # Split by L delimiter and take 3rd occurrence
        tvt_split = full_label.split("L")[2]
        tvt.append(tvt_split)
        
    return tvt
    
    
# create separate train/validate/test arrays
def new_split_arrays(filenames, features, labels, labels_numeric):
    
    # for each element, get whether it is train/validate/test from its filename
    tvt = get_tvt_arrays(filenames)
    # print unique occurrences of tvt
    tvt_count = Counter(tvt)
    print("Number of occurrences of each value in tvt:", tvt_count)
    
    train_data = {"filenames": [], 
                  "features": [], 
                  "labels": [], 
                  "labels_numeric": []}
    validate_data = {"filenames": [], 
                     "features": [], 
                     "labels": [], 
                     "labels_numeric": []}
    test_data = {"filenames": [],
                 "features": [], 
                 "labels": [], 
                 "labels_numeric": []}
    
    for i in range(len(filenames)):
        if tvt[i] == "train":
            train_data["filenames"].append(filenames[i])
            train_data["features"].append(features[i])
            train_data["labels"].append(labels[i])
            train_data["labels_numeric"].append(labels_numeric[i])
        elif tvt[i] == "valid":
            validate_data["filenames"].append(filenames[i])
            validate_data["features"].append(features[i])
            validate_data["labels"].append(labels[i])
            validate_data["labels_numeric"].append(labels_numeric[i])
        else:
            test_data["filenames"].append(filenames[i])
            test_data["features"].append(features[i])
            test_data["labels"].append(labels[i])
            test_data["labels_numeric"].append(labels_numeric[i])
            
    print("Got here")
        
    # save arrays as .npz files
    np.savez_compressed("new_train_data.npz", **train_data)
    np.savez_compressed("new_validate_data.npz", **validate_data)
    np.savez_compressed("new_test_data.npz", **test_data)

    print("Data saved to new_train_data.npz, new_validate_data.npz, and new_test_data.npz")
    
    #Check the size of each array to ensure it matches the distribution expected
    print(f"Train set size: {len(train_data['filenames'])}")
    print(f"Validation set size: {len(validate_data['filenames'])}")
    print(f"Test set size: {len(test_data['filenames'])}")
    
    
    




