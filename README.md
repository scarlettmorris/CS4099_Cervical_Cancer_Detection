# Deep Learning for Cervical Cancer Detection using UNI Feature Extractions from Histopathological Images
This project forms the software artifact for a BSci Dissertation. 


## Setup 
### Working with original embeddings dataset:
If working with the original embeddings dataset (consisting of the .zarr files in a directory called 'embeddings'), run the notebook process_embeddings.ipynb to process each zarr file, extract each patch's label, convert feature embeddings and corresponding labels into numpy arrays, and then compress these arrays into new_processed_data.npz. 

Run new_load_data.ipynb to load the numpy arrays (an array for filenames, features, labels and a numeric representation of each label) and then split them into train, validate and test sets (new_train_data.npz, new_validate_data.npz, and new_test_data.npz).

### Working directly with new_train_data.npz, new_validate_data.npz, and new_test_data.npz
Datasets can be found at: 10.5281/zenodo.15121399

Ensure files are in the same directory as all .py files. 

### Working with the Docker container

## Usage 
Run:
```
python main.py <model_flag>
```
where model_flag is one of:

* `RF` : Random Forest
* `MLP`: Multilayer Perceptron
* `SVM`: Support Vector Machine
* `XGB`: XGBoost
* `TRN`: Transformer
