import os
import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump, load
from metrics import get_training_time, plot_confusion_matrix

class SVMModel:
    
    def train_model(self, train_set, logger):
        # get the features & labels from the data
        X_train, y_train = train_set["features"], train_set["labels_numeric"]

        svm = LinearSVC(tol=0.01, max_iter=1000, C=0.1)
        #svm = LinearSVC()

        # train model
        logger.info("Training SVM Classifier...")
        start_time = time.time()
        start_time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        logger.info(f"Training start time: {start_time_stamp}")
        svm.fit(X_train[:300000], y_train[:300000])
        end_time = time.time()
        end_time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
        logger.info(f"Training end time: {end_time_stamp}")
        
        time_to_train_hours = get_training_time(start_time, end_time)
        logger.info(f"Time to train: {time_to_train_hours} hours.")
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        dump(svm, f'models/model_SVM_{timestamp}.joblib')
        
        return svm
        
    
    def tune_model(self, train_set, logger):
        # get the features & labels from the data
        X_train, y_train = train_set["features"], train_set["labels_numeric"]
        
        # Create the random grid
        random_grid = {
            'C': [0.1, 1, 10, 100, 1000],  
            'max_iter': [1000, 3000, 5000],
            'tol': [1e-4, 1e-3, 1e-2]  
            }
        # dual = False - https://github.com/scikit-learn/scikit-learn/issues/17339 when number of samples > number of features                
        svm = LinearSVC(dual=False)
        logger.info("Tuning SVM Classifier...")
        
        svm_tuned = RandomizedSearchCV(estimator=svm, param_distributions=random_grid, n_iter=100, cv=3, verbose=1, random_state=42, n_jobs=-1)
        
        start_time = time.time()
        start_time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        logger.info(f"Training start time: {start_time_stamp}")
        
        svm_tuned.fit(X_train[:100000], y_train[:100000])
        end_time = time.time()
        end_time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
        logger.info(f"Training end time: {end_time_stamp}")
        
        time_to_train_hours = get_training_time(start_time, end_time)
        logger.info(f"Time to train: {time_to_train_hours} hours.")
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        
        # See hyperparameters chosen
        logger.info("Hyperparameter results from the random grid search:")
        logger.info(svm_tuned.best_params_)
        logger.info(svm_tuned.best_score_)

        return svm_tuned
        
    def evaluate_model(self, data_set, data_set_type, model, logger):
        X, y = data_set["features"], data_set["labels_numeric"]
        
        if data_set_type == 'train':
            # evaluate on training set
            # run model on training set
            print("Evaluating on training set...")
            set = "training"
        elif data_set_type == 'validate':
            # evaluate on validation set
            # run model on validation set
            print("Evaluating on validation set...")
            set = "validation"
        elif data_set_type == 'test':
            # evaluate on test set
            print("Evaluating on test set...")
            set = "test"
            
        pred = model.predict(X)
        accuracy = accuracy_score(y, pred)
        
        logger.info(f"{set} Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report, {set} data:")
        logger.info(classification_report(y, pred, target_names=["normal_inflammation", "low_grade", "high_grade", "malignant"]))
        title = f"Confusion Matrix: SVM on {set} data"
        plot_confusion_matrix(y, pred, title, set)
        cm = confusion_matrix(y, pred)
        
        # Calculate sensitivity (recall) and specificity for malignancy (last class)
        tn = cm[0, 0] + cm[1, 1] + cm[2, 2]  # Sum of true negatives for non-malignant classes
        tp = cm[3, 3]  # True positives for malignant class
        fn = sum(cm[3, :]) - tp  # False negatives for malignant class
        fp = sum(cm[:, 3]) - tp  # False positives for malignant class
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        logger.info(f"Malignancy Sensitivity (Recall): {sensitivity:.4f}")
        logger.info(f"Malignancy Specificity: {specificity:.4f}")
        
        
    def load_model(self, model_path, logger):
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        logger.info(f"Loading model from: {model_path}")
        model = load(model_path)
        
        return model