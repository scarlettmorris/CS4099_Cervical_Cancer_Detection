import os
import random
import time
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump, load
from metrics import get_training_time, plot_confusion_matrix

# XGBoost implementation adapted from https://gist.github.com/pb111/cc341409081dffa5e9eaf60d79562a03
class XGBModel:
    
    def train_model(self, train_set, validate_set, logger):
        # get the features & labels from the data
        X_train, y_train = train_set["features"], train_set["labels_numeric"]
        X_validate, y_validate = validate_set["features"], validate_set["labels_numeric"]
        
        # Convert data into DMatrix format (efficient data structure for XGBoost)
        d_train = xgb.DMatrix(X_train[:300000], label=y_train[:300000])
        d_validate = xgb.DMatrix(X_validate, label=y_validate)
        
        # baseline parameters taken from - https://github.com/StAndrewsMedTech/icairdpath-public/blob/main/repath/postprocess/slide_level_metrics.py
        baseline_params = {
        "objective": "multi:softmax", # outputs class labels instead of probabilities.
        "num_class": 4,               
        "eval_metric": "merror",      
        "eta": 0.5,                   
        "max_depth": 10,              
        "seed": 42                    
        }
        
        # best hyperparameters from random search: {'max_depth': 15, 'subsample': 0.6, 'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.1, 'seed': 42}
        params = {
        "objective": "multi:softmax",
        "num_class": 4,               
        "eval_metric": "merror",      
        "eta": 0.5,                   
        "max_depth": 15, 
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "gamma": 0,   
        "learning_rate": 0.1,
        "seed": 42                    
        }
        
        num_rounds = 100
            
        eval = [(d_train, "train"), (d_validate, "validate")]
        # train model
        logger.info("Training XGBoost Classifier...")
        start_time = time.time()
        start_time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        logger.info(f"Training start time: {start_time_stamp}")
        start_time = time.time()
        xgb_model = xgb.train(params, d_train, num_rounds, eval, early_stopping_rounds=10)
        end_time = time.time()
        end_time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
        logger.info(f"Training end time: {end_time_stamp}")
        
        time_to_train_hours = get_training_time(start_time, end_time)
        logger.info(f"Time to train: {time_to_train_hours} hours.")
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        dump(xgb_model, f'models/model_XGB_{timestamp}.joblib')

        return xgb_model
        
    
    def tune_model(self, train_set, validate_set, logger):
        # get the features & labels from the data
        X_train, y_train = train_set["features"], train_set["labels_numeric"] 
        X_validate, y_validate = validate_set["features"], validate_set["labels_numeric"]
        
        # Create the random grid
        # random grid = {}
        
        # Convert data into DMatrix format (efficient data structure for XGBoost)
        d_train = xgb.DMatrix(X_train[:300000], label=y_train[:300000])
        d_validate = xgb.DMatrix(X_validate, label=y_validate)
        
        random_grid = {
            "max_depth": [3, 5, 10, 15],          # Tree depth
            "learning_rate": [0.01, 0.05, 0.1, 0.3],  # Step size shrinkage
            "subsample": [0.6, 0.8, 1.0],         # Fraction of samples used for training
            "colsample_bytree": [0.6, 0.8, 1.0],  # Fraction of features used per tree
            "gamma": [0, 0.1, 0.2, 0.3],          # Minimum loss reduction required for split
        }
        
        num_rounds = 100
        num_searches = 10
        best_score = float("inf")
        best_params = None
        best_model = None
                        
        logger.info("Tuning XGB Classifier...")
        start_time = time.time()
        start_time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        logger.info(f"Tuning start time: {start_time_stamp}")
        
        for i in range(num_searches):
        # Randomly sample hyperparameters
            params = {
                "objective": "multi:softmax",
                "num_class": 4,
                "eval_metric": "merror",
                "eta": 0.5,
                "max_depth": random.choice(random_grid["max_depth"]),
                "subsample": random.choice(random_grid["subsample"]),
                "colsample_bytree": random.choice(random_grid["colsample_bytree"]),
                "gamma": random.choice(random_grid["gamma"]),
                "learning_rate": random.choice(random_grid["learning_rate"]),
                "seed": 42
            }

            logger.info(f"Hyperparameters: {params}")

            # Train model
            eval = [(d_train, "train"), (d_validate, "validate")]
            xgb_model = xgb.train(params, d_train, num_rounds, eval, early_stopping_rounds=10, verbose_eval=False)

            # Validation error (merror)
            validation_score = xgb_model.best_score
            logger.info(f"Validation merror: {validation_score}")

            # Update best model if performance improved
            if validation_score < best_score:
                best_score = validation_score
                best_params = params
                best_model = xgb_model
                
        end_time = time.time()
        end_time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
        logger.info(f"Tuning end time: {end_time_stamp}")

        logger.info("Best Hyperparameters from Random Search:")
        logger.info(best_params)
        logger.info(f"Best Validation Score (merror): {best_score:.4f}")
        
        time_to_train_hours = get_training_time(start_time, end_time)
        logger.info(f"Time to tune: {time_to_train_hours} hours.")

        return best_model
        
    def evaluate_model(self, data_set, data_set_type, model, logger):
        X, y = data_set["features"], data_set["labels_numeric"]
        d_matrix = xgb.DMatrix(X, label=y)
        
        if data_set_type == 'train':
            # evaluate on training set
            # run model on training set
            logger.info("Evaluating on training set...")
            set = "training"
        elif data_set_type == 'validate':
            # evaluate on validation set
            # run model on validation set
            logger.info("Evaluating on validation set...")
            set = "validation"
        elif data_set_type == 'test':
            # evaluate on test set
            logger.info("Evaluating on test set...")
            set = "test"
            
        pred = model.predict(d_matrix)
        accuracy = accuracy_score(y, pred)
        
        logger.info(f"{set} Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report, {set} data:")
        logger.info(classification_report(y, pred, target_names=["normal_inflammation", "low_grade", "high_grade", "malignant"]))
        title = f"Confusion Matrix: XGBoost on {set} data"
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