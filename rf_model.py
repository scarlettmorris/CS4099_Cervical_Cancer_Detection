import os
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump, load
from metrics import get_training_time, plot_confusion_matrix

class RFModel:
    
    def train_model(self, train_set, logger):
        X_train, y_train = train_set["features"], train_set["labels_numeric"]

        rfc = RandomForestClassifier(n_estimators=188, min_samples_split=2, min_samples_leaf=2, max_features='log2', max_depth=23, criterion= 'entropy', bootstrap= True)

        logger.info("Training Random Forest Classifier...")
        start_time = time.time()
        start_time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        logger.info(f"Training start time: {start_time_stamp}")
        # Add in sampling if investigating for experiment e.g. X_train[:300000]
        # Train model
        rfc.fit(X_train, y_train)
        end_time = time.time()
        end_time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
        logger.info(f"Training end time: {end_time_stamp}")
        
        time_to_train_hours = get_training_time(start_time, end_time)
        logger.info(f"Time to train: {time_to_train_hours} hours.")
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        dump(rfc, f'models/model_RF_{timestamp}.joblib')
        
        return rfc
        
    
    # Function adapted from - https://github.com/StAndrewsMedTech/icairdpath-public/blob/main/repath/postprocess/slide_level_metrics.py
    def tune_model(self, train_set, logger):
        X_train, y_train = train_set["features"], train_set["labels_numeric"]
        
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]
        criterion = ['gini', 'entropy']
        max_features = ['sqrt', 'log2']
        max_depth = [int(x) for x in np.linspace(2, 100, num = 10)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1,2,4]
        bootstrap = [True, False]
        
        # Grid of parameters for Random Search
        random_grid = {'n_estimators': n_estimators,
                       'criterion': criterion ,
                       'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
                        
        rfc = RandomForestClassifier()
        logger.info("Tuning Random Forest Classifier...")
        
        rfc_tuned = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=0, random_state=42, n_jobs = -1)
        
        start_time = time.time()
        start_time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        logger.info(f"Training start time: {start_time_stamp}")
        
        # Tune on only a subset of the data, due to memory limitations
        rfc_tuned.fit(X_train[:300000], y_train[:300000])
        end_time = time.time()
        end_time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
        logger.info(f"Training end time: {end_time_stamp}")
        
        time_to_train_hours = get_training_time(start_time, end_time)
        logger.info(f"Time to train: {time_to_train_hours} hours.")
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        
        # Output hyperparameters chosen
        logger.info("Hyperparameter results from the random grid search:")
        logger.info(rfc_tuned.best_params_)
        logger.info(rfc_tuned.best_score_)

        return rfc_tuned
        
    def evaluate_model(self, data_set, data_set_type, model, logger):
        X, y = data_set["features"], data_set["labels_numeric"]
        
        if data_set_type == 'train':
            print("Evaluating on training set...")
            set = "training"
        elif data_set_type == 'validate':
            print("Evaluating on validation set...")
            set = "validation"
        elif data_set_type == 'test':
            print("Evaluating on testing set...")
            set = "test"
            
        pred = model.predict(X)
        accuracy = accuracy_score(y, pred)
        
        logger.info(f"{set} Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report, {set} data:")
        logger.info(classification_report(y, pred, target_names=["normal_inflammation", "low_grade", "high_grade", "malignant"]))
        title = f"Confusion Matrix: RF on {set} data"
        plot_confusion_matrix(y, pred, title, set)
        cm = confusion_matrix(y, pred)
        
        # Calculate sensitivity and specificity for malignancy
        tn = cm[0, 0] + cm[1, 1] + cm[2, 2] 
        tp = cm[3, 3]
        fn = sum(cm[3, :]) - tp 
        fp = sum(cm[:, 3]) - tp 
        
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
    