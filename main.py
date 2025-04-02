import sys

from load_data import load_npz_data
from rf_model import RFModel
from mlp_model import MLP
from logger import Logger
from svm_model import SVMModel
from transformer_model import TransformerModel
from xgb_model import XGBModel

def main(model):
    train_set = load_npz_data("new_train_data.npz")
    validate_set = load_npz_data("new_validate_data.npz")
    test_set = load_npz_data("new_test_data.npz")
    
    log = Logger(model)
    logger = log.get_logger()
    
    logger.info("Train, validate & test sets loaded.")
    
    if model == 'RF':
        # random forest classifier
        logger.info("Model: Random Forest Classifier")
        rf_model = RFModel()
        # train the model (no hyperparameter tuning) - get a baseline model 
        # trained_rf = rf_model.train_model(train_set, logger)
        # uncomment if not doing hyperparameter tuning
        # tuned_rf = rf_model.tune_model(train_set, logger)
        # get best hyperparameters from the tuning and run on test set 
        # evaluate the model on training set
        trained_tuned_rf = rf_model.train_model(train_set, logger)
        #model_path = "models/tuned_RF.joblib"
        #model = rf_model.load_model(model_path, logger)
        rf_model.evaluate_model(train_set, 'train', trained_tuned_rf, logger)
        # evaluate the model on validation set
        rf_model.evaluate_model(validate_set, 'validate', trained_tuned_rf, logger)
        # evaluate the model on test set
        rf_model.evaluate_model(test_set, 'test', trained_tuned_rf, logger)
        
    elif model == 'MLP':
        # multi-layer perceptron
        logger.info("Model: MLP")
        mlp_model = MLP()
        #trained_mlp = mlp_model.train_model(train_set, logger)
        model_path = "models/mlp.pth"  # Update with the correct path if necessary
        model = mlp_model.load_model(model_path, logger)
        mlp_model.evaluate_model(train_set, validate_set, test_set, model, logger)

    elif model == 'SVM':
        # support vector machine
        logger.info("Model: SVM")
        #svm_model = SVMModel(train_set, validate_set)
        svm_model = SVMModel()
        #trained_svm = svm_model.train_model(train_set, logger)
        #tuned_svm = svm_model.tune_model(train_set, logger)
        model_path = "models/tuned_SVM.joblib"
        model = svm_model.load_model(model_path, logger)
        #trained_tuned_svm = svm_model.train_model(train_set, logger)
        svm_model.evaluate_model(train_set, 'train', model, logger)
        svm_model.evaluate_model(validate_set, 'validate', model, logger)
        svm_model.evaluate_model(test_set, 'test', model, logger)
        
    elif model == 'XGB':
        # XGBoost
        logger.info("Model: XGBoost")
        #xgboost_model = XGBoostModel(train_set, validate_set)
        xgboost_model = XGBModel()
        trained_xgb = xgboost_model.train_model(train_set, validate_set, logger)
        #tuned_xgb = xgboost_model.tune_model(train_set, validate_set, logger)
        #trained_tuned_xgb = xgboost_model.train_model(train_set, validate_set, logger)
        model_path = "models/tuned_XGB.joblib"
        #model = xgboost_model.load_model(model_path, logger)
        xgboost_model.evaluate_model(train_set, 'train', trained_xgb, logger)
        xgboost_model.evaluate_model(validate_set, 'validate', trained_xgb, logger)
        #xgboost_model.evaluate_model(test_set, 'test', model, logger)
        
    elif model == 'TRN':
        # transformer
        logger.info("Model: Transformer")
        #transformer = TransformerModel(train_set, validate_set)
        transformer_model = TransformerModel()
        #trained_transformer = transformer_model.train_model(train_set, logger)
        #transformer_model.evaluate_model('train', trained_transformer, logger)
        #transformer_model.evaluate_model('validate', trained_transformer, logger)
        model_path = "models/transformer.pth"
        model = transformer_model.load_model(model_path, logger)
        transformer_model.evaluate_model('train', model, logger)
        transformer_model.evaluate_model('validate', model, logger)
        transformer_model.evaluate_model('test', model, logger)
        
    #elif model == 'LR':
        # logistic regression
        #ogger.info("Model: Logistic Regression")
        #logreg_model = LogRegModel(train_set, validate_set)
        #logreg_model = LogRegModel()
        #trained_logreg = logreg_model.train_model(train_set, logger)
        #tuned_logreg = logreg_model.tune_model(train_set, logger)
        #logreg_model.evaluate_model(train_set, 'train', tuned_logreg, logger)
        #logreg_model.evaluate_model(validate_set, 'validate', tuned_logreg, logger)
        
    elif model == 'debug':
        print("Debug mode")
        
if __name__ == "__main__":
    model = sys.argv[1]
    main(model)