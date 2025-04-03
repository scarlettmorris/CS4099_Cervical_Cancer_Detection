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
        # Random Forest classifier
        # Uncomment code depending on if training/tuning/running saved model
        logger.info("Model: Random Forest Classifier")
        rf_model = RFModel()
        # trained_rf = rf_model.train_model(train_set, logger)
        # tuned_rf = rf_model.tune_model(train_set, logger)
        #trained_tuned_rf = rf_model.train_model(train_set, logger)
        model_path = "models/tuned_RF.joblib"
        model = rf_model.load_model(model_path, logger)
        rf_model.evaluate_model(train_set, 'train', model, logger)
        rf_model.evaluate_model(validate_set, 'validate', model, logger)
        rf_model.evaluate_model(test_set, 'test', model, logger)
        
    elif model == 'MLP':
        # Multi-layer Perceptron
        # Uncomment code depending on if training/running saved model
        logger.info("Model: MLP")
        mlp_model = MLP()
        #trained_mlp = mlp_model.train_model(train_set, logger)
        model_path = "models/mlp.pth"
        model = mlp_model.load_model(model_path, logger)
        mlp_model.evaluate_model(train_set, validate_set, test_set, model, logger)

    elif model == 'SVM':
        # Support Vector Machine
        # Uncomment code depending on if training/tuning/running saved model
        logger.info("Model: SVM")
        svm_model = SVMModel()
        #trained_svm = svm_model.train_model(train_set, logger)
        #tuned_svm = svm_model.tune_model(train_set, logger)
        model_path = "models/tuned_SVM.joblib"
        model = svm_model.load_model(model_path, logger)
        svm_model.evaluate_model(train_set, 'train', model, logger)
        svm_model.evaluate_model(validate_set, 'validate', model, logger)
        svm_model.evaluate_model(test_set, 'test', model, logger)
        
    elif model == 'XGB':
        # XGBoost
        # Uncomment code depending on if training/tuning/running saved model
        logger.info("Model: XGBoost")
        xgboost_model = XGBModel()
        #trained_xgb = xgboost_model.train_model(train_set, validate_set, logger)
        #tuned_xgb = xgboost_model.tune_model(train_set, validate_set, logger)
        model_path = "models/tuned_XGB.joblib"
        model = xgboost_model.load_model(model_path, logger)
        xgboost_model.evaluate_model(train_set, 'train', model, logger)
        xgboost_model.evaluate_model(validate_set, 'validate', model, logger)
        xgboost_model.evaluate_model(test_set, 'test', model, logger)
        
    elif model == 'TRN':
        # Transformer
        # Uncomment code depending on if training/running saved model
        logger.info("Model: Transformer")
        transformer_model = TransformerModel()
        #trained_transformer = transformer_model.train_model(train_set, logger)
        model_path = "models/transformer.pth"
        model = transformer_model.load_model(model_path, logger)
        transformer_model.evaluate_model('train', model, logger)
        transformer_model.evaluate_model('validate', model, logger)
        transformer_model.evaluate_model('test', model, logger)
        
    elif model == 'debug':
        print("Debug mode")
        
if __name__ == "__main__":
    model = sys.argv[1]
    main(model)