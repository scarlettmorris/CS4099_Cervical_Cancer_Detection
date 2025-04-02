import os
from metrics import get_training_time, plot_confusion_matrix
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, TensorDataset
from load_data import get_arrays
import torch.optim as optim
from torch import cuda
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# code adapted from https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html 
# and https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/transformer_tutorial.ipynb
# and https://medium.com/@heyamit10/building-a-multiclass-classification-model-in-pytorch-a-detailed-practical-guide-b03fc93aa400

class TransformerModel(nn.Module):
    
    # hyperparameters from - https://doi.org/10.1007/978-3-031-66955-2_23
    def __init__(self, input_dim=1024, model_dim=512, feedforward_dim=2048, num_classes=4, num_heads=4, num_layers=6, dropout = 0.15):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=feedforward_dim, activation = "gelu", dropout = dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        x = self.fc(x)
        return x
    
    def train_model(self, train_set, logger):
        train_filenames, train_features, train_labels, train_labels_numeric = get_arrays("new_train_data.npz")
        validate_filenames, validate_features, validate_labels, validate_labels_numeric = get_arrays("new_validate_data.npz")
        
        train_loader = prepare_data(train_features, train_labels_numeric)
        validate_loader = prepare_data(validate_features, validate_labels_numeric)
        logger.info("Dataset loaded successfully!")
        
        model = TransformerModel()
        logger.info(model)
        logger.info("Training Transformer...")
        start_time = time.time()
        start_time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        logger.info(f"Training start time: {start_time_stamp}")
        
        criterion = nn.CrossEntropyLoss()
        device = 'cuda' if cuda.is_available() else 'cpu'
        optimiser = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=50, gamma=0.1)
        model.to(device)
        
        loss_values = []
        validation_loss_values = []
        train_accuracies = []
        validation_accuracies = []
        
        # changed to 100 epochs after investigations
        for epoch in range(100):
            model.train()
            running_loss = 0.0  
            predictions, real_labels = [], []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimiser.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

                running_loss += loss.item()
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                real_labels.extend(labels.cpu().numpy())
                
            avg_loss = running_loss / len(train_loader)
            loss_values.append(avg_loss)
            accuracy = accuracy_score(real_labels, predictions)
            train_accuracies.append(accuracy)
            
            model.eval()
            validation_loss = 0.0
            validation_predictions, validation_labels = [], []
            
            with torch.no_grad():
                for inputs, labels in validate_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    validation_loss += loss.item()
                    validation_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    validation_labels.extend(labels.cpu().numpy())
                    
            avg_val_loss = validation_loss / len(validate_loader)
            validation_loss_values.append(avg_val_loss)
            validation_accuracy = accuracy_score(validation_labels, validation_predictions)
            validation_accuracies.append(validation_accuracy)
            
            scheduler.step()
            
            logger.info(f"Epoch [{epoch+1}/200], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}")
        
        log_metrics(logger, "Training", real_labels, predictions)
        log_metrics(logger, "Validation", validation_labels, validation_predictions)
        
        end_time = time.time()
        end_time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
        logger.info(f"Training end time: {end_time_stamp}")
        time_to_train_hours = get_training_time(start_time, end_time)
        logger.info(f"Time to train: {time_to_train_hours} hours.")
        
        plt.figure()
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy")
        plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()
        time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) 
        plt.savefig(f"training_validation_transformer_{time_stamp}.png")

        logger.info("Training completed successfully!")
        torch.save(model.state_dict(), 'transformer.pth')
        return model
    
    def evaluate_model(self, data_set_type, model, logger):
        
        if data_set_type == 'train':
            # evaluate on training set
            # run model on training set
            print("Evaluating on training set...")
            set = "training"
            train_filenames, train_features, train_labels, train_labels_numeric = get_arrays("new_train_data.npz")
            loader = prepare_data(train_features, train_labels_numeric)
        elif data_set_type == 'validate':
            # evaluate on validation set
            # run model on validation set
            print("Evaluating on validation set...")
            set = "validation"
            validate_filenames, validate_features, validate_labels, validate_labels_numeric = get_arrays("new_validate_data.npz")
            loader = prepare_data(validate_features, validate_labels_numeric)
        elif data_set_type == 'test':
            # evaluate on test set
            print("Evaluating on testing set...")
            set = "test"
            test_filenames, test_features, test_labels, test_labels_numeric = get_arrays("new_test_data.npz")
            loader = prepare_data(test_features, test_labels_numeric)
            
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(device)
    
        predictions, real_labels = [], []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                real_labels.extend(labels.cpu().numpy())
            
        accuracy = accuracy_score(real_labels, predictions)
        logger.info(f"{set} Accuracy: {accuracy:.4f}")
        log_metrics(logger, set, real_labels, predictions)
        
    def load_model(self, model_path, logger):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None

        logger.info(f"Loading model from: {model_path}")
        model = TransformerModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        # eval() turns dropout off, and batch normalisation uses the stored running mean and variance rather
        model.eval()

        return model
        


def prepare_data(features, labels_numeric, batch_size=64, shuffle=True):
    # Dataloader feeds the data in batches to the neural net for training and processing. 
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_numeric, dtype=torch.long)

    dataset = TensorDataset(features_tensor, labels_tensor)
    # the DataLoader handles batching and shuffling, making it easy to feed the data into the model
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    return dataloader
    
def log_metrics(logger, dataset, y, pred):
    accuracy = accuracy_score(y, pred)
    logger.info(f"{dataset} Accuracy: {accuracy:.4f}")
    
    logger.info(f"Classification Report - {dataset} Data:")
    logger.info(classification_report(y, pred, target_names=["normal_inflammation", "low_grade", "high_grade", "malignant"]))

    plot_confusion_matrix(y, pred, f"Confusion Matrix: Transformer on {dataset} Data", dataset)

    cm = confusion_matrix(y, pred)
    tn = cm[0, 0] + cm[1, 1] + cm[2, 2]  
    tp = cm[3, 3]
    fn = sum(cm[3, :]) - tp
    fp = sum(cm[:, 3]) - tp

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    logger.info(f"Malignancy Sensitivity (Recall): {sensitivity:.4f}")
    logger.info(f"Malignancy Specificity: {specificity:.4f}")