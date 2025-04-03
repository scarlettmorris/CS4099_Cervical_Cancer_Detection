import os
from metrics import plot_confusion_matrix
import torch
import torch.nn as nn
import time
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from load_data import get_arrays
import torch.optim as optim
from torch import cuda
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Code adapted from online tutorial: https://medium.com/@heyamit10/building-a-multiclass-classification-model-in-pytorch-a-detailed-practical-guide-b03fc93aa400

class MLP(nn.Module):
    # features_dim: 1024 (number of features)
    # classes_dim: 4 (number of output classes)
    
    def __init__(self):
        # initial network architecture taken from DOI: 10.15388/Informatica.2018.158
        super(MLP, self).__init__()
        features_size = 1024
        classes_size = 4
        
        self.fc1 = nn.Linear(features_size, 512)  # First hidden layer with 1000 neurons (decreased to 512 in investigation)
        self.relu1 = nn.ReLU()  # ReLU activation
        self.dropout1 = nn.Dropout(0.6)  # Dropout at 50%, then increased to 60%
        
        self.fc2 = nn.Linear(512, 20)  # Second hidden layer with 40 neurons (decreased to 20 in investigation)
        self.relu2 = nn.ReLU()  # ReLU activation
        self.dropout2 = nn.Dropout(0.5)  # Dropout at 30%, then increased to 50%
        
        self.fc3 = nn.Linear(20, classes_size)  # Output layer (num_classes = 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
    def train_model(self, train_set, logger):
        train_filenames, train_features, train_labels, train_labels_numeric = get_arrays("new_train_data.npz")
        validate_filenames, validate_features, validate_labels, validate_labels_numeric = get_arrays("new_validate_data.npz")
        
        train_loader = prepare_data(train_features, train_labels_numeric)
        validate_loader = prepare_data(validate_features, validate_labels_numeric, shuffle=False)
        logger.info("Dataset loaded.")
    
        model = MLP()
        logger.info(model)
        logger.info("Training MLP...")
        start_time = time.time()
        time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        logger.info(f"Training start time: {time_stamp}")
        
        device = 'cuda'
        criterion = nn.CrossEntropyLoss()
        optimiser = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        logger.info("Loss function and optimiser defined.")

        model.to(device)
    
        loss_values = []
        validation_loss_values = []
        train_accuracies = []
        val_accuracies = []
    
        # Training for 100 epochs
        for epoch in range(100):  
            # Turn model back to training mode after each validation at the end of each epoch
            model.train()
            running_loss = 0.0  
            predictions = []
            real_labels = []

            # Loop over training batches
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
            
                optimiser.zero_grad()           
                outputs = model(inputs)         
                loss = criterion(outputs, labels) 
                loss.backward()                 
                optimiser.step()                

                running_loss += loss.item()    
                predictions.extend(torch.argmax(outputs, dim = 1).cpu().numpy())
                real_labels.extend(labels.cpu().numpy())

            avg_loss = running_loss / len(train_loader)
            loss_values.append(avg_loss)
            accuracy = accuracy_score(real_labels, predictions)
            train_accuracies.append(accuracy)
        
            model.eval()
            validation_loss = 0.0
            validation_predictions = []
            validation_labels = []
        
            # Common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to 
            # turn off gradients computation - https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
            with torch.no_grad():
                for inputs, labels in validate_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                    validation_loss += loss.item()
                    validation_predictions.extend(torch.argmax(outputs, dim = 1).cpu().numpy())
                    validation_labels.extend(labels.cpu().numpy())
             
            avg_val_loss = validation_loss / len(validate_loader)
            validation_loss_values.append(avg_val_loss)   
            validation_accuracy = accuracy_score(validation_labels, validation_predictions)
            val_accuracies.append(validation_accuracy)
        
            logger.info(f"Epoch [{epoch+1}/100], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}")
        
        log_metrics(logger, "Training", real_labels, predictions)
        log_metrics(logger, "Validation", validation_labels, validation_predictions)
        
        # Plot the training and validation accuracy on the same plot
        # Code adapted from https://github.com/omaratef3221/pytorch_tutorials/blob/main/Ex_1_Tabular_Classification.ipynb
        plt.figure()
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy")
        plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()
        plt.savefig('training_validation_over_epoch.png')

        logger.info("Training completed.")
        
        end_time = time.time()
    
        time_to_train_seconds = end_time - start_time
        time_to_train_mins = time_to_train_seconds / 60
        time_to_train_hours = time_to_train_mins / 60
        logger.info(f"Time to train: {time_to_train_hours:.2f} hours")
    
        # Save model
        torch.save(model.state_dict(), 'mlp.pth')
        logger.info("Model saved.")

        return model
    
    def load_model(self, model_path, logger):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None

        logger.info(f"Loading model from: {model_path}")
        model = MLP().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        return model

    def evaluate_model(self, train_set, validate_set, test_set, model, logger): 
        loaders = {
            "Train": prepare_data(train_set["features"], train_set["labels_numeric"], shuffle=False),
            "Validation": prepare_data(validate_set["features"], validate_set["labels_numeric"], shuffle=False),
            "Test": prepare_data(test_set["features"], test_set["labels_numeric"], shuffle=False),
        }

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()

        with torch.no_grad():
            for data_set_type, data_loader in loaders.items():
                predictions = []
                true_labels = []

                for inputs, labels in data_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    predictions.extend(preds)
                    true_labels.extend(labels.cpu().numpy())

                log_metrics(logger, data_set_type, true_labels, predictions)

def prepare_data(features, labels_numeric, batch_size=64, shuffle=True):
        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_numeric, dtype=torch.long)

        dataset = TensorDataset(features_tensor, labels_tensor)
        # The DataLoader makes it easy to feed the data into the model
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return dataloader
    
def log_metrics(logger, dataset, y, pred):
    logger.info(f"Evaluating on {dataset} set")
    accuracy = accuracy_score(y, pred)
    logger.info(f"{dataset} Accuracy: {accuracy:.4f}")
    
    logger.info(f"Classification Report - {dataset} Data:")
    logger.info(classification_report(y, pred, target_names=["normal_inflammation", "low_grade", "high_grade", "malignant"]))

    plot_confusion_matrix(y, pred, f"Confusion Matrix: MLP on {dataset} Data", dataset)

    # Calculate sensitivity and specificity for malignancy
    cm = confusion_matrix(y, pred)
    tn = cm[0, 0] + cm[1, 1] + cm[2, 2]  
    tp = cm[3, 3]
    fn = sum(cm[3, :]) - tp
    fp = sum(cm[:, 3]) - tp

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    logger.info(f"Malignancy Sensitivity (Recall): {sensitivity:.4f}")
    logger.info(f"Malignancy Specificity: {specificity:.4f}")
    