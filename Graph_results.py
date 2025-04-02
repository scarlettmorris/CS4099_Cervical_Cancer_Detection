import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ['RF', 'SVM', 'XGB', 'MLP', 'Transformer']

# Accuracy scores
accuracy_scores = [69.8, 68.9, 69.0, 68.6, 67.6]

# Sensitivity scores
sensitivity_scores = [96.1, 90.7, 90.9, 89.3, 89.2]

# Bar chart for Accuracy Scores
plt.figure(figsize=(8, 6))
plt.bar(models, accuracy_scores, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy Scores (%)')
plt.title('Accuracy Scores by Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('val_accuracy_scores.png')
plt.show()

# Bar chart for Sensitivity Scores
plt.figure(figsize=(8, 6))
plt.bar(models, sensitivity_scores, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Sensitivity Scores (%)')
plt.title('Sensitivity Scores by Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('val_sensitivity_scores.png')
plt.show()
