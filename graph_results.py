import matplotlib.pyplot as plt
import numpy as np

# Models
models = ['RF', 'SVM', 'XGB', 'MLP', 'Transformer', 'ICAIRD']

# Accuracy scores
accuracy_scores = [69.9, 69.1, 68.9, 67.5, 67.7, 78.79]

# Sensitivity scores
sensitivity_scores = [95.5, 89.9, 90.5, 87.4, 87.3, 76.04]

colors_acc = ['skyblue'] * (len(models) - 1) + ['pink']
colors_sens = ['skyblue'] * (len(models) - 1) + ['pink']

# Bar chart for Accuracy
plt.figure(figsize=(8, 6))
plt.bar(models, accuracy_scores, color=colors_acc)
plt.xlabel('Models')
plt.ylabel('Accuracy Scores (%)')
plt.title('Accuracy of each Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('accuracy_scores.png')
plt.show()

# Bar chart for Sensitivity
plt.figure(figsize=(8, 6))
plt.bar(models, sensitivity_scores, color=colors_sens)
plt.xlabel('Models')
plt.ylabel('Sensitivity Scores (%)')
plt.title('Malignant Sensitivity of each Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sensitivity_scores.png')
plt.show()
