import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def get_training_time(start_time, end_time):
    time_to_train_seconds = end_time - start_time
    time_to_train_mins = time_to_train_seconds / 60
    time_to_train_hours = time_to_train_mins / 60
    return time_to_train_hours

# function adapted from - tutorial at https://scikit-learn.org/dev/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
def plot_confusion_matrix(y, pred, title, dataset):
    cm = confusion_matrix(y, pred)
    # Convert count to percentage to display as % of TP/FP per class
    cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100 
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=["normal", "low_grade", "high_grade", "malignant"])
    disp.plot()
    plt.title(title)
    plt.show()
    time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    plt.savefig(f'confusion_matrix_{dataset}_{time_stamp}.png')