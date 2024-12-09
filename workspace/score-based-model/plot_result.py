import re
import matplotlib.pyplot as plt

# Path to the results file
results_file_path = 'results.txt'

# Initialize lists to store metrics
steps = []
accuracies = []
auc_rocs = []
f1_scores = []
precisions = []
recalls = []
specificities = []
npvs = []

# Read and parse the results file
with open(results_file_path, 'r') as file:
    content = file.read()

# Use regular expressions to extract the metrics
step_pattern = re.compile(r'Results for step (\d+):')
accuracy_pattern = re.compile(r'Accuracy: ([\d\.]+)')
auc_roc_pattern = re.compile(r'AUC-ROC: ([\d\.]+)')
f1_score_pattern = re.compile(r'F1 Score: ([\d\.]+)')
precision_pattern = re.compile(r'Precision \(PPV\): ([\d\.]+)')
recall_pattern = re.compile(r'Recall \(Sensitivity\): ([\d\.]+)')
specificity_pattern = re.compile(r'Specificity: ([\d\.]+)')
npv_pattern = re.compile(r'NPV: ([\d\.]+)')

steps = [int(step) for step in step_pattern.findall(content)]
accuracies = [float(accuracy) for accuracy in accuracy_pattern.findall(content)]
auc_rocs = [float(auc_roc) for auc_roc in auc_roc_pattern.findall(content)]
f1_scores = [float(f1_score) for f1_score in f1_score_pattern.findall(content)]
precisions = [float(precision) for precision in precision_pattern.findall(content)]
recalls = [float(recall) for recall in recall_pattern.findall(content)]
specificities = [float(specificity) for specificity in specificity_pattern.findall(content)]
npvs = [float(npv) if npv != 'nan' else float('nan') for npv in npv_pattern.findall(content)]

# Plot the metrics
plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.plot(steps, accuracies, label='Accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.title('Accuracy per Step')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(steps, auc_rocs, label='AUC-ROC', color='orange')
plt.xlabel('Steps')
plt.ylabel('AUC-ROC')
plt.title('AUC-ROC per Step')
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(steps, f1_scores, label='F1 Score', color='green')
plt.xlabel('Steps')
plt.ylabel('F1 Score')
plt.title('F1 Score per Step')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(steps, precisions, label='Precision (PPV)', color='red')
plt.xlabel('Steps')
plt.ylabel('Precision (PPV)')
plt.title('Precision (PPV) per Step')
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(steps, recalls, label='Recall (Sensitivity)', color='purple')
plt.xlabel('Steps')
plt.ylabel('Recall (Sensitivity)')
plt.title('Recall (Sensitivity) per Step')
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(steps, specificities, label='Specificity', color='brown')
plt.xlabel('Steps')
plt.ylabel('Specificity')
plt.title('Specificity per Step')
plt.legend()

plt.tight_layout()
plt.show()
