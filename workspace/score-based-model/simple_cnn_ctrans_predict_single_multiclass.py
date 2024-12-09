import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, matthews_corrcoef, \
    precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Custom Dataset for loading .npy feature files
class NumpyDataset(Dataset):
    def __init__(self, feature_dir, transform=None):
        self.feature_dir = feature_dir
        self.transform = transform
        self.file_paths = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        features = np.load(file_path)
        # Extract the label (last digit before ".npy")
        label = int(file_path.split('_')[-1].split('.')[0])
        # discard lable 0 files

        if self.transform:
            features = self.transform(features)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long), file_path


# Define the model (simple neural network for feature classification)
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)  # Update to 4 classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Output logits for 4 classes
        return x


# Paths to the test data directories
test_dir = '/mnt/swarm_alpha/KIRC_4grading/samples_cut/ctrans_features'  # Update this with the actual path

batch_size = 1  # Batch size of 1 to print each file's prediction separately

# Initialize the model
input_size = 768
model = SimpleNN(input_size)

# Load the best model weights
best_model_path = '/opt/hpe/swarm-learning-hpe/workspace/score-based-model/saved_models_ctrans/best_model_ctrans.pth'
model.load_state_dict(torch.load(best_model_path))

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create dataset and dataloader
test_dataset = NumpyDataset(test_dir)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Evaluation function to print predictions, probabilities, ground truth, and calculate metrics
def print_predictions_and_calculate_metrics(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []
    file_paths_list = []

    with torch.no_grad():
        for inputs, labels, file_paths in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass to get outputs and predictions
            outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)  # Convert logits to probabilities
            preds_class = torch.argmax(probs, dim=1)  # Get predicted class

            for i in range(len(file_paths)):
                #print(f"File: {file_paths[i]}, GT: {labels[i].item()}, Prediction: {preds_class[i].item()}, "
                      #f"Probabilities: {probs[i].cpu().numpy()}")
                y_true.append(labels[i].cpu().item())
                y_pred.append(preds_class[i].cpu().item())
                file_paths_list.append(file_paths[i])

    # Convert to numpy arrays for metrics calculation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print confusion matrix and class-wise accuracy
    print(f"Confusion Matrix:\n{cm}")
    accuracy = (y_true == y_pred).mean()
    print(f"Accuracy: {accuracy}")

    # Class-wise precision, recall, and F1 score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Precision (Weighted): {precision}")
    print(f"Recall (Weighted): {recall}")
    print(f"F1 Score (Weighted): {f1}")

    # Log top misclassified files
    misclassified_files = [(file_paths_list[i], y_true[i], y_pred[i]) for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    #print(f"Top misclassified files (GT, Prediction): {misclassified_files}")


# Print predictions and calculate metrics
print_predictions_and_calculate_metrics(model, test_loader)
