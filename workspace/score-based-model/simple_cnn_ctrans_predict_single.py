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
        label = 1 if 'KIRC' in self.feature_dir else 0  # Assuming the directory name indicates the class
        if self.transform:
            features = self.transform(features)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), file_path


# Define the model (simple neural network for feature classification)
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Paths to the test data directories
kirc_test_dir = '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRC_ctrans/image_grid_350000'
kirp_test_dir = '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRP_ctrans/image_grid_350000'

batch_size = 1  # Batch size of 1 to print each file's prediction separately

# Initialize the model
input_size = 768
model = SimpleNN(input_size)

# Load the best model weights
best_model_path = 'saved_models_ctrans/best_model_ctrans.pth'
model.load_state_dict(torch.load(best_model_path))

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create datasets and dataloaders
kirc_test_dataset = NumpyDataset(kirc_test_dir)
kirp_test_dataset = NumpyDataset(kirp_test_dir)
test_dataset = torch.utils.data.ConcatDataset([kirc_test_dataset, kirp_test_dataset])

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Evaluation function to print predictions, probabilities, ground truth, and calculate metrics
def print_predictions_and_calculate_metrics(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels, file_paths in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            preds_class = (preds > 0.5).int()

            for i in range(len(file_paths)):
                print(
                    f"File: {file_paths[i]}, GT: {labels[i].item()}, Prediction: {preds_class[i].item()}, Probability: {preds[i].item():.4f}")
                y_true.append(labels[i].cpu().item())
                y_pred.append(preds[i].cpu().item())

    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_class = (y_pred > 0.5).astype(int)

    auc_roc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    log_loss_val = log_loss(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred_class)
    precision = precision_score(y_true, y_pred_class)
    recall = recall_score(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred_class)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)

    # Print metrics
    print(f"AUC-ROC: {auc_roc}")
    print(f"PR AUC: {pr_auc}")
    print(f"Log Loss: {log_loss_val}")
    print(f"Brier Score: {brier}")
    print(f"MCC: {mcc}")
    print(f"Precision (PPV): {precision}")
    print(f"Recall (Sensitivity): {recall}")
    print(f"Specificity: {specificity}")
    print(f"NPV: {npv}")
    print(f"F1 Score: {f1}")
    # print accuracy
    accuracy = (y_true == y_pred_class).mean()
    print(f"Accuracy: {accuracy}")

    # Print sorted predictions and ground truths
    sorted_indices = np.argsort(y_pred)
    sorted_y_true = y_true[sorted_indices]
    sorted_y_pred = y_pred[sorted_indices]
    for gt, pred in zip(sorted_y_true, sorted_y_pred):
        print(f"GT: {gt}, Pred: {pred:.4f}")


# Print predictions and probabilities for step 10000
print_predictions_and_calculate_metrics(model, test_loader)
