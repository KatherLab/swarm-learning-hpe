import os
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
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
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

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
batch_size = 32  # You can adjust the batch size

# Initialize the model
input_size = 768
model = SimpleNN(input_size)

# Load the best model weights
best_model_path = 'saved_models_ctrans/best_model_ctrans.pth'
model.load_state_dict(torch.load(best_model_path))

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Evaluation function
def evaluate_model(model, dataloader, step):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_class = (y_pred > 0.5).astype(int)

    accuracy = (y_true == y_pred_class).mean()
    auc_roc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_class)
    precision = precision_score(y_true, y_pred_class)
    recall = recall_score(y_true, y_pred_class)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    specificity = tn / (tn + fp)
    sensitivity = recall
    ppv = precision
    npv = tn / (tn + fn)

    print(f"Accuracy: {accuracy}")
    print(f"AUC-ROC: {auc_roc}")
    print(f"F1 Score: {f1}")
    print(f"Precision (PPV): {precision}")
    print(f"Recall (Sensitivity): {recall}")
    print(f"Specificity: {specificity}")
    print(f"NPV: {npv}")

    # Write results to file
    with open('results.txt', 'a') as f:
        f.write(f"Results for step {step}:\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"AUC-ROC: {auc_roc}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Precision (PPV): {precision}\n")
        f.write(f"Recall (Sensitivity): {recall}\n")
        f.write(f"Specificity: {specificity}\n")
        f.write(f"NPV: {npv}\n")

# Evaluate the model for each step
for step in range(2000, 354000, 2000):
    kirc_test_dir = f'/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRC_ctrans/image_grid_{step}'
    kirp_test_dir = f'/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRP_ctrans/image_grid_{step}'

    # Create datasets and dataloaders
    kirc_test_dataset = NumpyDataset(kirc_test_dir)
    kirp_test_dataset = NumpyDataset(kirp_test_dir)
    test_dataset = torch.utils.data.ConcatDataset([kirc_test_dataset, kirp_test_dataset])

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model on the test set
    evaluate_model(model, test_loader, step)
