import os
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import copy
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
        label = 1 if 'KIRC' in file_path else 0  # Assuming the directory name indicates the class
        if self.transform:
            features = self.transform(features)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Paths to the data directories
base_dir = '/mnt/swarm_beta/sbm_evaluation/pathology_data/'
train_kirc_dir = os.path.join(base_dir, 'train_ctrans/KIRC')
train_kirp_dir = os.path.join(base_dir, 'train_ctrans/KIRP')
val_kirc_dir = os.path.join(base_dir, 'validation_ctrans/KIRC')
val_kirp_dir = os.path.join(base_dir, 'validation_ctrans/KIRP')
test_kirc_dir = os.path.join(base_dir, 'test_ctrans/KIRC')
test_kirp_dir = os.path.join(base_dir, 'test_ctrans/KIRP')

# Create datasets and dataloaders
batch_size = 32
epochs = 50

train_dataset = NumpyDataset(train_kirc_dir) + NumpyDataset(train_kirp_dir)
val_dataset = NumpyDataset(val_kirc_dir) + NumpyDataset(val_kirp_dir)
test_dataset = NumpyDataset(test_kirc_dir) + NumpyDataset(test_kirp_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dataloaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}

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

# Initialize the model
input_size = 768
model = SimpleNN(input_size)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TensorBoard writer
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

# Directory to save the best model and checkpoints
model_dir = 'saved_models_ctrans'
os.makedirs(model_dir, exist_ok=True)
best_model_path = os.path.join(model_dir, 'best_model_ctrans.pth')

# Early stopping parameters
early_stop_patience = 5

# Training function
def train_model(model, criterion, optimizer, dataloaders, num_epochs=25, early_stop_patience=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc_roc = 0.0
    early_stop_counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            y_true = []
            y_pred = []

            # Iterate over data with progress bar
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} {epoch+1}/{num_epochs}"):
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((preds > 0.5) == labels.data)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.detach().cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_auc_roc = roc_auc_score(y_true, y_pred)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC-ROC: {epoch_auc_roc:.4f}')

            # Log to TensorBoard
            writer.add_scalar(f'{phase} Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase} Accuracy', epoch_acc, epoch)
            writer.add_scalar(f'{phase} AUC-ROC', epoch_auc_roc, epoch)

            # Save the model for each epoch
            torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch}.pth'))

            # Deep copy the model
            if phase == 'val':
                if epoch_auc_roc > best_auc_roc:
                    best_auc_roc = epoch_auc_roc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f'Best val AUC-ROC: {best_auc_roc:4f}')

    # Load best model weights and save the model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), best_model_path)
    return model

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
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

# Train the model
model = train_model(model, criterion, optimizer, dataloaders, num_epochs=epochs, early_stop_patience=early_stop_patience)

# Evaluate the model
evaluate_model(model, dataloaders['test'])
