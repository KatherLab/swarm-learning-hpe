import os
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import copy
from tqdm import tqdm


# Custom Dataset for loading .npy feature files with four classes
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
        # Update the label logic here for four classes
        if 'KIRC_G1' in file_path:
            label = 0
        elif 'KIRC_G2' in file_path:
            label = 1
        elif 'KIRC_G3' in file_path:
            label = 2
        elif 'KIRC_G4' in file_path:
            label = 3
        else:
            raise ValueError("Unknown label")

        if self.transform:
            features = self.transform(features)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# Paths to the data directories
base_dir = '/mnt/swarm_alpha/KIRC_4grading/data_ctrans_train_val_test'
train_dirs = [os.path.join(base_dir, f'train/KIRC_G{i}') for i in range(1, 5)]
val_dirs = [os.path.join(base_dir, f'val/KIRC_G{i}') for i in range(1, 5)]
test_dirs = [os.path.join(base_dir, f'test/KIRC_G{i}') for i in range(1, 5)]

# Create datasets and dataloaders
batch_size = 32
epochs = 50

train_datasets = [NumpyDataset(d) for d in train_dirs]
val_datasets = [NumpyDataset(d) for d in val_dirs]
test_datasets = [NumpyDataset(d) for d in test_dirs]

train_dataset = torch.utils.data.ConcatDataset(train_datasets)
val_dataset = torch.utils.data.ConcatDataset(val_datasets)
test_dataset = torch.utils.data.ConcatDataset(test_datasets)

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
        self.fc3 = nn.Linear(128, 4)  # Update the output layer to 4 classes
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
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
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
    best_acc = 0.0
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
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} {epoch + 1}/{num_epochs}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.detach().cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log to TensorBoard
            writer.add_scalar(f'{phase} Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase} Accuracy', epoch_acc, epoch)

            # Save the model for each epoch
            torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch}.pth'))

            # Deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f'Best val Acc: {best_acc:4f}')

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
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Confusion Matrix:\n{cm}")


# Train the model
model = train_model(model, criterion, optimizer, dataloaders, num_epochs=epochs,
                    early_stop_patience=early_stop_patience)

# Evaluate the model
evaluate_model(model, dataloaders['test'])

'''
Best val Acc: 0.883236
Accuracy: 0.8846117779444861
F1 Score: 0.8844646067237826
Precision: 0.8853848043638798
Recall: 0.8846117779444861
Confusion Matrix:
[[ 415   92   34    0]
 [   9 8072  741  100]
 [   4  810 7673  173]
 [   1  127  370 2707]]
 '''