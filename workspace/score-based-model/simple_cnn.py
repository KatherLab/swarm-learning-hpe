import os
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import copy
from tqdm import tqdm

# Paths to the data directories
base_dir = '/mnt/swarm_alpha/KIRC_survival_data_train_val_test'  # Replace with your actual base directory path
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Hyperparameters
batch_size = 32
img_height = 224
img_width = 224
epochs = 50

# Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, transform=data_transforms['val']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}

# Create dataloaders
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=16),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=16),
    'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=16)
}

# Initialize the model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TensorBoard writer
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

# Directory to save the best model
model_dir = 'saved_models_KIRC_survival_data'
os.makedirs(model_dir, exist_ok=True)
best_model_path = os.path.join(model_dir, 'best_model.pth')

# Early stopping parameters
early_stop_patience = 5
early_stop_counter = 0

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
