import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from swarmlearning.pyt import SwarmCallback
import os
import datetime
import glob


class Multiclass(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        return out
        
def loadData(dataDir, experiment, data_folder):
    """
    load data from dataDir, preprocess and return train and test data in torch tensors

    Parameters
    ----------
    dataDir : str
        Directory where data is stored
    experiment : str
        Experiment name: "test_R0", "test_R1", "test_R2", "test_Q0"
    data_folder: str
        Data folder name: "harmony", "pca", "scANVI", "scVI"

    Returns
    -------
    X_train : torch tensor
    y_train : torch tensor
    X_test : torch tensor
    y_test : torch tensor
    input_dim : int
    output_dim : int
    ohe : sklearn.preprocessing.OneHotEncoder fitted
    """
    
    # Define data paths
    X_train_path = glob.glob(os.path.join(dataDir,f'heart/{data_folder}/{experiment}/*X_*_train.npy'))
    y_train_path = glob.glob(os.path.join(dataDir,f'heart/{data_folder}/{experiment}/*Y_*_train.npy'))
    X_train_path = X_train_path[0]
    y_train_path = y_train_path[0]
    print(f"Loading train data from {X_train_path} and {y_train_path}")
    
    X_test_path = glob.glob(os.path.join(dataDir,f'heart/{data_folder}/{experiment}/*X_*_test.npy'))
    y_test_path = glob.glob(os.path.join(dataDir,f'heart/{data_folder}/{experiment}/*Y_*_test.npy'))
    X_test_path = X_test_path[0]
    y_test_path = y_test_path[0]
    print(f"Loading test data from {X_test_path} and {y_test_path}")
    
    # Load data
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    print("Data loaded successfully")
        
    # Preprocess labels
    REP1= { "r0$|r1$|r2$|r3$|q0$":"",
            "_":" ", "cells$":"cell", "es$": "e", "ts$":"t",
            "vsmcs":"smooth muscle",
            "neuronal|neurons":"neuron",
	        "fibroblasts":"fibroblast",
            "endothelium":"endothelial",
            "macrophages":"macrophage",
            "adipocytes":"adipocyte",
            "monocytes":"monocyte",
            "pericytes":"pericyte",
            "cardiomyocytes":"cardiomyocyte",
            "lymphocyte|lymphocytes|lymphoids":"lymphoid",
            "endothelium":"endothelial",
	        "^mast cells$|^mast cell$|^masts$":"mast",
	        "naive cd4 t":"cd4 naive t",
            "memory cd4 t":"cd4 memory t",
            "cd14\+ mono":"cd14 mono",
            "fcgr3a\+ mono":"cd16 mono"}

    REP2 = { "unassigned":"unassigned",
            "^b cell$":"b",
            "^cytotoxic t cell$":"cd8 t", 
            "^dendritic cell$":"dc",
            "^cd4\+ t cell$":"cd4 t",
            "^cd16\+ mono$":"cd16 mono",
            "^megakaryocyte$":"mk",
            "^natural killer cell$":"nk",
            "^plasmacytoid dendritic cell$":"pdc",
            "smooth muscle cell$":"smooth muscle", 
            "^mast cells$":"mast",
            "^smooth muscle cells$|vascular smooth muscle":"smooth muscle",
            "^ventricular cardiomyocyte$":"cardiomyocyte",
            "macrophages|macrophage|monocytes|monocyte":"myeloid",
            "^t\/nk cell$|^t\/nk cells$|^b cell$|^b cells$":"lymphoid"}
    
    df_y_train= pd.DataFrame(y_train)
    df_y_test= pd.DataFrame(y_test)

    df_y_train["cell_type_common"] = df_y_train[0].replace("I|II|III", "", regex=True).str.strip().str.lower()
    df_y_train["cell_type_common"] = df_y_train["cell_type_common"].replace(REP1, regex=True).str.strip()
    df_y_train["cell_type_common"] = df_y_train["cell_type_common"].replace(REP2, regex=True).str.strip()

    df_y_test["cell_type_common"] = df_y_test[0].replace("I|II|III", "", regex=True).str.strip().str.lower()
    df_y_test["cell_type_common"] = df_y_test["cell_type_common"].replace(REP1, regex=True).str.strip()
    df_y_test["cell_type_common"] = df_y_test["cell_type_common"].replace(REP2, regex=True).str.strip()

    y_train = df_y_train["cell_type_common"]
    y_test = df_y_test["cell_type_common"]
    
    # Encode labels
    encoder_data = pd.DataFrame(['adipocyte', 'atrial cardiomyocyte', 'cardiomyocyte',
        'cytoplasmic cardiomyocyte', 'endocardium', 'endothelial',
        'epicardium', 'fibroblast', 'lymphatic', 'lymphoid', 'mast',
        'mesothelial', 'myeloid', 'neuron', 'pericyte', 'prolif',
        'smooth muscle']).values.reshape(-1, 1) 
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(encoder_data)
    
    # Transform data to one-hot encoding, standardize and convert to tensors
    def transformation(X, y):
        y_1 = ohe.transform(y.values.reshape(-1, 1))
        X_1 = StandardScaler().fit_transform(X)
        X = torch.tensor(X_1, dtype=torch.float32)
        y = torch.tensor(y_1, dtype=torch.float32)
        return X, y
    
    X_train, y_train = transformation(X_train, y_train)
    X_test, y_test = transformation(X_test, y_test)
    
    input_dim = X_train.size()[1]
    output_dim = y_train.size()[1]
    
    print("Data preprocessed successfully")
    
    return X_train, y_train, X_test, y_test, input_dim, output_dim, ohe

def stats(model, X_test, y_test, output_dim, ohe, device, scratchDir, experiment_name):
    X_test, y_test = X_test.to(device), y_test.to(device)

    model.eval()
    y_pred = model(X_test)

    y_test = y_test.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    y_pred_labels = np.take(ohe.categories_, np.argmax(y_pred, axis=1)) 
    y_test_labels = ohe.inverse_transform(y_test).flatten()

    # Classification Report
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    report = pd.DataFrame(report).transpose()
    print(report.to_string)
    
    # AUROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(output_dim):
        fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.style.use('default')
    colors = plt.cm.tab20c(np.linspace(0,1, output_dim))
    plt.figure()
    for i in range(output_dim):
        plt.plot(fpr[i], tpr[i], lw=3, linestyle='dotted', color=colors[i],
                label='{0} (AUROC = {1:0.2f})'
                ''.format(ohe.categories_[0][i], roc_auc[i]))
    plt.plot(fpr["micro"], tpr["micro"],
            label='Average (AUROC = {0:0.2f})'
            ''.format(roc_auc["micro"]), linewidth=3, color='royalblue')

    plt.plot([0, 1], [0, 1], color='lightgrey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{experiment_name}")
    plt.legend(loc="lower left", bbox_to_anchor=(1,0))
    
    # Save Stats  
    file_name_prefix = experiment_name.replace(" ", "_")

    plt.savefig(os.path.join(scratchDir, f"{file_name_prefix}_roc.png"), bbox_inches = 'tight')
    report.to_csv(os.path.join(scratchDir, f"{file_name_prefix}_classification_report.csv"))
    np.save(os.path.join(scratchDir, f"{file_name_prefix}_y_pred.npy"), y_pred_labels)
    np.save(os.path.join(scratchDir, f"{file_name_prefix}_y_test.npy"), y_test_labels)

def directory(experiment, data_folder):
    dataDir = os.getenv('DATA_DIR', '/platform/data')
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M")

    folder_name_prefix = experiment.replace(" ", "_")
    folder = data_folder + folder_name_prefix + "_" + date_string
    
    new_dir = os.path.join(scratchDir, folder)
    os.makedirs(new_dir)

    if os.path.exists(new_dir):
        print("Directory created successfully:", new_dir)
    else:
        print("Failed to create directory:", new_dir)
    
    scratchDir = new_dir
    
    return dataDir, scratchDir
    
def main():
    # Set parameters and directories
    batchSz = 500
    default_max_epochs = 5
    default_min_peers = 2
    default_syncFrequency = 100
    default_experiment_name = "Swarm Learning"
    max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
    min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
    syncFrequency = int(os.getenv('SYNC_FREQUENCY', str(default_syncFrequency)))
    data_folder = os.getenv('DATA_FOLDER')
    experiment = os.getenv('EXPERIMENT')
    experiment_name = os.getenv('EXPERIMENT_NAME', default_experiment_name)
    dataDir, scratchDir = directory(experiment_name, data_folder)
    
    # Check if CUDA is available
    usecuda = torch.cuda.is_available()
    if usecuda:
        print('CUDA is accessable')
    else:
        print('CUDA  is not accesable')
    device = torch.device("cuda" if usecuda else "cpu")
    
    # Load data
    X_train, y_train, X_test, y_test, input_dim, output_dim, ohe = loadData(dataDir, experiment, data_folder)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
    
    # Define model, loss function, optimizer and number of batches per epoch
    model = Multiclass(input_dim=input_dim , output_dim=output_dim).to(device)
    model_name = 'multiclass'
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    trainDs = torch.utils.data.TensorDataset(X_train, y_train)
    valDs = torch.utils.data.TensorDataset(X_val, y_val)
    
    trainLoader = torch.utils.data.DataLoader(trainDs, batch_size=batchSz)
    
    # Create Swarm callback
    swarmCallback = None
    swarmCallback = SwarmCallback(syncFrequency=syncFrequency,
                                  minPeers=min_peers,
                                  useAdaptiveSync=False,
                                  adsValData=valDs,
                                  adsValBatchSize=batchSz,
                                  model=model)
    
    # initalize swarmCallback and do first sync 
    swarmCallback.on_train_begin()
    
    # Train model
    for epoch in range(max_epochs):
        # Set model in training mode and run through each batch
        model.train()
        for batchIdx, (X_batch, y_batch) in enumerate(trainLoader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # compute and store metrics
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            if batchIdx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}'.format(
                    epoch, max_epochs-1, batchIdx * len(X_batch), len(trainLoader.dataset),
                    100. * batchIdx / len(trainLoader), loss.item(), acc*100))
            if swarmCallback is not None:
                    swarmCallback.on_batch_end()     
                
        # Set model in evaluation mode and run through the test set
        model.eval()
        X_val, y_val = X_val.to(device), y_val.to(device)
        y_pred = model(X_val)
        ce = loss_fn(y_pred, y_val)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_val, 1)).float().mean()
        ce = float(ce)
        acc = float(acc)
        print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")
    
    # Handles what to do when training ends        
    swarmCallback.on_train_end()

    # Statistics
    stats(model, X_test, y_test, output_dim, ohe, device, scratchDir, experiment_name)
    
    # Save model and weights
    saved_model_path = os.path.join(scratchDir, model_name, 'saved_model.pt')
    os.makedirs(scratchDir, exist_ok=True)
    os.makedirs(os.path.join(scratchDir, model_name), exist_ok=True)
    torch.save(model, saved_model_path)
    print('Saved the trained model!')
     
if __name__ == '__main__':
  main()