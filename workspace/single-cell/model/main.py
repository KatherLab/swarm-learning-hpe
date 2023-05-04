import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_curve, auc, classification_report
import copy
import datetime
import os
import time
from swarmlearning.pyt import SwarmCallback

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
        
def loadData(dataDir, device):
    """
    load data from dataDir, preprocess and return train and test data in torch tensors

    Parameters
    ----------
    dataDir : str
        Directory where data is stored
    device : str
        cpu or cuda

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
    
    # Load data
    X_train = np.load(os.path.join(dataDir,'Single_cell_swarm_input/X_train.npy'))
    X_test = np.load(os.path.join(dataDir,'Single_cell_swarm_input/X_test.npy'))
    y_train = np.load(os.path.join(dataDir,'Single_cell_swarm_input/y_train.npy'))
    y_test = np.load(os.path.join(dataDir,'Single_cell_swarm_input/y_test.npy'))
    
    # Convert to pandas dataframe
    df_ytrain= pd.DataFrame(y_train)
    df_ytest= pd.DataFrame(y_test) 
    
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
    
    df_ytrain["cell_type_common"] = df_ytrain[0].replace("I|II|III", "", regex=True).str.strip().str.lower()
    df_ytrain["cell_type_common"] = df_ytrain["cell_type_common"].replace(REP1, regex=True).str.strip()
    df_ytrain["cell_type_common"] = df_ytrain["cell_type_common"].replace(REP2, regex=True).str.strip()

    df_ytest["cell_type_common"] = df_ytest[0].replace("I|II|III", "", regex=True).str.strip().str.lower()
    df_ytest["cell_type_common"] = df_ytest["cell_type_common"].replace(REP1, regex=True).str.strip()
    df_ytest["cell_type_common"] = df_ytest["cell_type_common"].replace(REP2, regex=True).str.strip()

    y_train = df_ytrain["cell_type_common"]
    y_test = df_ytest["cell_type_common"]
    
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
    
    return X_train, y_train, X_test, y_test, input_dim, output_dim, ohe

def main():
    # Set parameters and directories
    batchSz = 500
    default_max_epochs = 5
    default_min_peers = 2
    default_syncFrequency = 20
    
    dataDir = os.getenv('DATA_DIR', '/platform/data')
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    modelDir = os.getenv('MODEL_DIR', '/platform/model')
    max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
    min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
    syncFrequency = int(os.getenv('SYNC_FREQUENCY', str(default_syncFrequency)))
    
    # Check if CUDA is available
    usecuda = torch.cuda.is_available()
    if usecuda:
        print('CUDA is accessable')
    else:
        print('CUDA  is not accesable')
    device = torch.device("cuda" if usecuda else "cpu")
    
    # Load data
    X_train, y_train, X_test, y_test, input_dim, output_dim, ohe = loadData(dataDir, device)
    
    # Define model, loss function, optimizer and number of batches per epoch
    model = Multiclass(input_dim=input_dim , output_dim=output_dim).to(device)
    model_name = 'multiclass'
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    trainDs = torch.utils.data.TensorDataset(X_train, y_train)
    testDs = torch.utils.data.TensorDataset(X_test, y_test)
    
    trainLoader = torch.utils.data.DataLoader(trainDs, batch_size=batchSz)
    
    # Create Swarm callback
    swarmCallback = None
    swarmCallback = SwarmCallback(syncFrequency=syncFrequency,
                                  minPeers=min_peers,
                                  useAdaptiveSync=False,
                                  adsValData=testDs,
                                  adsValBatchSize=batchSz,
                                  model=model)
    
    # initalize swarmCallback and do first sync 
    swarmCallback.on_train_begin()
    
    # Train model
    for epoch in range(max_epochs):
        epoch_loss = []
        epoch_acc = []
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
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            if batchIdx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}'.format(
                    epoch, batchIdx * len(X_batch), len(trainLoader.dataset),
                    100. * batchIdx / len(trainLoader), loss.item(), acc*100))
            if swarmCallback is not None:
                    swarmCallback.on_batch_end()     
                
        # Set model in evaluation mode and run through the test set
        model.eval()
        X_test, y_test = X_test.to(device), y_test.to(device)
        y_pred = model(X_test)
        ce = loss_fn(y_pred, y_test)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
        ce = float(ce)
        acc = float(acc)
        print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")
        swarmCallback.on_epoch_end(epoch)
    
    # Handles what to do when training ends        
    swarmCallback.on_train_end()

    # AUROC
    y_test = y_test.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    y_pred_labels = np.take(ohe.categories_, np.argmax(y_pred, axis=1)) 
    y_test_labels = ohe.inverse_transform(y_test).flatten()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(output_dim):
        fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.style.use('default')
    plt.figure()
    for i in range(output_dim):
        plt.plot(fpr[i], tpr[i], lw=3, linestyle='dotted',
                label='{0} (AUROC = {1:0.2f})'
                ''.format(ohe.categories_[0][i], roc_auc[i]))
    plt.plot(fpr["micro"], tpr["micro"],
            label='Average (AUROC = {0:0.2f})'
            ''.format(roc_auc["micro"]), linewidth=3)

    plt.plot([0, 1], [0, 1], color='lightgrey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Swarm Learning')
    plt.legend(loc="lower left", bbox_to_anchor=(1,0))
    plt.savefig(os.path.join(scratchDir, 'roc_curve.png'), bbox_inches="tight")
    
    # Classification report
    print(classification_report(y_test_labels, y_pred_labels))
    
    # Save model and weights
    saved_model_path = os.path.join(scratchDir, model_name, 'saved_model.pt')
    # Pytorch model save function expects the directory to be created before hand.
    os.makedirs(scratchDir, exist_ok=True)
    os.makedirs(os.path.join(scratchDir, model_name), exist_ok=True)
    torch.save(model, saved_model_path)
    print('Saved the trained model!')
     
if __name__ == '__main__':
  main()