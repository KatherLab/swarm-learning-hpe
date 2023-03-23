import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
import os
import time
from swarmlearning.pyt import SwarmCallback


default_max_epochs = 5
default_min_peers = 2 
trainPrint = True
swSyncInterval = 128 

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out
        
def loadData():
    ### Part to be adjusted for different data START
    path = os.path.join(dataDir,'data_bce.csv')
    data_raw = pd.read_csv(path)
    
    data = pd.get_dummies(data_raw.iloc[: , :-1])
    data = data.drop('diagnosis_B', axis=1)
    
    target = 'diagnosis_M'
    ### Part to be adjusted for different data END

    X = data.drop(target, axis=1)
    X = StandardScaler().fit_transform(X)
    y = data[target]
    
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)    
        
    # transform numpy to torch.Tensor
    xTrain, yTrain, xTest, yTest = map(torch.tensor, (#xTrain.to_numpy().astype(np.float32), 
                                                      xTrain.astype(np.float32),
                                                      yTrain.to_numpy().astype(np.int_), 
                                                      #xTest.to_numpy().astype(np.float32),
                                                      xTest.astype(np.float32),
                                                      yTest.to_numpy().astype(np.int_)))    
    # convert torch.Tensor to a dataset
    yTrain = yTrain.type(torch.LongTensor)
    yTest = yTest.type(torch.LongTensor)
    trainDs = torch.utils.data.TensorDataset(xTrain,yTrain)
    testDs = torch.utils.data.TensorDataset(xTest,yTest)
    
    # get model dimensions
    global input_dim
    input_dim = X.shape[1]
    
    return trainDs, testDs
    
def doTrainBatch(model,device,trainLoader,optimizer,epoch,max_epochs):
    model.train()
    for batchIdx, (data, target) in enumerate(trainLoader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #target = target.unsqueeze(1).float()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if trainPrint and batchIdx % 100 == 0:
            print('Train Epoch: {}/{} ({:.0f}%)\tLoss: {:.6f}'.format(
                  epoch, max_epochs, epoch/max_epochs * 100, loss.item()))
        # Swarm Learning Interface
        if swarmCallback is not None:
            swarmCallback.on_batch_end()

def test(model, device, testLoader):
    model.eval()
    testLoss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #target = target.unsqueeze(1).float()
            testLoss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    testLoss /= len(testLoader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        testLoss, correct, len(testLoader.dataset),
        100. * correct / len(testLoader.dataset)))

def main():
    dataDir = os.getenv('DATA_DIR', '/platform/data')
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    modelDir = os.getenv('MODEL_DIR', '/platform/model')
    max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
    min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
    batchSz = 128
    trainDs, testDs = loadData(dataDir)
    useCuda = torch.cuda.is_available()
    
    if useCuda:
        print("Cuda is accessable")
    else:
        print("Cuda is not accessable")
        
    device = torch.device("cuda" if useCuda else "cpu")  
    model = MLP(input_dim).to(device)
    model_name = 'mlp_table'
    opt = optim.Adam(model.parameters())
    trainLoader = torch.utils.data.DataLoader(trainDs,batch_size=batchSz)
    testLoader = torch.utils.data.DataLoader(testDs,batch_size=batchSz)
    
    # Create Swarm callback
    swarmCallback = None
    swarmCallback = SwarmCallback(syncFrequency=swSyncInterval,
                                  minPeers=min_peers,
                                  useAdaptiveSync=False,
                                  adsValData=testDs,
                                  adsValBatchSize=batchSz,
                                  model=model)
    # initalize swarmCallback and do first sync 
    swarmCallback.on_train_begin()
        
    for epoch in range(1, max_epochs + 1):
        doTrainBatch(model,device,trainLoader,opt,epoch,swarmCallback)      
        test(model,device,testLoader)
        swarmCallback.on_epoch_end(epoch)

    # handles what to do when training ends        
    swarmCallback.on_train_end()

    # Save model and weights
    saved_model_path = os.path.join(scratchDir, model_name, 'saved_model.pt')
    # Pytorch model save function expects the directory to be created before hand.
    os.makedirs(scratchDir, exist_ok=True)
    os.makedirs(os.path.join(scratchDir, model_name), exist_ok=True)
    torch.save(model, saved_model_path)
    print('Saved the trained model!')
 
if __name__ == '__main__':
  main()