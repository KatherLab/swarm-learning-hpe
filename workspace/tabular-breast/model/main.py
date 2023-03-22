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
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(31, 16)
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
    data_raw = pd.read_csv('data_bcw.csv')
    data = pd.get_dummies(data_raw.iloc[: , :-1])
    data = data.drop('diagnosis_B', axis=1)
    
    target = 'diagnosis_M'

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
    max_epochs = 2000
    batchSz = 20
    useCuda = torch.cuda.is_available()
    device = torch.device("cuda" if useCuda else "cpu")  
    model = MLP().to(device)
    opt = optim.Adam(model.parameters())
    trainDs, testDs = loadData()
    trainLoader = torch.utils.data.DataLoader(trainDs,batch_size=batchSz)
    testLoader = torch.utils.data.DataLoader(testDs,batch_size=batchSz)
        
    for epoch in range(1, max_epochs + 1):
        doTrainBatch(model,device,trainLoader,opt,epoch,max_epochs)      
        test(model,device,testLoader)

main()