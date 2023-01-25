import warnings
import argparse
import torch

import utils as utils
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from train_function import doTrainBatch, test
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from dataclasses import dataclass
import torchvision.models as models
import pandas as pd
from swarmlearning.pyt import SwarmCallback
import logging
import torchvision
from data_utils import ConcatCohorts_Classic, DatasetLoader_Classic, LoadTrainTestFromFolders, GetTiles
#Getting all the arguments from the experement file
parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
exp_name = 'saved_model'
parser.add_argument('--adressExp', type = str, default = "/tmp/test/model/exp_A.txt", help = 'Address to the experiment File')
args = parser.parse_args() 

#setting the syn interval to the swarm learning
swSyncInterval = 4

#Check for the CUDA and if available assign it
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
warnings.filterwarnings("ignore")
args = utils.ReadExperimentFile(args)
args.useCsv = True

# Define the model to be used
class mnistNet(nn.Module):
    def __init__(self):
        super(mnistNet, self).__init__()
        self.dense = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.2)
        self.dense1 = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, 256)
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 2)
    def forward(self, x):
        #x = torch.flatten(x, 1)        
        x = self.dense(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.dense1(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.dense2(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.dense3(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.dense4(x)
        output = x
        return output

#Reading all the arguments in the experiment file
stats_total = {}
stats_df = pd.DataFrame()
args.target_label = args.target_labels[0]
targetLabel=args.target_labels

#path to the features and lables
dataDir = os.getenv('DATA_DIR', '/platform/data') #!
#feature_and_label_path = os.path.join(dataDir, r"Data_features/"+str(targetLabel[0])+"/Features/Data_set_A_"+str(targetLabel[0])+"/feature_lables") #!
print('###############################################################')
#Check for the CUDA and if available assign it
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
warnings.filterwarnings("ignore")
args = utils.ReadExperimentFile(args)
print(args)
args.useCsv = True

# Define the model to be used
model_res = models.resnet18(pretrained=True)

#Reading all the arguments in the experiment file
stats_total = {}
stats_df = pd.DataFrame()
args.target_label = 'ER Status By IHC'
targetLabel=args.target_labels
dataDir = os.getenv('DATA_DIR', '/platform/data')

#path to the features and lables
args.projectFolder = os.path.join(dataDir,r"project")
print('###############################################################')
#args.projectFolder = '/home/swarm/Desktop/oliver_swarm_code_test/test_proj_dir'
args.datadir_train= [os.path.join(dataDir,r"BLOCKS_NORM_MACENKO")]
args.clini_dir = [os.path.join(dataDir,r"tabels/TCGA-BRCA-A2-CLINI.xlsx")]
args.slide_dir = [os.path.join(dataDir,r"tabels/TCGA-BRCA-A2_SLIDE.xlsx")]

#os.mkdir()
# Create a report file to put in all the enteries
reportFile  = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
reportFile.write('**********************************************************************'+ '\n')
reportFile.write(str(args))
reportFile.write('\n' + '**********************************************************************'+ '\n')


patientsList, labelsList, slidesList, clinicalTableList, slideTableList = ConcatCohorts_Classic(imagesPath = args.datadir_train, cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir, label = args.target_label, reportFile = reportFile)
print('\nLOAD THE DATASET FOR TRAINING...\n')  
final_patient_list = patientsList
values, counts = np.unique(labelsList, return_counts=True)

le = preprocessing.LabelEncoder()
labelsList = le.fit_transform(labelsList)

args.num_classes = len(set(labelsList))
args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))        

utils.Summarize_Classic(args, list(labelsList), reportFile)
   
print(patientsList)
print('IT IS A train test split of 90 10!')
print('USE PRE SELECTED TILES')
patientID = np.array(patientsList)
labels = np.array(labelsList)
print(patientID)
args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
os.makedirs(args.split_dir, exist_ok = True)
args.feature_label_dir = os.path.join(args.projectFolder, 'feature_lables')
os.makedirs(args.feature_label_dir, exist_ok = True)

counter = 0


trainData_patientID , testData_patientID, trainData_labels , testData_Labels  = train_test_split(patientID, labels, test_size=0.001)
print(testData_patientID)
train_index=[]
test_index=[]
for i in range(len(trainData_patientID)):  
    a =  patientID.tolist().index(trainData_patientID[i])
    train_index.append(a)
    print(train_index)
for j in range(len(testData_patientID)):  
    b =  patientID.tolist().index(testData_patientID[j])
    test_index.append(b)
    print(test_index)


#print(kf.split(patientID, labels).shape)
#train_index, test_index = .split(patientID, labels)
print('GENERATE NEW TILES')
#train_index = [i for i in train_index if i not in val_index]

counter = 0
print('\nLOAD TRAIN DATASET\n')

train_data = GetTiles(patients = trainData_patientID, labels = trainData_labels, imgsList = slidesList, label = args.target_label, slideTableList = slideTableList, maxBlockNum = args.maxBlockNum, test = False, seed = args.seed)

train_x = list(train_data['tileAd'])
train_y = list(train_data[args.target_label])

df = pd.DataFrame(list(zip(train_x, train_y)), columns =['X', 'y'])
print(df)

df.to_csv(os.path.join(args.split_dir, 'SPLIT_TRAIN_' + str(counter)+ '.csv'), index = False)
path_train = os.path.join(args.split_dir, 'SPLIT_TRAIN_' + str(counter)+ '.csv')
model = models.resnet18(pretrained=True).to(device)

test_data = GetTiles(patients = testData_patientID, labels = testData_Labels, imgsList = slidesList, label = args.target_label,
                  slideTableList = slideTableList, maxBlockNum = args.maxBlockNum, test = True, seed = args.seed)  

test_x = list(test_data['tileAd'])
test_y = list(test_data[args.target_label])
test_pid = list(test_data['patientID'])

df = pd.DataFrame(list(zip(test_pid, test_x, test_y)), columns = ['pid', 'X', 'y'])
#print(df)
df.to_csv(os.path.join(args.split_dir, 'SPLIT_TEST_' + str(counter) + '.csv'), index = False)

path_test = os.path.join(args.split_dir, 'SPLIT_TEST_' + str(counter) + '.csv')
# Check if the file  having featurs and lables already exists if not extract the features
batchSz = args.batch_size
#trainDs, testDs = loadData(feature_lable_path)
model_ft = models.resnet18(pretrained=True).to(device)
input_size = 512
params = {'batch_size': args.batch_size,
                          'shuffle': True,
                          'num_workers': 0,
                          'pin_memory' : False}
train_set =DatasetLoader_Classic(train_x, train_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
trainLoader= torch.utils.data.DataLoader(train_set, **params)
test_set = DatasetLoader_Classic(test_x, test_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
testLoader = torch.utils.data.DataLoader(test_set, **params)



print(model_ft)
print('\nINIT OPTIMIZER ...', end = ' ')
optimizer = utils.get_optim(model_ft, args, params = False)
print('DONE!')
criterion = nn.CrossEntropyLoss()


print('\nSTART TRAINING ...', end = ' ')
model_ft = models.resnet18(pretrained=True).to(device)
input_size = 512
params = {'batch_size': args.batch_size,
                          'shuffle': True,
                          'num_workers': 0,
                          'pin_memory' : False}
# get the bath size from the expirement file
max_epochs = args.max_epochs
batchSz = 124
model_name = 'resnet_histo'
swarmCallback = None
default_max_epochs = 4

# provide a sawrm call back
swarmCallback = SwarmCallback(syncFrequency=swSyncInterval,
                          minPeers=2,
                          adsValData=testLoader,
                          adsValBatchSize=batchSz,
                          #node_weightage = 63,
                          model=model_ft,)
                  
swarmCallback.logger.setLevel(logging.DEBUG)
                          
# initalize swarmCallback and do first sync
swarmCallback.on_train_begin()

for epoch in range(1, max_epochs + 1):
        trainedModel = doTrainBatch(model = model_ft,device = device, trainLoader = trainLoader, optimizer = optimizer, epoch = epoch, swarmCallback=swarmCallback)      
        test(model = trainedModel ,device = device, testLoader = testLoader)
        swarmCallback.on_epoch_end(epoch)
        
swarmCallback.on_train_end()

print('DONE!')

scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
saved_model_path = os.path.join(scratchDir, model_name, 'saved_model.pt')
os.makedirs(scratchDir, exist_ok=True)
os.makedirs(os.path.join(scratchDir, model_name), exist_ok=True)
torch.save(trainedModel.state_dict(), saved_model_path)
print('Saved the trained model!')
print(saved_model_path)


