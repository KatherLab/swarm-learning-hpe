# Import Libraries
import sys
import os.path
import datetime
import itertools
import argparse
from shutil import copy2
import random
import logging
import Networks
import Dataloader
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models.resnet
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# Swarm Learning
from swarmlearning.pyt import SwarmCallback
import time

def main():
    # Assume that we are on a CUDA machine, then this should print a CUDA device
    debug = False

    # Define experiments
    skip = [2] # 100 Frames 
    temporal = [True]
    middleframe = [False]
    binary = True
      
    # Generate all combinations
    all_combinations = itertools.product(middleframe, skip, temporal)

    # Filter combinations to use either skip or middleframe, but not both
    filtered_combinations = [(m, s, t) for m, s, t in all_combinations if (s == 0 or not m)]

    # Define hyperparameters
    if binary:
        num_class = 2
    else:
        num_class = 6
    width = 480
    height = 270
    batch_size = 64
    epochs = 64
    lstm_size = 160
    num_workers = 6

    # Swarm Learning parameters
    default_min_peers = 2
    default_syncFrequency = 100
    min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
    syncFrequency = int(os.getenv('SYNC_FREQUENCY', str(default_syncFrequency)))

    # Iterate through combinations 
    for middleframe, skip, temporal in tqdm(filtered_combinations):
        # Define trial name and define directories
        # Get directory information from swarm learning platform
        dataDir = os.getenv('DATA_DIR', '/platform/data')
        scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')   
    
        frames = 200 if skip == 0 else 200 / skip
        trial_name = f"bi_Appendectomy_Classification_swarm_temp{temporal}_mid{middleframe}_{frames}frames"

        data_folder = dataDir
        
        output_folder = trial_name + datetime.datetime.now().strftime("%Y%m%d-%H%M") + "/"
        output_folder = os.path.join(scratchDir, output_folder)
        os.makedirs(output_folder)

        if os.path.exists(output_folder):
            print("Directory created successfully:", output_folder)
        else:
            print("Failed to create directory:", output_folder)

        # SetUp logging
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(output_folder, f'exp_{timestamp}.log')
        logging.basicConfig(
        filename=log_filename,  # Log file name
        filemode='w',        
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        level=logging.DEBUG  # Log level
        )

        def logprint(message):
            print(message)
            logging.info(message)   

        # Print experiment settings
        logprint(f"Center: swarm, Frames: {frames}, Temporal: {temporal}, Midlleframe: {middleframe}")

        # Set Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_cpu = torch.device("cpu")
        logprint(f"Using {device}")

        # Define normalisation and transformation
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose(
            [transforms.AutoAugment(),
            transforms.ToTensor(),
            normalize])
        transform_val = transforms.Compose(
            [transforms.ToTensor(),
            normalize])
        
        # Initialise weights
        weights = np.zeros(num_class, dtype=np.float32)

        # Initialise train and val sets
        train_sets = []
        val_sets = []

        # Load Data
        logprint("Starting data loading...")
        
        op_path = os.path.join(data_folder, "train")
        subfolders_train = [f for f in os.listdir(op_path) if os.path.isdir(os.path.join(op_path, f))]

        train_subfolders, val_subfolders = train_test_split(subfolders_train, test_size=0.2, random_state=42)

        for id in train_subfolders:
            logprint(f"Processing training folder: {op_path}, ID: {id}")
            if debug and len(train_sets) > 3:
                continue
            
            dataset = Dataloader.AppendectomyDataset(op_path, str(id), width, height, transform_train, middleframe, skip, binary=binary)
            weights[dataset.lbl] += 1
            if temporal:
                train_sets.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers))
            else:
                train_sets.append(dataset)

        for id in val_subfolders:
            logprint(f"Processing validation folder: {op_path}, ID: {id}")
            if debug and len(val_sets) > 3:
                continue
            
            dataset = Dataloader.AppendectomyDataset(op_path, str(id), width, height, transform_val, not temporal, skip, binary=binary)
            if temporal:
                val_sets.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers))
            else:
                val_sets.append(dataset)

        if not temporal:
            train_set = torch.utils.data.ConcatDataset(train_sets)
            val_set = torch.utils.data.ConcatDataset(val_sets)
            
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
            train_sets = [train_loader]
            val_sets = [val_loader]

        # Update weights
        n_samples = np.sum(weights)
        weights = n_samples/(num_class*weights)
        weights[weights == np.inf] = 0.1
        logprint(f"Weights: {weights}")

        # Define neural network
        net = Networks.PhaseLSTMConvNext(num_class, temporal, lstm_size)
        logprint("Using PhaseLSTMConvNext")

        for param in net.model.stem.parameters():
            param.requires_grad = False
        for param in net.model.stages[0].parameters():
            param.requires_grad = False
        for param in net.model.stages[1].parameters():
            param.requires_grad = False
        for param in net.model.stages[2].parameters():
            param.requires_grad = False
        for param in net.model.stages[3].parameters():
            param.requires_grad = False
        
        net.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weights).to(device))
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)

        # Initialise best F1 score
        best_f1 = 0

        # Create Swarm callback
        swarmCallback = None
        swarmCallback = SwarmCallback(syncFrequency=syncFrequency,
                                    minPeers=min_peers,
                                    useAdaptiveSync=False,
                                    adsValData=val_sets,
                                    adsValBatchSize=batch_size,
                                    model=net)

        # initalize swarmCallback and do first sync
        swarmCallback.on_train_begin()

        # Model traing loop
        logprint("Starting training...")
        for epoch in range(epochs):

            train_loss = 0
            val_loss = 0
            train_accuracy = 0
            val_accuracy = 0
            train_count = 0
            val_count = 0

            train_metrics = []
            val_metrics = []

            for i in range(num_class):
                train_metrics.append([0,0,0])
                val_metrics.append([0,0,0])

            random.shuffle(train_sets)
            for op in train_sets:

                optimizer.zero_grad()
                num_batches = len(op)
                logprint(f"Number of Batches: {num_batches}")
                optimized = False
                batch = 0
                loss = 0

                hidden_state = None

                for data in op:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)

                    if temporal:
                        pred, hidden_state = net(images, hidden_state)
                        # Detach hidden state after EVERY forward pass
                        if hidden_state is not None:
                            h, c = hidden_state
                            hidden_state = (h.detach(), c.detach())
                    else:
                        pred = net(images)
                    
                    if temporal:
                        mid = pred.shape[0]//2
                        batch_loss = criterion(pred[mid:], labels[mid:])
                    else:
                        batch_loss = criterion(pred, labels)

                    # Scale the loss by the number of accumulation steps
                    batch_loss = batch_loss / 3
                    batch_loss.backward()
                    
                    loss += batch_loss.item() * 3  # For logging purposes
                    
                    batch += 1
                    if batch % 3 == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        optimized = True
                    else:
                        optimized = False

                    train_loss += loss
                    predicted = torch.argmax(pred, dim=1)
                    if temporal:
                        predicted = predicted[-1:]
                    train_count += labels.size(0)
                    
                    for i in range(predicted.shape[0]):
                        lbl = int(labels[i].item())
                        prd = int(predicted[i].item())

                        if lbl == prd:
                            train_metrics[lbl][0] += 1
                        else:
                            train_metrics[lbl][1] += 1
                            train_metrics[prd][2] += 1

                    if batch % 3 == 0:
                        loss = 0

                    if swarmCallback is not None:
                        swarmCallback.on_batch_end()

                if not optimized:
                    optimizer.step()

            with torch.no_grad():
                for op in val_sets:
                    num_batches = len(op)
                    batch = 0
                    loss = 0

                    hidden_state = None

                    for data in op:
                        images, labels = data
                        images = images.to(device)

                        labels = labels.to(device)
            
                        if temporal:
                            pred, hidden_state = net(images, hidden_state)
                        else:
                            pred = net(images)

                        if temporal:
                            mid = pred.shape[0]//2
                            loss += criterion(pred[mid:], labels[mid:])
                        else:
                            loss += criterion(pred, labels)

                        val_loss += loss.item()
                        
                        predicted = torch.argmax(pred, dim=1)

                        if temporal:
                            predicted = predicted[-1:]
                        val_count += labels.size(0)

                        for i in range(predicted.shape[0]):
                            lbl = int(labels[i].item())
                            prd = int(predicted[i].item())

                            if lbl == prd:
                                val_metrics[lbl][0] += 1
                            else:
                                val_metrics[lbl][1] += 1
                                val_metrics[prd][2] += 1
                            
            f1_train = []
            f1_val = []

            for i in range(num_class):
                f1_t = 0
                f1_v = 0

                if 2*train_metrics[i][0] + train_metrics[i][1] + train_metrics[i][2] > 0:
                    f1_t = (2*train_metrics[i][0])/(2*train_metrics[i][0] + train_metrics[i][1] + train_metrics[i][2])
                    f1_train.append(f1_t)
                    
                
                if 2*val_metrics[i][0] + val_metrics[i][1] + val_metrics[i][2] > 0:
                    f1_v = (2*val_metrics[i][0])/(2*val_metrics[i][0] + val_metrics[i][1] + val_metrics[i][2])
                    f1_val.append(f1_v)
                p_m = i,f1_t, train_metrics[i][0] + train_metrics[i][1], f1_v, val_metrics[i][0] + val_metrics[i][1]
                logprint(f"{p_m}")

            f1_train = np.nanmean(f1_train)
            f1_val = np.nanmean(f1_val)


            mes = "Epoche %3d: Train (loss %.3f, f1 %.3f) Val (loss %.3f, f1 %.3f prev best f1 %.3f)" % (epoch + 1, train_loss/train_count, f1_train, val_loss/val_count, f1_val, best_f1)
            logprint(mes)

            if f1_val > best_f1:
                net.save(output_folder + "best.pkl")
                best_f1 = f1_val
            
            torch.cuda.empty_cache()
        
        # Handles what to do when training ends
        swarmCallback.on_train_end()
        
        net.save(output_folder + "last.pkl")

        # Close and remove handlers after each combination
        logger = logging.getLogger()
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
            

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")    
