from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch.utils.data as data
import csv
import scipy.ndimage.interpolation
import skimage.color
import skimage.transform
import matplotlib.pyplot as plt
import torch
import glob
import csv
import multiprocessing

lbl_sta = ["Histopathological stage of appendicitis", "Histopathologisches Stadium der Appendizitis"]
stage = [["Stage 0", 
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "N/A",],
        ["Stadium 0",
        "Stadium 1",
        "Stadium 2",
        "Stadium 3",
        "N/A"
        ]]


class AppendectomyDataset(data.Dataset):
    def __init__(self, data_folder, op, width, height, transform=None, middleframe=True, skip=0, binary=False):
        self.transform = transform
        manager = multiprocessing.Manager()
        self.images = manager.list()
        self.target = []
        self.image_path = os.path.join(data_folder, op)
        self.width = width
        self.height = height

        files = glob.glob(self.image_path + '/*.png')
        files.sort()

        

        if middleframe:
            middle = len(files)//2
            files = files[middle:middle+1]
        elif skip > 0:
            tmp = []

            for i in range(0, len(files), skip+1):
                tmp.append(files[i])
            
            files = tmp

        csv_files = glob.glob(data_folder + '/*.csv')
        
        self.lbl = -1

        assert len(csv_files) == 1
        
        with open(csv_files[0], newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')

            for row in csvreader:
                if op == row[1]:
                    print(op, row)
                    self.lbl = int(row[2])

                    break

        if binary:
            self.lbl =  0 if self.lbl in [0, 1, 2, 3] else 1 

        print(data_folder, self.lbl)
        assert(self.lbl > -1)
        
        for frame in files:
            print(frame)
            

            self.images.append((frame, None))
            self.target.append(self.lbl)

    def loadimage(self, frame):
        im = Image.open(frame)
        w = im.width
        h = im.height
        height2 = int(self.width * (h / w))

        offset_y = (self.height - height2) // 2

        img_y = im.resize((self.width, height2))
        if offset_y != 0:
            img = Image.new('RGB', (self.width, self.height), (0, 0, 0))
            img.paste(img_y, box=(0, offset_y))
        else:
            img = img_y

        return img

    def __getitem__(self, index):
        if self.images[index][1] is None:
            if index == 0:
                print("load first frame")
            img = self.loadimage(self.images[index][0])

            self.images[index] = (self.images[index][0], img)

        target = self.target[index]
        img = self.images[index][1]
        

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.target)
