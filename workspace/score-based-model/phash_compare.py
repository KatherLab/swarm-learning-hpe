import os
from PIL import Image
import imagehash
import numpy as np
from matplotlib import pyplot as plt


def calculate_phash(image_path):
 image = Image.open(image_path)
 return imagehash.phash(image)

def average_phash(folder):
 phashes = []
 for filename in os.listdir(folder):
     if filename.endswith(('.jpg', '.jpeg', '.png')):
         phash = calculate_phash(os.path.join(folder, filename))
         phashes.append(np.array(phash.hash).astype(np.float32))  # Convert to array and float for averaging
 average_phash = np.mean(phashes, axis=0)
 return average_phash

folder1 = '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRC/image_grid_348000'
folder2 = '/mnt/swarm_beta/sbm_evaluation/pathology_data/KIRC'

average_phash_real = average_phash(folder1)
average_phash_synthetic = average_phash(folder2)

# Compute the Hamming distance between the two average pHashes
difference = np.sum(average_phash_real != average_phash_synthetic)

if difference == 0:
    print("The average pHashes are identical.")
elif difference < 10:
    print("The average pHashes are similar.")
else:
    print("The average pHashes are different.")


print(f"Average pHash difference between folders: {difference}")

