import os
from PIL import Image
import imagehash
import numpy as np
import matplotlib.pyplot as plt

def calculate_phash(image_path):
 image = Image.open(image_path)
 return imagehash.phash(image)

def phash_distribution(folder):
 phashes = []
 for filename in os.listdir(folder):
     if filename.endswith(('.jpg', '.jpeg', '.png')):
         phash = calculate_phash(os.path.join(folder, filename))
         phashes.append(np.array(phash.hash).astype(np.float32).flatten())  # Flatten the hash to a 1D array
 return np.array(phashes)

folder1 = '/mnt/swarm_beta/sbm_evaluation/pathology_data/KIRC'
folder2 = '/mnt/swarm_beta/sbm_evaluation/pathology_data/KIRP'

phash_real = phash_distribution(folder1)
print(phash_real)
# save the phash_real to a file
np.save('phash_KIRC_real.npy', phash_real)

phash_synthetic = phash_distribution(folder2)
print(phash_synthetic)
# save the phash_synthetic to a file
np.save('phash_KIRP_real.npy', phash_synthetic)

# Plotting the distribution of pHashes for both folders
plt.hist(phash_real.flatten(), bins=256, alpha=0.5, label='Real Images')
plt.hist(phash_synthetic.flatten(), bins=256, alpha=0.5, label='Synthetic Images')
plt.legend(loc='upper right')
plt.show()
