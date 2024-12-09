from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

kirp_original_dir = '/mnt/swarm_beta/sbm_evaluation/pathology_data/mini_KIRP'
#/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRC/image_grid_4000
#/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRC/image_grid_58000
#/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRC/image_grid_330000
#/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRP/image_grid_4000
#/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRP/image_grid_66000
#/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRP/image_grid_330000
kirc_generated_dir_early = '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRC/image_grid_4000'
kirc_generated_dir_mid = '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRC/image_grid_58000'
kirc_generated_dir_late = '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRC/image_grid_330000'
kirp_generated_dir_early = '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRP/image_grid_4000'
kirp_generated_dir_mid = '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRP/image_grid_66000'
kirp_generated_dir_late = '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRP/image_grid_330000'
#Each of the folders contain 16 jpgs, do pca and plot them together in one plot
import numpy as np
import os
# Helper function to simulate loading images and flatten them
def simulate_load_and_flatten_images(num_images, image_size=(64, 64)):
    images = []
    for _ in range(num_images):
        img_array = np.random.rand(*image_size).flatten()
        images.append(img_array)
    return np.array(images)

# Simulate loading images from all directories
kirp_original = simulate_load_and_flatten_images(16)
kirc_generated_early = simulate_load_and_flatten_images(16)
kirc_generated_mid = simulate_load_and_flatten_images(16)
kirc_generated_late = simulate_load_and_flatten_images(16)
kirp_generated_early = simulate_load_and_flatten_images(16)
kirp_generated_mid = simulate_load_and_flatten_images(16)
kirp_generated_late = simulate_load_and_flatten_images(16)

# Combine all images into a single array
all_images = np.concatenate([
    kirp_original,
    kirc_generated_early,
    kirc_generated_mid,
    kirc_generated_late,
    kirp_generated_early,
    kirp_generated_mid,
    kirp_generated_late
])

# Standardize the data
scaler = StandardScaler()
all_images_scaled = scaler.fit_transform(all_images)

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_images_scaled)

# Assign labels for plotting
labels = ['KIRP Original'] * len(kirp_original) + \
         ['KIRC Early'] * len(kirc_generated_early) + \
         ['KIRC Mid'] * len(kirc_generated_mid) + \
         ['KIRC Late'] * len(kirc_generated_late) + \
         ['KIRP Early'] * len(kirp_generated_early) + \
         ['KIRP Mid'] * len(kirp_generated_mid) + \
         ['KIRP Late'] * len(kirp_generated_late)

# Plotting
plt.figure(figsize=(14, 10))
for label in np.unique(labels):
    indices = np.where(np.array(labels) == label)
    plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=label)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Pathology Images')
plt.legend()
plt.show()