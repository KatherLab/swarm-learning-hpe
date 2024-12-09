import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Function to load features from a directory
def load_features_from_directory(directory):
    feature_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
    features = [np.load(file) for file in feature_files]
    return np.array(features)

# Function to generate color map
def generate_color_map(base_color, num_shades):
    cmap = cm.get_cmap(base_color, num_shades)
    color_map = [mcolors.to_hex(cmap(i)) for i in range(num_shades)]
    return color_map

# Function to perform PCA and plot results with convex hulls for specific labels
def perform_pca_and_plot(directories, labels):
    all_features = []
    all_labels = []

    for directory, label in zip(directories, labels):
        features = load_features_from_directory(directory)
        all_features.append(features)
        all_labels.extend([label] * len(features))

    # Combine all features into a single array
    all_features_combined = np.vstack(all_features)

    # Standardize the data
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features_combined)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_features_scaled)

    # Define color schemes
    unique_labels = np.unique(all_labels)
    num_shades_kirp = sum('KIRP' in label for label in unique_labels) - 1  # Exclude 'KIRP Original'
    num_shades_kirc = sum('KIRC' in label for label in unique_labels) - 1  # Exclude 'KIRC Original'

    kirp_colors = generate_color_map('Blues', num_shades_kirp)
    kirc_colors = generate_color_map('Reds', num_shades_kirc)

    kirp_label_index = 0
    kirc_label_index = 0

    color_dict = {}
    for label in unique_labels:
        if label == 'KIRP Original':
            color_dict[label] = 'blue'
        elif label == 'KIRC Original':
            color_dict[label] = 'red'
        elif 'KIRP' in label:
            color_dict[label] = kirp_colors[kirp_label_index]
            kirp_label_index += 1
        else:
            color_dict[label] = kirc_colors[kirc_label_index]
            kirc_label_index += 1

    # Plotting
    plt.figure(figsize=(14, 10))
    for label in unique_labels:
        indices = np.where(np.array(all_labels) == label)
        if label in ['KIRP Original', 'KIRC Original']:
            hull = ConvexHull(pca_result[indices])
            for simplex in hull.simplices:
                plt.plot(pca_result[indices][simplex, 0], pca_result[indices][simplex, 1], 'k-')
            plt.fill(pca_result[indices][hull.vertices, 0], pca_result[indices][hull.vertices, 1], alpha=0.3,
                     label=label, color=color_dict[label])
        else:
            plt.scatter(pca_result[indices, 0], pca_result[indices, 1], color=color_dict[label], alpha=0.7)

    # Custom legend
    plt.scatter([], [], color='blue', alpha=0.7, label='KIRP samples')
    plt.scatter([], [], color='red', alpha=0.7, label='KIRC samples')
    plt.scatter([], [], color='blue', alpha=1.0, label='KIRP Original')
    plt.scatter([], [], color='red', alpha=1.0, label='KIRC Original')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Pathology Features')
    plt.legend()
    plt.show()

# Generate directories and labels dynamically
base_dir_kirp = '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRP_ctrans'
base_dir_kirc = '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRC_ctrans'
directories = []
labels = []

# Fetch directories and create labels for KIRP
for folder in sorted(os.listdir(base_dir_kirp)):
    directories.append(os.path.join(base_dir_kirp, folder))
    labels.append(f'KIRP {folder}')

# Fetch directories and create labels for KIRC
for folder in sorted(os.listdir(base_dir_kirc)):
    directories.append(os.path.join(base_dir_kirc, folder))
    labels.append(f'KIRC {folder}')

# Add original directories if needed
directories.extend([
    '/mnt/swarm_beta/sbm_evaluation/pathology_data/train_ctrans/KIRP',
    '/mnt/swarm_beta/sbm_evaluation/pathology_data/train_ctrans/KIRC'
])
labels.extend([
    'KIRP Original',
    'KIRC Original'
])

# Perform PCA and plot
perform_pca_and_plot(directories, labels)
