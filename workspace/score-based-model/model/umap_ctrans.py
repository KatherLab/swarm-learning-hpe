import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from scipy.spatial import ConvexHull


# Function to load features from a directory
def load_features_from_directory(directory):
    feature_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
    features = [np.load(file) for file in feature_files]
    return np.array(features)


# Function to perform UMAP and plot results with convex hulls for specific labels
def perform_umap_and_plot(directories, labels):
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

    # Perform UMAP
    umap = UMAP(n_components=2, random_state=42)
    umap_result = umap.fit_transform(all_features_scaled)

    # Define color scheme
    color_dict = {
        'KIRP Original': 'blue',
        'KIRC Original': 'red',
        'KIRP sample Early': 'lightblue',
        'KIRC sample Early': 'lightcoral',
        'KIRP sample Middle': 'deepskyblue',
        'KIRC sample Middle': 'salmon',
        'KIRP sample Late': 'darkblue',
        'KIRC sample Late': 'darkred'
    }

    # Plotting
    plt.figure(figsize=(14, 10))
    for label in np.unique(all_labels):
        indices = np.where(np.array(all_labels) == label)
        if label in ['KIRP Original', 'KIRC Original']:
            hull = ConvexHull(umap_result[indices])
            for simplex in hull.simplices:
                plt.plot(umap_result[indices][simplex, 0], umap_result[indices][simplex, 1], 'k-')
            plt.fill(umap_result[indices][hull.vertices, 0], umap_result[indices][hull.vertices, 1], alpha=0.3,
                     label=label, color=color_dict[label])
        else:
            plt.scatter(umap_result[indices, 0], umap_result[indices, 1], label=label, color=color_dict[label])

    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP of Pathology Features')
    plt.legend()
    plt.show()


# Example usage
directories = [
    '/mnt/swarm_beta/sbm_evaluation/pathology_data/train_ctrans/KIRP',
    '/mnt/swarm_beta/sbm_evaluation/pathology_data/train_ctrans/KIRC',

    '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRP_ctrans/image_grid_2000',
    '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRC_ctrans/image_grid_2000',

    '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRP_ctrans/image_grid_10000',
    '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRC_ctrans/image_grid_10000',

    '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRP_ctrans/image_grid_352000',
    '/mnt/swarm_beta/sbm_evaluation/conditional_cut/KIRC_ctrans/image_grid_352000',
]

labels = [
    'KIRP Original',
    'KIRC Original',
    'KIRP sample Early',
    'KIRC sample Early',
    'KIRP sample Middle',
    'KIRC sample Middle',
    'KIRP sample Late',
    'KIRC sample Late',
]

perform_umap_and_plot(directories, labels)
