#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"
data = {
    'ViT-MIL': {'Localhost1 - 40% data': [0.63, 0.72, 0.68, 0.68, 0.64], 'Localhost2 - 30% data': [0.67, 0.71, 0.68, 0.64, 0.63],
                   'Localhost3 - 10% data': [0.52, 0.52, 0.46, 0.62, 0.54, 0.62],
                   'SL local best ckpt': [0.75, 0.77, 0.76, 0.75,0.72],
                   'SL global best ckpt': [0.76, 0.74, 0.71, 0.75, 0.74, 0.75, 0.73], 'Local merged - 100% data': [0.69, 0.7, 0.72, 0.58, 0.78, 0.76, 0.75, 0.72, 0.76, 0.75]},
    'ViT-LSTM-MIL': {'Localhost1 - 40% data': [0.70, 0.69, 0.70, 0.72, 0.71], 'Localhost2 - 30% data': [0.66, 0.68, 0.72, 0.69, 0.69],
                 'Localhost3 - 10% data': [0.52, 0.46, 0.55, 0.50, 0.57], 'SL local best ckpt': [0.75, 0.75, 0.76, 0.73, 0.75],
                 'SL global best ckpt': [0.76, 0.74, 0.75, 0.75, 0.75, 0.74], 'Local merged - 100% data': [0.75, 0.75, 0.74, 0.76, 0.76, 0.76, 0.76, 0.74, 0.74, 0.77, 0.77, 0.77]},
    'Att-MIL': {'Localhost1 - 40% data': [0.71, 0.74, 0.69,0.71,0.72], 'Localhost2 - 30% data': [0.62, 0.62,0.58,0.66,0.54],
                'Localhost3 - 10% data': [0.46, 0.51,0.55,0.5,0.48], 'SL local best ckpt': [0.66, 0.66, 0.66,0.71,],
                'SL global best ckpt': [0.7, 0.71, 0.66, 0.69, 0.49], 'Local merged - 100% data':  [0.69, 0.69, 0.68, 0.66, 0.65]},
    'ENb4-2D': {'Localhost1 - 40% data': [0.54,0.60], 'Localhost2 - 30% data': [0.52,0.59],
                   'Localhost3 - 10% data': [0.52, 0.52, 0.46],'SL local best ckpt': [0.47, 0.55], 'SL global best ckpt': [0.23, 0.25], 'Local merged - 100% data': [0.6, 0.47, 0.56, 0.52, 0.52]},
'ENb7-2D': {'Local merged - 100% data': [0.53, 0.51, 0.74, 0.58, 0.71, 0.59, 0.45, 0.72, 0.77, 0.71, 0.63, 0.69]},
    'ResNet50-2D' : {'Localhost1 - 40% data': [0.62,0.61,0.63,0.61], 'Localhost2 - 30% data': [0.60,0.59,0.58,0.59],
                   'Localhost3 - 10% data': [0.52, 0.52, 0.46, 0.52, 0.54, 0.52],
                   'SL local best ckpt': [0.60,0.60,0.61,0.62],
                   'SL global best ckpt': [0.60,0.60,0.61,0.62], 'Local merged - 100% data': [0.59,0.58,0.60,0.60]},
'ENb7-3D': {'Local merged - 100% data': [0.56, 0.53, 0.56, 0.54, 0.53, 0.63, 0.47, 0.6, 0.53, 0.43, 0.5, 0.51]},

    'ResNet18-3D': {'Localhost1 - 40% data': [0.69, 0.71, 0.72, 0.68, 0.67], 'Localhost2 - 30% data': [0.77, 0.76, 0.73, 0.75],
                    'Localhost3 - 10% data': [0.67, 0.64, 0.67,0.69,0.63], 'SL local best ckpt': [0.82, 0.8, 0.79],
                    'SL global best ckpt': [0.79, 0.78, 0.78],
                    'Local merged - 100% data': [0.8, 0.82, 0.77, 0.73, 0.79, 0.81, 0.79]},
    'ResNet50-3D': {'Localhost1 - 40% data': [0.69, 0.72, 0.55, 0.49, 0.72], 'Localhost2 - 30% data': [0.69, 0.80],
                    'Localhost3 - 10% data': [0.63, 0.68], 'SL local best ckpt': [0.78, 0.83, 0.76, 0.82],
                    'SL global best ckpt': [0.68],
                    'Local merged - 100% data': [0.85, 0.8, 0.81, 0.77, 0.84, 0.79, 0.85, 0.87]},
    'ResNet101-3D': {'Localhost1 - 40% data': [0.8, 0.74], 'Localhost2 - 30% data': [0.37, 0.41],
                     'Localhost3 - 10% data': [0.56, 0.51], 'SL local best ckpt': [0.82, 0.82, 0.83],
                     'SL global best ckpt': [0.86, 0.78, 0.77], 'Local merged - 100% data':  [0.82, 0.81, 0.83, 0.82, 0.84, 0.83, 0.78]},
    'ResNet152-3D': {'Local merged - 100% data': [0.84, 0.81, 0.86, 0.8]},
    'DenseNet121-3D': {'Localhost1 - 40% data': [0.47, 0.79, 0.72], 'Localhost3 - 10% data': [0.62, 0.67],
                   'SL local best ckpt': [0.71, 0.71, 0.78, 0.79], 'SL global best ckpt': [0.73, 0.52, ], 'Local merged - 100% data': [0.75, 0.72, 0.68, 0.68, 0.69, 0.75, 0.72, 0.67, 0.78, 0.71, 0.68]},
    'UNet3D' : {'Local merged - 100% data': [0.66, 0.69, 0.63]},
}
# Prepare data for the box plot
data_list = []
for model, values in data.items():
    for category, auc_roc_values in values.items():
        for value in auc_roc_values:
            data_list.append([model, category, value])

df = pd.DataFrame(data_list, columns=['Model', 'Category', 'AUC-ROC'])

# Create the box plot
plt.figure(figsize=(14, 6))
sns.boxplot(x='Model', y='AUC-ROC', hue='Category', data=df)

# Set labels
plt.xlabel("Models")
plt.ylabel("AUC-ROC Values")

# Show the plot
plt.tight_layout()
plt.show()