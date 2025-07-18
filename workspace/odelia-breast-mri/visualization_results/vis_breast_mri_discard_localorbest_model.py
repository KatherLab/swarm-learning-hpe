#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"
data = {
    'ResNet50-2D': {'Localhost1 - 40% data': [0.55, 0.57, 0.50, 0.60,0.55],
                    'Localhost2 - 30% data': [0.60, 0.59, 0.58, 0.59, 0.52],
                    'Localhost3 - 10% data': [0.52, 0.52, 0.52, 0.60,0.53],
                    'SL global best ckpt': [0.50, 0.60, 0.61, 0.62,0.56],
                    'Local merged - 100% data': [0.59, 0.58, 0.60, 0.50,0.60,0.55]},
    'Att-MIL': {'Localhost1 - 40% data': [0.57, 0.61,0.52,0.59,0.55], 'Localhost2 - 30% data': [0.55, 0.51,0.56,0.53,0.55],
                'Localhost3 - 10% data': [0.49, 0.49,0.49,0.42,0.52],
                'SL global best ckpt': [0.59, 0.6, 0.6,0.61,0.62,0.63], 'Local merged - 100% data': [0.5, 0.54, 0.57, 0.6, 0.58]},
    'ViT-MIL': {'Localhost1 - 40% data': [0.63, 0.62, 0.58,0.64,0.55], 'Localhost2 - 30% data': [0.63, 0.58, 0.65,0.65,0.61],
                   'Localhost3 - 10% data': [0.61, 0.55, 0.52, 0.53, 0.52, 0.62],
                   'SL global best ckpt': [0.67, 0.64, 0.6, 0.7, 0.69, 0.67, 0.64, 0.68], 'Local merged - 100% data': [0.67, 0.63, 0.63, 0.65, 0.57, 0.64, 0.65, 0.63, 0.63, 0.61]},
    'ViT-LSTM-MIL': {'Localhost1 - 40% data': [0.63, 0.6, 0.60,0.63,0.58,0.57], 'Localhost2 - 30% data': [0.61, 0.64, 0.62,0.65,0.53],
                 'Localhost3 - 10% data': [0.54, 0.49, 0.57,0.56,0.50],
                 'SL global best ckpt': [0.66, 0.68, 0.7, 0.64, 0.61], 'Local merged - 100% data': [0.68, 0.71, 0.67, 0.68, 0.7, 0.67, 0.7, 0.63, 0.62, 0.69, 0.65, 0.65]},

'DenseNet121-3D': {'Localhost1 - 40% data': [0.65,0.62,0.60,0.59,0.65], 'Localhost3 - 10% data': [0.62, 0.67,0.56,0.62,0.60],'Localhost2 - 30% data': [0.54,0.50,0.58,0.58,0.58],
                   'SL global best ckpt': [0.62, 0.62,0.67,0.71,0.66],
                   'Local merged - 100% data': [0.67, 0.58, 0.61, 0.68]},
    'ResNet18-3D': {'Localhost1 - 40% data': ['0.62', '0.48', '0.66', '0.65'], 'Localhost2 - 30% data': ['0.65', '0.63', '0.70', '0.61'],
                    'Localhost3 - 10% data': ['0.52', '0.53', '0.50', '0.60', '0.52'], 'SL global best ckpt': ['0.66', '0.66', '0.69', '0.65'],
                    'Local merged - 100% data': [0.57, 0.51, 0.58, 0.57, 0.7, 0.62]},
    'ResNet50-3D': {'Localhost1 - 40% data': ['0.54', '0.50', '0.64', '0.72', '0.65', '0.58', '0.72', '0.62', '0.72'], 'Localhost2 - 30% data': ['0.66', '0.71', '0.70', '0.75', '0.76', '0.75', '0.77'],
                    'Localhost3 - 10% data': ['0.64', '0.50', '0.69', '0.52', '0.55', '0.58', '0.57', '0.59'], 'SL global best ckpt': ['0.78', '0.74', '0.73', '0.75', '0.75'],
                    'Local merged - 100% data': [0.61, 0.69, 0.68, 0.64, 0.72]},
    'ResNet101-3D': {'Localhost1 - 40% data': ['0.60', '0.75', '0.59', '0.58', '0.61'], 'Localhost2 - 30% data': ['0.74', '0.76', '0.77', '0.72', '0.78'],
                    'Localhost3 - 10% data': ['0.52', '0.64', '0.53', '0.56', '0.49'], 'SL global best ckpt': ['0.79', '0.74', '0.78', '0.79', '0.77'],
                    'Local merged - 100% data': [0.74, 0.75, 0.73, 0.71]},}

# convert string values to float
for model, values in data.items():
    for category, auc_roc_values in values.items():
        data[model][category] = [float(value) for value in auc_roc_values]

data_list = []
for model, values in data.items():
    for category, auc_roc_values in values.items():
        for value in auc_roc_values:
            data_list.append([model, category, value])

df = pd.DataFrame(data_list, columns=['Model', 'Category', 'AUC-ROC'])

# insert None values between groups
models = df['Model'].unique()
df_with_gap = pd.DataFrame(columns=df.columns)
for model in models:
    df_with_gap = df_with_gap.append(df[df['Model'] == model], ignore_index=True)
    df_with_gap = df_with_gap.append(pd.DataFrame([[model, None, np.nan]] * len(df['Category'].unique()), columns=df.columns), ignore_index=True)

plt.figure(figsize=(14, 6))
sns.boxplot(x='Model', y='AUC-ROC', hue='Category', data=df_with_gap)

plt.xlabel("Models")
plt.ylabel("AUC-ROC Values")

plt.tight_layout()
plt.show()