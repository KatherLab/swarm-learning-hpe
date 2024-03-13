import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set the path to your results directory
results_dir = '/opt/hpe/swarm-learning-hpe/workspace/marugoto_mri/user/data-and-scratch/scratch'


# Initialize a dictionary to store AUC values per folder category
auc_dict = defaultdict(list)

# Traverse the directory tree
for root, _, files in os.walk(results_dir):
    for file in files:
        if file == 'roc-Malign=1.svg':
            folder_name = os.path.basename(root)
            match = re.search(r'(\d{4}_\d{2}_\d{2}_\d{6}_DUKE_)(\w+)(_swarm_|_local_)', folder_name)
            if match:
                main_category = match.group(3)[:-1]  # remove the trailing '_'
                model_name = match.group(2)
                folder_category = f"{main_category}_{model_name}"

                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    auc_match = re.search(r'<!-- \(AUC = (\d+\.\d+)\) -->', content)
                    if auc_match:
                        auc_value = float(auc_match.group(1))
                        auc_dict[folder_category].append(auc_value)

# Create a DataFrame from the AUC values dictionary
data = []
for category, auc_list in auc_dict.items():
    main_category, model_name = category.split('_', 1)
    for auc_value in auc_list:
        data.append({'Main Category': main_category, 'Model Name': model_name, 'AUC': auc_value})

df = pd.DataFrame(data)

# Create the box plot with swarmplot overlay
sns.set(rc={'figure.figsize': (8, 8)})
sns.set_style("whitegrid")
title = 'AUC values per folder category'
x_label = 'Main Category'
y_label = 'AUC'

sns.swarmplot(x='Main Category', y='AUC', data=df, hue='Model Name', size=2, dodge=True)
fg = sns.boxplot(x='Main Category', y='AUC', data=df, hue='Model Name', flierprops=dict(markerfacecolor='0.50', markersize=0.8))

fg.set_title(title, fontsize=25, fontname="Calibri")
fg.set_xlabel(x_label, fontsize=20, fontname="Calibri")
fg.set_ylabel(y_label, fontsize=20, fontname="Calibri")
a = fg.get_yticks()

fg.set_yticklabels(np.around(a, decimals=2), size=14)
fg.set_xticklabels(df['Main Category'].unique(), size=15, fontname="Calibri")
handles, _ = fg.get_legend_handles_labels()
fg.legend(handles, df['Model Name'].unique(), fontsize=12, loc='lower right')

plt.savefig('box_plot.svg')
plt.savefig('box_plot.png')
plt.show()



for key,value in auc_dict.items():
    print(key,value)