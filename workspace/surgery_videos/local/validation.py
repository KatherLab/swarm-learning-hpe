'''
Run with:

python3 -m validation -e -b -o /mnt/sda1/surgery_swarm/validation -d /mnt/sda1/surgery_swarm/data -m /mnt/sda1/surgery_swarm/output

- `-b`: Binary flag if set only calss 0/1
- `-o`: Output folder
- `-d`: Data folder
- `-m`: Model folder
'''

import glob
import os.path
import itertools
import argparse
from shutil import copy2
import Networks
import Dataloader
import torch
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models.resnet
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, classification_report, auc, f1_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import traceback

# Create the parser
parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument('-b', '--binary', action='store_true', help='If it is binary')
parser.add_argument('-o', '--output_folder', type=str, help='Output folder: /mnt/sda1/surgery_swarm/validation_01_center04')
parser.add_argument('-d', '--data_folder', type=str, help='Data folder: /mnt/sda1/surgery_swarm/data')
parser.add_argument('-m', '--model_folder', type=str, help='Model folder: /mnt/sda1/surgery_swarm/output_01')
# Parse the arguments
args = parser.parse_args()

# Assume that we are on a CUDA machine, then this should print a CUDA device
debug = False

# Define experiments
center = ['center01', 'center02', 'center03', 'center04', 'center05', 'all_centers']
skip = [20, 8, 4, 2, 0] # Frames 10, 25, 50, 100, 200
temporal = [True, False]
middleframe = [True, False]

# Define experiment parameters

output_folder = args.output_folder

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Check entered paths
def check_path(path):
    if os.path.exists(path):
        print(F"{path} exists")
    else:
        raise FileNotFoundError(f"{path} was not found or is a directory")

check_path(output_folder)
check_path(args.model_folder)

# Set binary settings   
if args.binary:
    num_class = 2
    classes=[0,1]
    binary = True
else:
    num_class = 6
    classes=[0,1,2,3,4,5] 
    binary= False

# Define hyperparameters
width = 480
height = 270
batch_size = 64
lstm_size = 160
num_workers = 6

# Define Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Generate all combinations
all_combinations = itertools.product(center, middleframe, skip, temporal)

# Filter combinations to use either skip or middleframe, but not both
filtered_combinations = [(c, m, s, t) for c, m, s, t in all_combinations if (s == 0 or not m)]

# Initialize results Dataframe
results_table = pd.DataFrame()
error_list = []

for center, middleframe, skip, temporal in tqdm(filtered_combinations):
    # Define experiment name
    frames = 200 if skip == 0 else 200 / skip
    experiment_name = f"{center} {frames} frames" + f"{', temporal' if temporal else ''}" + f"{', middleframe' if middleframe else ''}"
    model_folder = f"{args.model_folder}/Appendectomy_Classification_{center}_temp{temporal}_mid{middleframe}_{frames}frames*/"
    print(f"+++{model_folder}")

    try:
        # Define Seaborn Colormap for Graphics
        sns_color= "mako_r"

        # Load the trained model and set to evluation mode
        model_folder = glob.glob(model_folder)[0]
        model_path = model_folder + "best.pkl"
        model = Networks.PhaseLSTMConvNext(num_class, temporal, lstm_size) 
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # define normalisation of the test data
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        # function that only listss directories
        def list_directories(path):
            return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        
        # load test set
        test_sets = []

        '''
        if center == 'all_centers':
            centers = ['center01', 'center02', 'center03', 'center04', 'center05']

            for c in centers:
                op_path = os.path.join(args.data_folder, c, "val")
                subfolders_val = [f for f in os.listdir(op_path) if os.path.isdir(os.path.join(op_path, f))]

                for id in subfolders_val:
                    set = Dataloader.AppendectomyDataset(op_path, id, width=width, height=height, transform=transform_test, middleframe=middleframe, skip=skip, binary=binary)
                    test_sets.append(set)
        else:
            op_path = os.path.join(args.data_folder, center, "val")
            subfolders_val = [f for f in os.listdir(op_path) if os.path.isdir(os.path.join(op_path, f))]
    
            for id in subfolders_val:
                set = Dataloader.AppendectomyDataset(op_path, id, width=width, height=height, transform=transform_test, middleframe=middleframe, skip=skip, binary=binary)
                test_sets.append(set)
        '''
        centers = ['center01', 'center02', 'center03', 'center04', 'center05']
        
        for c in centers:
                op_path = os.path.join(args.data_folder, c, "val")
                subfolders_val = [f for f in os.listdir(op_path) if os.path.isdir(os.path.join(op_path, f))]

                for id in subfolders_val:
                    set = Dataloader.AppendectomyDataset(op_path, id, width=width, height=height, transform=transform_test, middleframe=middleframe, skip=skip, binary=binary)
                    test_sets.append(set)
    
        test_set = torch.utils.data.ConcatDataset(test_sets)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Initialize lists to store aggregated labels and predictions
        aggregated_y_test_labels = []
        aggregated_y_pred_labels = []
        aggregated_y_pred = []

        # Iterate over the test sets
        for dataset in test_sets:
            # Initialize counters for labels and predictions
            label_counter = Counter()
            pred_counter = Counter()
            pred_inter = []

            # Iterate over the DataLoader for the current dataset
            for data in torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
                images, labels = data
                images = images.to(device)
                # Get model predictions
                with torch.no_grad():
                    if temporal:
                        outputs, hidden_state = model(images, None)
                    else:
                        outputs = model(images, None)
                    preds = torch.argmax(outputs, dim=1)
                # Update counters and sum for predictions
                label_counter.update(labels.cpu().numpy())
                pred_counter.update(preds.cpu().numpy())
                
                # Append the numpy array of outputs to the list
                pred_inter.append(outputs.cpu().numpy())

            # Get the most common label and prediction for the operation
            most_common_label = label_counter.most_common(1)[0][0]
            most_common_pred = pred_counter.most_common(1)[0][0]
            # Average the predictions
            pred_inter = np.concatenate(pred_inter, axis=0)
            avg_pred = np.mean(pred_inter, axis=0)

            # Append the aggregated label and prediction
            aggregated_y_test_labels.append(most_common_label)
            aggregated_y_pred_labels.append(most_common_pred)
            aggregated_y_pred.append(avg_pred)
       
        y_test_labels = np.array(aggregated_y_test_labels).reshape(-1, 1)
        y_pred_labels = np.array(aggregated_y_pred_labels).reshape(-1, 1)
        y_pred = np.array(aggregated_y_pred)
        x, y = y_pred.shape
        y_pred = y_pred.reshape(x,y)
        ohe = OneHotEncoder(sparse_output=False)
        ohe.fit(np.array(classes).reshape(-1, 1))
        y_test = ohe.transform(y_test_labels)

        ## Classification Report
        report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
        report = pd.DataFrame(report).transpose()
        report.to_csv(os.path.join(output_folder, f"{experiment_name.replace(' ', '_')}_classification_report.csv"))

        ## AUROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(y_test.shape[1]):
            fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        auroc = pd.DataFrame.from_dict(roc_auc, orient='index', columns=['value'])
        plt.style.use('default')
        colors= sns.color_palette(sns_color, n_colors=y_test.shape[1]+1)
        plt.figure(figsize=(8, 8))
        for i in range(y_test.shape[1]):
            plt.plot(fpr[i], tpr[i], lw=2, linestyle='-', color=colors[i],
                        label='{0} (AUROC = {1:0.2f})'
                        ''.format(ohe.categories_[0][i], roc_auc[i]))
        plt.plot(fpr["micro"], tpr["micro"],
                    label='Average (AUROC = {0:0.2f})'
                    ''.format(roc_auc["micro"]), linewidth=3, color=colors[i+1])
        plt.plot([0, 1], [0, 1], color='lightgrey', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"{experiment_name}")
        plt.legend(loc="lower left", bbox_to_anchor=(1, 0))
        plt.savefig(os.path.join(output_folder, f"{experiment_name.replace(' ', '_')}_auroc.png"), bbox_inches='tight')
        plt.clf()

        ## F1 Score
        categories = np.transpose(ohe.categories_).flatten()
        score = f1_score(y_test_labels, y_pred_labels,
                            labels=categories, average=None)
        score_df = pd.DataFrame({'class': categories, 'score': score})
        plt.figure(figsize=(8, 8))
        sns.barplot(data=score_df, x='class', y='score', hue='class', palette=sns.color_palette(sns_color), legend=False)
        plt.ylabel("F1 score")
        plt.xlabel("class")
        plt.xticks(rotation=90)
        plt.title(f"{experiment_name}")
        plt.savefig(os.path.join(output_folder, f"{experiment_name.replace(' ', '_')}_f1_score.png"), bbox_inches='tight')
        plt.clf()

        ## Confusion Matrix
        # Flatten the label lists
        y_pred_labels_cm = [item[0] for item in y_pred_labels]  # Flatten predicted labels
        y_test_labels_cm = [item[0] for item in y_test_labels]  # Flatten true labels
        # Create DataFrame
        pred_df = pd.DataFrame(list(zip(y_pred_labels_cm, y_test_labels_cm)), columns=['pred_label', 'true_label'])
        # Verify lengths
        assert len(y_pred_labels_cm) == len(y_test_labels_cm), "Mismatch in length of predicted and true labels."
        # Confusion Matrix with all classes
        df = pd.crosstab(pred_df['pred_label'], pred_df['true_label'], rownames=['Predicted'], colnames=['True'], dropna=False)
        df = df.reindex(index=classes, columns=classes, fill_value=0)
        # Normalize
        column_sums = df.sum(axis=0)
        norm_df = df.div(column_sums, axis=1).fillna(0)  # Fill NaN with 0 after normalization
        # Plot
        plt.figure(figsize=(8, 8))
        sns.heatmap(norm_df, annot=True, cmap=sns.color_palette(sns_color, as_cmap=True), fmt=".2f", xticklabels=classes, yticklabels=classes)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{experiment_name}")
        plt.savefig(os.path.join(output_folder, f"{experiment_name.replace(' ', '_')}_confusion_matrix.png"), bbox_inches='tight')
        plt.clf()

        ## Save Rersult Arrays (if we want to apply other statistics later)
        np.save(os.path.join(output_folder, f"{experiment_name.replace(' ', '_')}_y_pred.npy"), y_pred_labels)
        np.save(os.path.join(output_folder, f"{experiment_name.replace(' ', '_')}_y_test.npy"), y_test_labels)

        ## Save values in Dataframe
        # Initialize a new DataFrame to store the combined data
        combined_data = {}

        # Add precision, recall, f1-score, and support to the combined data
        for metric in ['precision', 'recall', 'f1-score', 'support']:
            for idx in report.index:
                combined_data[f'{metric}_{idx}'] = [report.at[idx, metric]]

        # Add auroc values to the combined data
        for idx in auroc.index:
            combined_data[f'auroc_{idx}'] = [auroc.at[idx, 'value']]

        # Create the combined DataFrame
        combined_df = pd.DataFrame(combined_data)
        combined_df.index = [experiment_name]
        results_table = pd.concat([results_table, combined_df])
        results_table.to_excel(os.path.join(output_folder,"results.xlsx"))
        torch.cuda.empty_cache()
    except Exception as e:
        # Print the error message
        print(f"An error occurred: {e}")
        
        # Capture and print the full traceback
        tb = traceback.format_exc()
        print(tb)
        
        # Append the error details to the error list
        error_list.append([model_folder, e, tb])
        
        # Write the errors to the file
        with open(os.path.join(output_folder, "errors.txt"), 'w') as file:
            for item in error_list:
                file.write(f"Model Folder: {item[0]}\n")
                file.write(f"Error: {item[1]}\n")
                file.write(f"Traceback:\n{item[2]}\n")
                file.write("\n")
# save results table
results_table.to_excel(os.path.join(output_folder,"results.xlsx"))