#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data.datasets import DUKE_Dataset3D, DUKE_Dataset3D_external, DUKE_Dataset3D_collab
from data.datamodules import DataModule
from utils.roc_curve import plot_roc_curve, cm2acc, cm2x
import torch
from models import ResNet, VisionTransformer, EfficientNet, EfficientNet3D, EfficientNet3Db7, DenseNet121, UNet3D
from sklearn.metrics import f1_score


def predict(model_dir, test_data_dir, model_name, last_flag, prediction_flag, cohort_flag='aachen'):
    # ------------ Settings/Defaults ----------------
    # path_run = Path.cwd() / 'runs/2023_02_06_175325'
    # path_run = Path('/opt/hpe/swarm-learning-hpe/workspace/odelia-breast-mri/user-odelia-breast-mri-192.168.33.102/data-and-scratch/scratch/2023_02_06_205810/')
    path_run = Path(model_dir)
    #path_run = Path(
        #'/opt/hpe/swarm-learning-hpe/workspace/odelia-breast-mri/user-odelia-breast-mri-192.168.33.102/data-and-scratch/scratch/2023_02_06_224851')
    #path_out = Path().cwd() / 'results' / path_run.name
    path_out = Path(path_run, f"{prediction_flag}_{cohort_flag}")
    print("path_out.absolute()", path_out.absolute())
    path_out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}

    # ------------ Logging --------------------
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # ------------ Load Data ----------------
    if prediction_flag == 'ext':
        ds = DUKE_Dataset3D_external(
            flip=False,
            path_root=test_data_dir
        )
    elif prediction_flag == 'internal':
        ds = DUKE_Dataset3D(
            flip=False,
            path_root=test_data_dir
        )
    elif prediction_flag == 'collab':
        ds = DUKE_Dataset3D_collab(
            flip=False,
            path_root=test_data_dir
        )
    # only get 10 samples
    #ds_test = torch.utils.data.Subset(ds, range(10))
    ds_test = ds
    print('number of test data')
    print(len(ds_test))

    dm = DataModule(
        ds_test=ds_test,
        batch_size=1,
        # num_workers=0,
        # pin_memory=True,
    )

    # ------------ Initialize Model ------------
    if model_name == 'ResNet18':
        layers = [2, 2, 2, 2]
    elif model_name == 'ResNet34':
        layers = [3, 4, 6, 3]
    elif model_name == 'ResNet50':
        layers = [3, 4, 6, 3]
    elif model_name == 'ResNet101':
        layers = [3, 4, 23, 3]
    elif model_name == 'ResNet152':
        layers = [3, 8, 36, 3]
    else:
        layers = None

    if layers is not None:
        # ------------ Initialize Model ------------
        if last_flag:
            model = ResNet.load_last_checkpoint(path_run, version=0, layers=layers)
        else:
            model = ResNet.load_best_checkpoint(path_run, version=0, layers=layers)

    elif model_name in ['efficientnet_l1', 'efficientnet_l2', 'efficientnet_b4', 'efficientnet_b7']:
        if last_flag:
            model = EfficientNet.load_last_checkpoint(path_run, version=0, model_name=model_name)
        else:
            model = EfficientNet.load_best_checkpoint( path_run, version=0, model_name = model_name)
    elif model_name == 'EfficientNet3Db0':
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25"]
    elif model_name == 'EfficientNet3Db4':
        blocks_args_str = [
            "r1_k3_s11_e1_i48_o24_se0.25",
            "r3_k3_s22_e6_i24_o32_se0.25",
            "r3_k5_s22_e6_i32_o56_se0.25",
            "r4_k3_s22_e6_i56_o112_se0.25",
            "r4_k5_s11_e6_i112_o160_se0.25",
            "r5_k5_s22_e6_i160_o272_se0.25",
            "r2_k3_s11_e6_i272_o448_se0.25"]
    elif model_name == 'EfficientNet3Db7':
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o32_se0.25",
            "r4_k3_s22_e6_i32_o48_se0.25",
            "r4_k5_s22_e6_i48_o80_se0.25",
            "r4_k3_s22_e6_i80_o160_se0.25",
            "r6_k5_s11_e6_i160_o256_se0.25",
            "r6_k5_s22_e6_i256_o384_se0.25",
            "r3_k3_s11_e6_i384_o640_se0.25"]
    elif model_name == 'DenseNet121':
        if last_flag:
            model = DenseNet121.load_last_checkpoint(path_run, version=0)
        else:
            model = DenseNet121.load_best_checkpoint(path_run, version=0)
    elif model_name == 'UNet3D':
        if last_flag:
            model = UNet3D.load_last_checkpoint(path_run, version=0)
        else:
            model = UNet3D.load_best_checkpoint(path_run, version=0)
    else:
        raise Exception("Invalid network model specified")

    if model_name.startswith('EfficientNet3D'):
        if last_flag:
            model = EfficientNet3D.load_last_checkpoint(path_run, version=0, blocks_args_str=blocks_args_str)
        else:
            model = EfficientNet3D.load_best_checkpoint(path_run, version=0,blocks_args_str=blocks_args_str)
    model.to(device)
    model.eval()

    results = {'uid': [],'GT': [], 'NN': [], 'NN_pred': []}
    threshold = 0.5  # Threshold for binary classification

    for batch in tqdm(dm.test_dataloader()):
        source, target = batch['source'], batch['target']

        # Run Model
        pred = model(source.to(device)).cpu()
        pred_proba = torch.sigmoid(pred).squeeze()  # Assuming single output for positive class probability
        pred_binary = (pred_proba > threshold).long()  # Classify based on threshold
        #print(pred_proba)
        #print(pred_binary)
        results['GT'].extend(target.tolist())
        if isinstance(pred_binary.tolist(), int):
            # If pred_binary.tolist() is an integer, convert it to a list with a single element
            pred_binary_list = [pred_binary.tolist()]
            #print(pred_binary_list)
        else:
            # Otherwise, use the list directly
            pred_binary_list = pred_binary.tolist()
            #print(pred_binary_list)

        if isinstance(pred_proba.tolist(), float):
            # If pred_binary.tolist() is an integer, convert it to a list with a single element
            pred_proba_list = [pred_proba.tolist()]
            #print(pred_proba_list)
        else:
            # Otherwise, use the list directly
            pred_proba_list = pred_proba.tolist()
            #print(pred_proba_list)

        results['NN'].extend(pred_binary_list)
        results['NN_pred'].extend(pred_proba_list)
        results['uid'].extend(batch['uid'])

    df = pd.DataFrame(results)
    if last_flag:
        df.to_csv(path_out / 'results_last.csv')
    else:
        df.to_csv(path_out / 'results.csv')
    #  -------------------------- F1 score-------------------------
    f1 = f1_score(df['GT'], df['NN'])
    logger.info("F1 Score: {:.2f}".format(f1))
    #print("F1 Score: {:.2f}".format(f1))
    #  -------------------------- Confusion Matrix -------------------------
    cm = confusion_matrix(df['GT'], df['NN'])
    tn, fp, fn, tp = cm.ravel()
    # print tn, fp, fn, tp

    n = len(df)
    logger.info(
        "Confusion Matrix: TN {} ({:.2f}%), FP {} ({:.2f}%), FN {} ({:.2f}%), TP {} ({:.2f}%)".format(tn, tn / n * 100,
                                                                                                      fp, fp / n * 100,
                                                                                                      fn, fn / n * 100,
                                                                                                      tp, tp / n * 100))

    # ------------------------------- ROC-AUC ---------------------------------
    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    y_pred_lab = np.asarray(df['NN_pred'])
    y_true_lab = np.asarray(df['GT'])
    tprs, fprs, auc_val, thrs, opt_idx, cm = plot_roc_curve(y_true_lab, y_pred_lab, axis, fontdict=fontdict)
    #print ('auc_val: ',auc_val)
    fig.tight_layout()
    if last_flag:
        fig.savefig(path_out / f'roc_last.png', dpi=300)
    else:
        fig.savefig(path_out / f'roc.png', dpi=300)
    # -------------------------- Precision-Recall Curve (PRC) and Average Precision (AP) -------------------------
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true_lab, y_pred_lab)
    ap = average_precision_score(y_true_lab, y_pred_lab)



    #  -------------------------- Matthews Correlation Coefficient (MCC) -------------------------
    '''
    from sklearn.metrics import matthews_correlation_coefficient
    mcc = matthews_correlation_coefficient(y_true_lab, y_pred_lab)
    '''
    #  -------------------------- PPV and NPV -------------------------
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    #  -------------------------- Confusion Matrix -------------------------
    acc = cm2acc(cm)
    _, _, sens, spec = cm2x(cm)
    df_cm = pd.DataFrame(data=cm, columns=['False', 'True'], index=['False', 'True'])
    fig, axis = plt.subplots(1, 1, figsize=(4, 4))
    sns.heatmap(df_cm, ax=axis, cbar=False, fmt='d', annot=True)
    axis.set_title(f'Confusion Matrix ACC={acc:.2f}', fontdict=fontdict)  # CM =  [[TN, FP], [FN, TP]]
    axis.set_xlabel('Prediction', fontdict=fontdict)
    axis.set_ylabel('True', fontdict=fontdict)
    fig.tight_layout()
    if last_flag:
        fig.savefig(path_out / f'confusion_matrix_last.png', dpi=300)
    else:
        fig.savefig(path_out / f'confusion_matrix.png', dpi=300)

    logger.info(f"Malign  Objects: {np.sum(y_true_lab)}")
    logger.info("Confusion Matrix {}".format(cm))
    logger.info("Sensitivity {:.2f}".format(sens))
    logger.info("Specificity {:.2f}".format(spec))

    # -------------------------- Save Metrics -------------------------
    with open(path_out / 'metrics.txt', 'w') as f:
        f.write(f"AUC: {auc_val:.2f}\n")
        f.write(f"F1 Score: {f1:.2f}\n")
        f.write(f"Sensitivity: {sens:.2f}\n")
        f.write(f"Specificity: {spec:.2f}\n")
        f.write(f"PPV: {ppv:.2f}\n")
        f.write(f"NPV: {npv:.2f}\n")
        f.write(f"ACC: {acc:.2f}\n")
        f.write(f"AP: {ap:.2f}\n")

    # -------------------------- print Metrics -------------------------
    print(f"AUC: {auc_val:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Sensitivity: {sens:.2f}")
    print(f"Specificity: {spec:.2f}")
    print(f"PPV: {ppv:.2f}")
    print(f"NPV: {npv:.2f}")
    print(f"ACC: {acc:.2f}")
    print(f"AP: {ap:.2f}")



    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    wouter_data_path = "/mnt/sda1/swarm-learning/wouter_data/preprocessed_re/"
    athens_data_path = "/mnt/sda1/swarm-learning/athens_data/preprocessed_athens/"
    predict(
        model_dir = Path(
        '/mnt/sda1/odelia_paper_trained_results/2023_07_04_180000_DUKE_ext_ResNet101_swarm_learning'),
        test_data_dir=athens_data_path,


        #'/opt/hpe/swarm-learning-hpe/workspace/odelia-breast-mri/user/data-and-scratch/data/DUKE_ext/test',
        model_name='ResNet101',
        last_flag=False,
        prediction_flag='collab',
        cohort_flag = 'athens')