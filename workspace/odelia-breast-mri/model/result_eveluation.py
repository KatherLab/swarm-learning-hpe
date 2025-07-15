from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate(input_csv):
    df = pd.read_csv(input_csv)
    #  -------------------------- F1 score-------------------------
    from sklearn.metrics import f1_score
    f1 = f1_score(df['GT'], df['NN'])
    logger.info("F1 Score: {:.2f}".format(f1))
    print("F1 Score: {:.2f}".format(f1))

#  -------------------------- Confusion Matrix -------------------------
    cm = confusion_matrix(df['GT'], df['NN'])
    tn, fp, fn, tp = cm.ravel()
    # print tn, fp, fn, tp
    print("tn, fp, fn, tp: ",tn, fp, fn, tp) # tn, fp, fn, tp:  232 0 62 0

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
    print('auc_val: ', auc_val)
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

    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    evaluate("/mnt/sda1/odelia_paper_trained_results/2023_06_20_060932_DUKE_ext_ResNet101_swarm_learning/results.csv")