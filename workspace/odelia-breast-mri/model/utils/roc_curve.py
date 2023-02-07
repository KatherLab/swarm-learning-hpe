import numpy as np 
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib

def plot_roc_curve(y_true, y_score, axis, bootraping=1000, drop_intermediate=False, fontdict={}):
    # ----------- Bootstrapping ------------
    tprs, aucs, thrs = [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    rand_idxs = np.random.randint(0, len(y_true), size=(bootraping, len(y_true))) # Note: with replacement 
    for rand_idx in rand_idxs:
        y_true_set = y_true[rand_idx]
        y_score_set = y_score[rand_idx]
        fpr, tpr, thresholds = roc_curve(y_true_set, y_score_set, drop_intermediate=drop_intermediate)
        tpr_interp = np.interp(mean_fpr, fpr, tpr) # must be interpolated to gain constant/equal fpr positions
        tprs.append(tpr_interp) 
        aucs.append(auc(fpr, tpr))
        optimal_idx = np.argmax(tpr - fpr)
        thrs.append(thresholds[optimal_idx])

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0        
    std_tpr = np.std(tprs, axis=0, ddof=1)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # ------ Averaged based on bootspraping ------
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs, ddof=1)
 

    # --------- Specific Case -------------
    fprs, tprs, thrs = roc_curve(y_true, y_score, drop_intermediate=drop_intermediate)
    auc_val = auc(fprs, tprs)
    opt_idx = np.argmax(tprs - fprs)
    opt_tpr = tprs[opt_idx]
    opt_fpr = fprs[opt_idx]

  
    y_scores_bin = y_score>=thrs[opt_idx] # WANRING: Must be >= not > 
    conf_matrix = confusion_matrix(y_true, y_scores_bin) # [[TN, FP], [FN, TP]]
    


    axis.plot(fprs, tprs, color='b', label=rf"ROC (AUC = {auc_val:.2f} $\pm$ {std_auc:.2f})",
                lw=2, alpha=.8)
    axis.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    axis.hlines(y=opt_tpr, xmin=0.0, xmax=opt_fpr, color='g', linestyle='--')
    axis.vlines(x=opt_fpr, ymin=0.0, ymax=opt_tpr, color='g', linestyle='--')
    axis.plot(opt_fpr, opt_tpr, color='g', marker='o') 
    axis.plot([0, 1], [0, 1], linestyle='--', color='k')
    axis.set_xlim([0.0, 1.0])
    axis.set_ylim([0.0, 1.0])
    
    axis.legend(loc='lower right')
    axis.set_xlabel('1 - Specificity', fontdict=fontdict)
    axis.set_ylabel('Sensitivity', fontdict=fontdict)
    
    axis.grid(color='#dddddd')
    axis.set_axisbelow(True)
    axis.tick_params(colors='#dddddd', which='both')
    for xtick in axis.get_xticklabels():
        xtick.set_color('k')
    for ytick in axis.get_yticklabels():
        ytick.set_color('k')
    for child in axis.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('#dddddd')
 
    return tprs, fprs, auc_val, thrs, opt_idx, conf_matrix


def cm2acc(cm):
    # [[TN, FP], [FN, TP]] 
    tn, fp, fn, tp = cm.ravel()
    return (tn+tp)/(tn+tp+fn+fp)

def safe_div(x,y):
    if y == 0:
        return float('nan') 
    return x / y

def cm2x(cm):
    tn, fp, fn, tp = cm.ravel()
    pp = tp + fp  # predicted positive 
    pn = fn + tn  # predicted negative
    p = tp + fn   # actual positive
    n = fp + tn   # actual negative  

    ppv = safe_div(tp,pp)  # positive predictive value 
    npv = safe_div(tn,pn)  # negative predictive value 
    tpr = safe_div(tp,p)   # true positive rate (sensitivity, recall)
    tnr = safe_div(tn,n)   # true negative rate (specificity)
    # Note: other values are 1-x eg. fdr=1-ppv, for=1-npv, ....
    return ppv, npv, tpr, tnr