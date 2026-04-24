
"""
Validation + explainability for Appendectomy classification.

Original behavior:
- Loads best.pkl, evaluates on external validation set, writes classification report, AUROC, F1 barplot, confusion matrix, and results.xlsx.

Added behavior (validation-only):
- Optional per-operation per-frame probabilities + importance curves and per-frame CSV exports (use --per_frame).
- Optional export of top-K most important frames per operation for fast visual inspection.

Run:
python3 -m validation -b -o /mnt/sda1/surgery_swarm/validation -d /mnt/sda1/surgery_swarm/data -m /mnt/sda1/surgery_swarm/output

Extra flags:
--per_frame --topk_frames 5 --max_ops 0
"""

import os
import os.path
import glob
import itertools
import argparse
import traceback
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.utils.data
import torchvision.transforms as transforms

from sklearn.metrics import roc_curve, classification_report, auc, f1_score
from sklearn.preprocessing import OneHotEncoder

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import seaborn as sns
from tqdm import tqdm

import Networks
import Dataloader


# ---------------------------
# CLI
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--binary", action="store_true", help="If it is binary")
parser.add_argument("-o", "--output_folder", type=str, required=True, help="Output folder")
parser.add_argument("-d", "--data_folder", type=str, required=True, help="Data folder")
parser.add_argument("-m", "--model_folder", type=str, required=True, help="Model folder (expects best.pkl inside)")

# Explainability / visualization
parser.add_argument(
    "--per_frame",
    action="store_true",
    help="Save per-frame probabilities/importance curves and a per-op CSV (recommended for temporal models).",
)
parser.add_argument(
    "--topk_frames",
    type=int,
    default=5,
    help="Export top-K frames (by importance) per operation when --per_frame is set.",
)
parser.add_argument(
    "--max_ops",
    type=int,
    default=0,
    help="Limit number of operations processed (0 = no limit).",
)

args = parser.parse_args()

debug = False

# ---------------------------
# Experiments (kept aligned with original)
# ---------------------------
center = ["portugal"]
skip = [2]
temporal = [True]
middleframe = [False]

# Output folder
output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)


def check_path(path: str):
    if os.path.exists(path):
        print(f"{path} exists")
    else:
        raise FileNotFoundError(f"{path} was not found or is a directory")


check_path(output_folder)
check_path(args.model_folder)

# Class settings
if args.binary:
    num_class = 2
    classes = [0, 1]
    binary = True
else:
    num_class = 6
    classes = [0, 1, 2, 3, 4, 5]
    binary = False

# Hyperparameters (kept aligned with training/validation)
width = 480
height = 270
batch_size = 64
lstm_size = 160
num_workers = 6

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Generate combinations
all_combinations = itertools.product(center, middleframe, skip, temporal)
filtered_combinations = [(c, m, s, t) for c, m, s, t in all_combinations if (s == 0 or not m)]

# Results table + error log
results_table = pd.DataFrame()
error_list = []


# ---------------------------
# Helpers: per-frame explainability
# ---------------------------
def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_filename(s: str) -> str:
    return "".join([c if c.isalnum() or c in ["-", "_", "."] else "_" for c in s])


def export_topk_frames(dataset: Dataloader.AppendectomyDataset, importance: np.ndarray, out_dir: str, topk: int):
    """
    Save top-K frames as PNGs (raw resized+letterboxed images as in dataloader).
    Uses dataset.loadimage(frame_path) which matches spatial preprocessing.
    """
    ensure_dir(out_dir)
    topk = max(1, int(topk))
    k = min(topk, len(importance))
    idxs = np.argsort(-importance)[:k]

    for rank, idx in enumerate(idxs, start=1):
        frame_path = dataset.images[idx][0]
        try:
            img = dataset.loadimage(frame_path)  # PIL image
            out_path = os.path.join(out_dir, f"top{rank:02d}_idx{idx:04d}.png")
            img.save(out_path)
        except Exception as e:
            print(f"[WARN] Could not export frame {frame_path}: {e}")


def plot_importance_curve(prob: np.ndarray, importance: np.ndarray, out_path: str, title: str):
    """
    Save a 2-panel plot: predicted-class probability and normalized importance over frame index.
    """
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(prob)
    ax1.set_ylabel("P(pred class)")
    ax1.set_xlabel("Frame index")
    ax1.set_title(title)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(importance)
    ax2.set_ylabel("Importance (normalized)")
    ax2.set_xlabel("Frame index")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def per_frame_outputs_for_dataset(model, dataset: Dataloader.AppendectomyDataset, temporal_model: bool):
    """
    Returns:
      frame_paths: list[str] length T
      logits: np.ndarray shape (T, C)
      probs: np.ndarray shape (T, C)
      true_label: int

    Notes:
      - For temporal models, we run the operation sequence in order (no shuffling).
      - For non-temporal models, this still returns per-frame outputs (independent inference).
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_logits = []
    true_label = int(dataset.lbl) if hasattr(dataset, "lbl") else None

    hidden_state = None
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if temporal_model:
                logits, hidden_state = model(images, hidden_state)
                h, c = hidden_state
                hidden_state = (h.detach(), c.detach())
            else:
                logits = model(images, None)

            all_logits.append(logits.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)  # (T, C)
    probs = softmax_np(logits, axis=1)

    frame_paths = [dataset.images[i][0] for i in range(len(dataset))]

    if len(frame_paths) != logits.shape[0]:
        n = min(len(frame_paths), logits.shape[0])
        frame_paths = frame_paths[:n]
        logits = logits[:n]
        probs = probs[:n]

    if true_label is None:
        true_label = int(Counter([int(x) for x in labels.cpu().numpy().tolist()]).most_common(1)[0][0])

    return frame_paths, logits, probs, true_label


for center, middleframe, skip, temporal in tqdm(filtered_combinations):
    frames = 200 if skip == 0 else 200 / skip
    experiment_name = f"{center} {frames} frames" + (", temporal" if temporal else "") + (", middleframe" if middleframe else "")
    experiment_slug = safe_filename(experiment_name.replace(" ", "_"))

    model_folder = f"{args.model_folder}"
    print(f"+++ Model folder: {model_folder}")

    try:
        sns_color = "mako_r"

        # ---------------------------
        # Load model
        # ---------------------------
        model_path = os.path.join(model_folder, "best.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Expected model file not found: {model_path}")

        model = Networks.PhaseLSTMConvNext(num_class, temporal, lstm_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # ---------------------------
        # Data transforms
        # ---------------------------
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        # ---------------------------
        # Load test sets (per operation)
        # ---------------------------
        test_sets = []
        centers = [center]

        for c in centers:
            op_path = os.path.join(args.data_folder, c, "val")
            if not os.path.exists(op_path):
                raise FileNotFoundError(f"Validation folder not found: {op_path}")

            subfolders_val = [f for f in os.listdir(op_path) if os.path.isdir(os.path.join(op_path, f))]
            subfolders_val.sort()

            if args.max_ops and args.max_ops > 0:
                subfolders_val = subfolders_val[: args.max_ops]

            for op_id in subfolders_val:
                ds = Dataloader.AppendectomyDataset(
                    op_path,
                    op_id,
                    width=width,
                    height=height,
                    transform=transform_test,
                    middleframe=middleframe,
                    skip=skip,
                    binary=binary,
                )
                test_sets.append(ds)

        # ---------------------------
        # Aggregate per-operation predictions (kept aligned with original)
        # ---------------------------
        aggregated_y_test_labels = []
        aggregated_y_pred_labels = []
        aggregated_y_pred = []

        if args.per_frame:
            per_frame_root = os.path.join(output_folder, f"{experiment_slug}_per_frame")
            ensure_dir(per_frame_root)

        for op_idx, dataset in enumerate(test_sets):
            label_counter = Counter()
            pred_counter = Counter()
            pred_inter = []

            for data in torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
                images, labels = data
                images = images.to(device)

                with torch.no_grad():
                    if temporal:
                        outputs, _ = model(images, None)
                    else:
                        outputs = model(images, None)

                preds = torch.argmax(outputs, dim=1)

                label_counter.update(labels.cpu().numpy())
                pred_counter.update(preds.cpu().numpy())
                pred_inter.append(outputs.cpu().numpy())

            most_common_label = int(label_counter.most_common(1)[0][0])
            most_common_pred = int(pred_counter.most_common(1)[0][0])

            pred_inter = np.concatenate(pred_inter, axis=0)
            avg_pred = np.mean(pred_inter, axis=0)

            aggregated_y_test_labels.append(most_common_label)
            aggregated_y_pred_labels.append(most_common_pred)
            aggregated_y_pred.append(avg_pred)

            # ---------------------------
            # Per-frame importance + plots (optional)
            # ---------------------------
            if args.per_frame:
                frame_paths, logits_pf, probs_pf, true_label_pf = per_frame_outputs_for_dataset(
                    model=model, dataset=dataset, temporal_model=temporal
                )

                op_pred_class = int(np.argmax(avg_pred))
                prob_pred_class = probs_pf[:, op_pred_class]

                imp = prob_pred_class.astype(np.float64)
                imp = imp - imp.min()
                s = float(imp.sum())
                if s <= 1e-12:
                    importance = np.ones_like(imp) / max(1, len(imp))
                else:
                    importance = imp / s

                op_id = os.path.basename(dataset.image_path) if hasattr(dataset, "image_path") else f"op{op_idx:04d}"
                op_slug = safe_filename(op_id)
                op_dir = os.path.join(per_frame_root, op_slug)
                ensure_dir(op_dir)

                df_pf = pd.DataFrame(
                    {
                        "frame_index": np.arange(len(frame_paths), dtype=int),
                        "frame_path": frame_paths,
                        "true_label": np.full(len(frame_paths), true_label_pf, dtype=int),
                        "op_pred_class": np.full(len(frame_paths), op_pred_class, dtype=int),
                        "p_op_pred_class": prob_pred_class,
                        "importance_norm": importance,
                        "entropy": (-np.sum(probs_pf * np.log(np.clip(probs_pf, 1e-12, 1.0)), axis=1)).astype(np.float64),
                    }
                )
                for k in range(num_class):
                    df_pf[f"logit_c{k}"] = logits_pf[:, k]
                for k in range(num_class):
                    df_pf[f"prob_c{k}"] = probs_pf[:, k]

                df_pf.to_csv(os.path.join(op_dir, "per_frame_predictions.csv"), index=False)

                plot_importance_curve(
                    prob=prob_pred_class,
                    importance=importance,
                    out_path=os.path.join(op_dir, "importance_curve.png"),
                    title=f"{experiment_name} | {op_id} | true={true_label_pf} pred={op_pred_class}",
                )

                export_topk_frames(
                    dataset=dataset,
                    importance=importance,
                    out_dir=os.path.join(op_dir, "top_frames"),
                    topk=args.topk_frames,
                )

        # ---------------------------
        # Global metrics
        # ---------------------------
        y_test_labels = np.array(aggregated_y_test_labels).reshape(-1, 1)
        y_pred_labels = np.array(aggregated_y_pred_labels).reshape(-1, 1)
        y_pred = np.array(aggregated_y_pred).reshape(len(aggregated_y_pred), -1)

        ohe = OneHotEncoder(sparse_output=False)
        ohe.fit(np.array(classes).reshape(-1, 1))
        y_test = ohe.transform(y_test_labels)

        # Classification report
        report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
        report = pd.DataFrame(report).transpose()
        report.to_csv(os.path.join(output_folder, f"{experiment_slug}_classification_report.csv"))

        # AUROC
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(y_test.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        auroc = pd.DataFrame.from_dict(roc_auc, orient="index", columns=["value"])

        plt.style.use("default")
        colors = sns.color_palette(sns_color, n_colors=y_test.shape[1] + 1)
        plt.figure(figsize=(8, 8))
        for i in range(y_test.shape[1]):
            plt.plot(
                fpr[i],
                tpr[i],
                lw=2,
                linestyle="-",
                color=colors[i],
                label="{0} (AUROC = {1:0.2f})".format(ohe.categories_[0][i], roc_auc[i]),
            )
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="Average (AUROC = {0:0.2f})".format(roc_auc["micro"]),
            linewidth=3,
            color=colors[y_test.shape[1]],
        )
        plt.plot([0, 1], [0, 1], color="lightgrey", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{experiment_name}")
        plt.legend(loc="lower left", bbox_to_anchor=(1, 0))
        plt.savefig(os.path.join(output_folder, f"{experiment_slug}_auroc.png"), bbox_inches="tight")
        plt.clf()

        # F1 Score
        categories = np.transpose(ohe.categories_).flatten()
        score = f1_score(y_test_labels, y_pred_labels, labels=categories, average=None)
        score_df = pd.DataFrame({"class": categories, "score": score})
        plt.figure(figsize=(8, 8))
        sns.barplot(
            data=score_df,
            x="class",
            y="score",
            hue="class",
            palette=sns.color_palette(sns_color),
            legend=False,
        )
        plt.ylabel("F1 score")
        plt.xlabel("class")
        plt.xticks(rotation=90)
        plt.title(f"{experiment_name}")
        plt.savefig(os.path.join(output_folder, f"{experiment_slug}_f1_score.png"), bbox_inches="tight")
        plt.clf()

        # Confusion Matrix
        y_pred_labels_cm = [int(item) for item in aggregated_y_pred_labels]
        y_test_labels_cm = [int(item) for item in aggregated_y_test_labels]
        pred_df = pd.DataFrame(list(zip(y_pred_labels_cm, y_test_labels_cm)), columns=["pred_label", "true_label"])
        df_cm = pd.crosstab(
            pred_df["pred_label"],
            pred_df["true_label"],
            rownames=["Predicted"],
            colnames=["True"],
            dropna=False,
        )
        df_cm = df_cm.reindex(index=classes, columns=classes, fill_value=0)

        column_sums = df_cm.sum(axis=0)
        norm_df = df_cm.div(column_sums, axis=1).fillna(0)

        plt.figure(figsize=(8, 8))
        sns.heatmap(
            norm_df,
            annot=True,
            cmap=sns.color_palette(sns_color, as_cmap=True),
            fmt=".2f",
            xticklabels=classes,
            yticklabels=classes,
        )
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{experiment_name}")
        plt.savefig(os.path.join(output_folder, f"{experiment_slug}_confusion_matrix.png"), bbox_inches="tight")
        plt.clf()

        # Save arrays
        np.save(os.path.join(output_folder, f"{experiment_slug}_y_pred_labels.npy"), np.array(aggregated_y_pred_labels))
        np.save(os.path.join(output_folder, f"{experiment_slug}_y_test_labels.npy"), np.array(aggregated_y_test_labels))
        np.save(os.path.join(output_folder, f"{experiment_slug}_y_pred_logits_mean.npy"), np.array(aggregated_y_pred))

        # Save values in Dataframe (as original)
        combined_data = {}
        for metric in ["precision", "recall", "f1-score", "support"]:
            for idx in report.index:
                combined_data[f"{metric}_{idx}"] = [report.at[idx, metric]]
        for idx in auroc.index:
            combined_data[f"auroc_{idx}"] = [auroc.at[idx, "value"]]

        combined_df = pd.DataFrame(combined_data)
        combined_df.index = [experiment_name]
        results_table = pd.concat([results_table, combined_df])
        results_table.to_excel(os.path.join(output_folder, "results.xlsx"))

        torch.cuda.empty_cache()

    except Exception as e:
        print(f"An error occurred: {e}")
        tb = traceback.format_exc()
        print(tb)

        error_list.append([model_folder, e, tb])
        with open(os.path.join(output_folder, "errors.txt"), "w") as file:
            for item in error_list:
                file.write(f"Model Folder: {item[0]}
")
                file.write(f"Error: {item[1]}
")
                file.write(f"Traceback:
{item[2]}

")

# Always write results table
results_table.to_excel(os.path.join(output_folder, "results.xlsx"))
