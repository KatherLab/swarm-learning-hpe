"""
validation_ultimate.py

Drop-in replacement built to match your existing codebase:
- Uses Dataloader.AppendectomyDataset (frame dataset) exactly like original validation.py
- Loads Networks.PhaseLSTMConvNext exactly like original validation.py
- Adds per-frame outputs + plots + top-k frames export
- Adds *fast* leave-one-out importance (causal w.r.t. mean-logit aggregation) WITHOUT extra forward passes

Example (binary, no skipping -> all frames):
python3 -m validation_ultimate \
  -b \
  -o /mnt/dlhd0/surgery_swarm/jeff_vis_ultimate \
  -d /mnt/dlhd0/surgery_swarm/data \
  -m /mnt/dlhd0/surgery_swarm/results_lst_exp/training_binary_1/Appendectomy_Classification_all_centers_tempTrue_midFalse_100.0frames20250607-1438/ \
  -c /mnt/dlhd0/surgery_swarm/data/portugal/portugal.csv \
  --centers portugal \
  --split val \
  --temporal \
  --skip 0 \
  --per_frame \
  --topk_frames 10 \
  --export_plots \
  --export_frames

Notes
- If -m points to a directory, we load <dir>/best.pkl (same behavior as original).
- If -m points to a file, we load that file.
- Dataloader.AppendectomyDataset expects a labels CSV located inside the split folder (e.g. .../portugal/val/*.csv).
  If you pass -c, this script will copy that CSV into each split folder if needed.
"""

import os
import glob
import argparse
from shutil import copy2
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.utils.data
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, classification_report, auc, confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder

import Networks
import Dataloader


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def resolve_model_path(model_arg: str) -> str:
    # Match original behavior: model_folder + "best.pkl"
    if os.path.isdir(model_arg):
        return os.path.join(model_arg, "best.pkl")
    return model_arg


def list_subdirs(path: str):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def entropy_np(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float32)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(xp, kernel, mode="valid").astype(np.float32)


def importance_norm_from_scores(scores: np.ndarray) -> np.ndarray:
    """
    Same normalization you saw earlier:
      raw = scores - min(scores)
      norm = raw / (sum(raw) + eps)
    """
    s = scores.astype(np.float32)
    raw = s - float(np.min(s))
    denom = float(np.sum(raw)) + 1e-12
    return (raw / denom).astype(np.float32)


def leave_one_out_importance(
    frame_logits: np.ndarray,
    op_pred_class: int,
) -> np.ndarray:
    """
    Fast causal-ish importance w.r.t mean-logit aggregation:
    - base = softmax(mean(logits))[op_pred_class]
    - for each frame t: recompute mean without frame t, take prob for op_pred_class
      importance[t] = base - prob_without_t
    This does NOT require extra model forward passes.

    frame_logits: (T, C)
    returns: (T,) float32
    """
    T, C = frame_logits.shape
    mean_logits = frame_logits.mean(axis=0)
    base_prob = softmax_np(mean_logits)[op_pred_class]

    # precompute sum to get leave-one-out mean efficiently
    sum_logits = frame_logits.sum(axis=0)  # (C,)
    imp = np.zeros((T,), dtype=np.float32)

    if T <= 1:
        return imp

    for t in range(T):
        mean_wo = (sum_logits - frame_logits[t]) / float(T - 1)
        prob_wo = softmax_np(mean_wo)[op_pred_class]
        imp[t] = float(base_prob - prob_wo)

    return imp.astype(np.float32)


def plot_curves(out_path: str, p_pred_class: np.ndarray, imp_norm: np.ndarray, ent: np.ndarray, title: str):
    x = np.arange(len(p_pred_class))
    plt.figure(figsize=(14, 6))
    plt.suptitle(title)

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(x, p_pred_class)
    ax1.set_ylabel("P(pred class)")
    ax1.grid(True, alpha=0.2)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(x, imp_norm, label="importance_norm")
    ax2.plot(x, ent, label="entropy", linestyle="--")
    ax2.set_ylabel("Importance / Entropy")
    ax2.set_xlabel("Frame index (after sampling)")
    ax2.grid(True, alpha=0.2)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()



def print_top_cases(op_records, topk: int = 2):
    """Print the top-K most confident TP/TN/FP/FN cases (by operation-level probabilities)."""
    tp, tn, fp, fn = [], [], [], []
    for r in op_records:
        if r["true"] == 1 and r["pred"] == 1:
            tp.append(r)
        elif r["true"] == 0 and r["pred"] == 0:
            tn.append(r)
        elif r["true"] == 0 and r["pred"] == 1:
            fp.append(r)
        elif r["true"] == 1 and r["pred"] == 0:
            fn.append(r)

    # Sort by confidence
    tp_sorted = sorted(tp, key=lambda x: x["prob1"], reverse=True)  # most confident positive correct
    fp_sorted = sorted(fp, key=lambda x: x["prob1"], reverse=True)  # most confident positive wrong
    tn_sorted = sorted(tn, key=lambda x: x["prob0"], reverse=True)  # most confident negative correct
    fn_sorted = sorted(fn, key=lambda x: x["prob1"])              # least confident positive (missed)

    def _print_group(name, group):
        print(f"\n{name}:")
        if not group:
            print("  None")
            return
        for r in group[:topk]:
            print(f"  {r['center']}/{r['op_id']} | true={r['true']} pred={r['pred']} prob1={r['prob1']:.4f} prob0={r['prob0']:.4f}")

    print("\n===== TOP CASES (op-level) =====")
    _print_group(f"Top {topk} TP", tp_sorted)
    _print_group(f"Top {topk} TN", tn_sorted)
    _print_group(f"Top {topk} FP", fp_sorted)
    _print_group(f"Top {topk} FN", fn_sorted)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--binary", action="store_true", help="Binary classification (2 classes)")
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    parser.add_argument("-d", "--data_folder", type=str, required=True)
    parser.add_argument("-m", "--model_folder", type=str, required=True, help="Model dir (contains best.pkl) or model file path")
    parser.add_argument("-c", "--csv_labels", type=str, default="", help="Optional labels CSV; will be copied into split folder if needed")

    parser.add_argument("--centers", type=str, default="portugal", help="Comma-separated center names")
    parser.add_argument("--split", type=str, default="val", help="Dataset split folder name, e.g. val/test")

    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lstm_size", type=int, default=160)

    parser.add_argument("--temporal", action="store_true", help="Use temporal=True model forward (matches original validation)")
    parser.add_argument("--middleframe", action="store_true", help="Only evaluate the middle frame")
    parser.add_argument("--skip", type=int, default=2, help="Skip factor used by dataloader. Use 0 for ALL frames.")
    parser.add_argument("--max_ops", type=int, default=0, help="If >0, process only first N ops (for debug)")

    # explainability
    parser.add_argument("--per_frame", action="store_true", help="Export per-frame CSV + optional plots/frames")
    parser.add_argument("--export_plots", action="store_true")
    parser.add_argument("--export_frames", action="store_true")
    parser.add_argument("--topk_frames", type=int, default=10)
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--use_leave_one_out", action="store_true",
                        help="Use leave-one-out importance (base - prob_without_frame). "
                             "If not set, importance_norm is computed from p(pred_class) shift-normalize.")

    args = parser.parse_args()

    ensure_dir(args.output_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # classes
    if args.binary:
        num_class = 2
        classes = [0, 1]
        binary = True
    else:
        num_class = 6
        classes = [0, 1, 2, 3, 4, 5]
        binary = False

    # model
    model_path = resolve_model_path(args.model_folder)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = Networks.PhaseLSTMConvNext(num_class, args.temporal, args.lstm_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # transforms
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    centers = [c.strip() for c in args.centers.split(",") if c.strip()]
    test_sets = []

    # build per-op datasets like original
    for center in centers:
        op_path = os.path.join(args.data_folder, center, args.split)
        if not os.path.isdir(op_path):
            raise FileNotFoundError(f"Split folder not found: {op_path}")

        # Ensure CSV exists inside split folder for AppendectomyDataset
        if args.csv_labels:
            csv_in_split = glob.glob(os.path.join(op_path, "*.csv"))
            if len(csv_in_split) == 0:
                ensure_dir(op_path)
                dst = os.path.join(op_path, os.path.basename(args.csv_labels))
                print(f"Copy labels CSV into split folder: {args.csv_labels} -> {dst}")
                copy2(args.csv_labels, dst)

        ops = list_subdirs(op_path)
        ops.sort()
        for op_id in ops:
            ds = Dataloader.AppendectomyDataset(
                op_path,
                op_id,
                width=args.width,
                height=args.height,
                transform=transform_test,
                middleframe=args.middleframe,
                skip=args.skip,
                binary=binary,
            )
            test_sets.append((center, op_id, ds))

    if args.max_ops and args.max_ops > 0:
        test_sets = test_sets[: int(args.max_ops)]

    # global accumulators (op-level)
    aggregated_y_test_labels = []
    aggregated_y_pred_labels = []
    aggregated_y_pred = []  # avg logits per op (C,)

    # for printing top TP/TN/FP/FN cases
    op_records = []

    per_frame_root = os.path.join(args.output_folder, "per_frame")
    if args.per_frame:
        ensure_dir(per_frame_root)

    # evaluate op-by-op (so we can export per-frame)
    for center, op_id, dataset in test_sets:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        label_counter = Counter()
        pred_counter = Counter()
        frame_logits_list = []  # (B,C) chunks, concatenated into (T,C)

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                # forward like original
                if args.temporal:
                    outputs, _hidden = model(images, None)
                else:
                    outputs = model(images, None)

                preds = torch.argmax(outputs, dim=1)

                label_counter.update(labels.cpu().numpy().tolist())
                pred_counter.update(preds.cpu().numpy().tolist())
                frame_logits_list.append(outputs.cpu().numpy())

        if len(frame_logits_list) == 0:
            print(f"[WARN] Empty op: {center}/{op_id}")
            continue

        frame_logits = np.concatenate(frame_logits_list, axis=0)  # (T,C)
        T, C = frame_logits.shape

        # op-level aggregation (matches original)
        most_common_label = label_counter.most_common(1)[0][0]
        avg_logits = frame_logits.mean(axis=0)  # (C,)
        op_pred_class = int(np.argmax(avg_logits))

        op_probs = softmax_np(avg_logits)  # (C,)
        prob0 = float(op_probs[0])
        prob1 = float(op_probs[1]) if C > 1 else 0.0

        op_records.append({
            "center": center,
            "op_id": op_id,
            "true": int(most_common_label),
            "pred": int(op_pred_class),
            "prob0": prob0,
            "prob1": prob1,
        })

        aggregated_y_test_labels.append(int(most_common_label))
        aggregated_y_pred_labels.append(op_pred_class)
        aggregated_y_pred.append(avg_logits)

        if args.per_frame:
            # per-frame probs, entropy
            probs = softmax_np(frame_logits, axis=1)  # (T,C)
            ent = entropy_np(probs, axis=1).astype(np.float32)  # (T,)
            p_pred_class = probs[:, op_pred_class].astype(np.float32)  # (T,)

            # two choices for "importance"
            if args.use_leave_one_out:
                imp = leave_one_out_importance(frame_logits, op_pred_class)  # (T,)
                imp_sm = moving_average(imp, args.smooth_window)
                imp_norm = importance_norm_from_scores(imp_sm)
                imp_kind = "leave_one_out_smoothed"
            else:
                # same as your earlier script: normalize from p_pred_class
                imp_norm = importance_norm_from_scores(p_pred_class)
                imp_kind = "p_pred_class_shift_norm"
                imp = None
                imp_sm = None

            # prepare output folder
            out_dir = os.path.join(per_frame_root, f"{center}_{op_id}")
            ensure_dir(out_dir)

            # write CSV
            # keep your familiar columns + extend for multi-class if needed
            rows = []
            for t in range(T):
                row = {
                    "frame_idx": t,
                    "true_label": int(most_common_label),
                    "op_pred_class": int(op_pred_class),
                    "p_op_pred_class": float(p_pred_class[t]),
                    "importance_norm": float(imp_norm[t]),
                    "entropy": float(ent[t]),
                }
                # logits/probs for each class
                for ci in range(C):
                    row[f"logit_c{ci}"] = float(frame_logits[t, ci])
                for ci in range(C):
                    row[f"prob_c{ci}"] = float(probs[t, ci])

                # optional causal importance values
                if imp is not None:
                    row["importance_leave_one_out"] = float(imp[t])
                    row["importance_leave_one_out_smoothed"] = float(imp_sm[t])
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(out_dir, "per_frame_predictions.csv"), index=False)

            # save a small op summary json
            op_prob = float(softmax_np(avg_logits)[op_pred_class])
            summary = {
                "center": center,
                "op_id": op_id,
                "num_frames_used": int(T),
                "skip": int(args.skip),
                "middleframe": bool(args.middleframe),
                "temporal": bool(args.temporal),
                "true_label": int(most_common_label),
                "op_pred_class": int(op_pred_class),
                "p_op_pred_class": op_prob,
                "importance_mode": imp_kind,
            }
            with open(os.path.join(out_dir, "summary.json"), "w") as f:
                import json
                json.dump(summary, f, indent=2)

            # plot
            if args.export_plots:
                title = f"{center} {args.skip if args.skip>0 else 'all'} frames, {'temporal' if args.temporal else 'notemporal'} | {op_id} | true={most_common_label} pred={op_pred_class}"
                plot_curves(
                    os.path.join(out_dir, "importance_curve.png"),
                    p_pred_class=p_pred_class,
                    imp_norm=imp_norm,
                    ent=ent,
                    title=title,
                )

            # export topk frames: copy original PNGs (best fidelity)
            if args.export_frames and args.topk_frames > 0:
                k = min(int(args.topk_frames), T)
                # importance ranks
                top_idx = np.argsort(-imp_norm)[:k]
                bot_idx = np.argsort(imp_norm)[:k]

                pos_dir = os.path.join(out_dir, "topk_positive")
                neg_dir = os.path.join(out_dir, "topk_negative")
                ensure_dir(pos_dir)
                ensure_dir(neg_dir)

                # dataset.images is a manager.list of (path, PIL_or_None)
                # path is at [i][0]
                for rank, idx in enumerate(top_idx.tolist(), start=1):
                    src = dataset.images[idx][0]
                    dst = os.path.join(pos_dir, f"rank{rank:02d}_frame{idx:04d}_imp{imp_norm[idx]:.6f}.png")
                    try:
                        copy2(src, dst)
                    except Exception:
                        pass

                for rank, idx in enumerate(bot_idx.tolist(), start=1):
                    src = dataset.images[idx][0]
                    dst = os.path.join(neg_dir, f"rank{rank:02d}_frame{idx:04d}_imp{imp_norm[idx]:.6f}.png")
                    try:
                        copy2(src, dst)
                    except Exception:
                        pass

    # -------------------------
    # Global metrics (op-level)
    # -------------------------
    y_test_labels = np.array(aggregated_y_test_labels).reshape(-1, 1)
    y_pred_labels = np.array(aggregated_y_pred_labels).reshape(-1, 1)
    y_pred = np.array(aggregated_y_pred)  # (N,C)

    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit(np.array(classes).reshape(-1, 1))
    y_test = ohe.transform(y_test_labels)

    # classification report
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(args.output_folder, "classification_report.csv"))

    # AUROC (one-vs-rest)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(y_test.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    pd.DataFrame.from_dict(roc_auc, orient="index", columns=["value"]).to_csv(os.path.join(args.output_folder, "auroc.csv"))

    # confusion matrix + f1
    cm = confusion_matrix(y_test_labels.flatten(), y_pred_labels.flatten(), labels=classes)
    np.savetxt(os.path.join(args.output_folder, "confusion_matrix.txt"), cm, fmt="%d")
    f1 = f1_score(y_test_labels.flatten(), y_pred_labels.flatten(), average="macro")
    with open(os.path.join(args.output_folder, "summary.txt"), "w") as f:
        f.write(f"num_ops={len(aggregated_y_test_labels)}\n")
        f.write(f"f1_macro={f1:.6f}\n")
        f.write(f"model_path={model_path}\n")
        f.write(f"skip={args.skip} middleframe={args.middleframe} temporal={args.temporal}\n")

    # print top TP/TN/FP/FN case names
    try:
        print_top_cases(op_records, topk=2)
    except Exception as e:
        print("[WARN] Failed to print top cases:", e)

    print("Done. Outputs in:", args.output_folder)
    if args.per_frame:
        print("Per-frame outputs in:", per_frame_root)


if __name__ == "__main__":
    main()
