# Appendectomy video classification

GitHub-ready snapshot of the scripts used in this workflow, with the original training/inference files plus the final explainability-focused validation script.

## Included files

- `src/Dataloader.py` — original frame dataset loader.
- `src/Networks.py` — original ConvNeXt + optional LSTM model.
- `src/training.py` — original training script.
- `src/validation.py` — original validation script.
- `src/validation_ultimate.py` — final validation script for external deployment and explainability.
- `extras/validation_updated_video_dataset.py` — earlier experimental validation script that expects an `AppendectomyVideoDataset` implementation that is not present in the provided `Dataloader.py`.

## Recommended final validation script

Use `src/validation_ultimate.py` for external datasets.

What it adds:

- per-operation predictions
- per-frame probabilities
- normalized importance scores
- optional leave-one-out importance
- uncertainty via entropy
- top-K positive and negative frames exported
- importance plots
- top 2 TP / TN / FP / FN printed at the end

## Repository layout

```text
appendectomy-video-classification-final/
├── README.md
├── FOLDER_STRUCTURE.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── Dataloader.py
│   ├── Networks.py
│   ├── training.py
│   ├── validation.py
│   └── validation_ultimate.py
└── extras/
    └── validation_updated_video_dataset.py
```

## Expected data layout

The scripts expect a structure like:

```text
data/
├── center01/
│   ├── train/
│   │   ├── <op_id>/
│   │   │   ├── 0001.png
│   │   │   ├── 0002.png
│   │   │   └── ...
│   │   └── labels.csv
│   └── val/
│       ├── <op_id>/
│       └── labels.csv
├── center02/
└── portugal/
    └── val/
        ├── A1/
        ├── A2/
        └── portugal.csv
```

`AppendectomyDataset` looks for exactly one CSV inside the split folder and matches `row[1] == op_id`, then uses `row[2]` as the label.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
cd src
python3 -m training -b -o /path/to/output -d /path/to/data
```

## Original validation

```bash
cd src
python3 -m validation -b -o /path/to/validation -d /path/to/data -m /path/to/model_run/
```

## Final external validation

Example using all frames with explainability outputs:

```bash
cd src
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
```

For leave-one-out importance:

```bash
python3 -m validation_ultimate ... --use_leave_one_out --smooth_window 9
```

## Key notes

- `--skip 0` uses all frames.
- `--skip 2` keeps every third frame.
- Binary mapping in `Dataloader.py` is:
  - classes `0,1,2,3 -> 0`
  - classes `4,5 -> 1`
- `validation_ultimate.py` is the safest script to use with the provided codebase because it uses `AppendectomyDataset`, not the missing `AppendectomyVideoDataset`.

## Outputs from `validation_ultimate.py`

```text
output/
├── classification_report.csv
├── auroc.csv
├── confusion_matrix.txt
├── summary.txt
└── per_frame/
    └── <center>_<op_id>/
        ├── per_frame_predictions.csv
        ├── summary.json
        ├── importance_curve.png
        ├── topk_positive/
        └── topk_negative/
```

## GitHub upload suggestion

From the repo root:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```
