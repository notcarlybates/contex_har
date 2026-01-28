#!/usr/bin/env python3
"""
Prepare PAAWS data for DeepConvContext model training.

Single script that:
  A) Syncs raw accelerometer + label CSVs into model-ready format
  B) Generates LOSO JSON annotation files
  C) Auto-generates a YAML training config

Usage:
    python prepare_paaws.py --paaws-dir /mnt/storage/for_release
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from glob import glob

import polars as pl
import yaml
from tqdm import tqdm


# Labels that should be treated as null (unlabeled)
NULL_LABELS = {"Video_Unavailable", "Indecipherable"}


# ──────────────────────────────────────────────────────────────────────
# Step A helpers: Actigraph parsing & label synchronisation
# ──────────────────────────────────────────────────────────────────────

def read_actigraph(filepath):
    """Read ActiGraph CSV with 10-line header. Returns (start_datetime, sampling_rate, DataFrame)."""
    with open(filepath) as f:
        line1 = f.readline()
        match = re.search(r"(\d+)\s+Hz", line1)
        sampling_rate = int(match.group(1)) if match else 80

        f.readline()  # serial number
        start_time = f.readline().split()[-1]  # "Start Time HH:MM:SS"
        start_date = f.readline().split()[-1]  # "Start Date M/D/YYYY"

    start = datetime.strptime(f"{start_date} {start_time}", "%m/%d/%Y %H:%M:%S")
    df = pl.read_csv(filepath, skip_rows=10, has_header=True)
    step = timedelta(seconds=1 / sampling_rate)
    timestamps = [start + i * step for i in range(len(df))]
    df = df.with_columns(pl.Series("Timestamp", timestamps))
    return start, sampling_rate, df


def sync_labels(accel_df, label_df, label_column="PA_TYPE"):
    """Assign activity labels to accelerometer rows by timestamp overlap.

    Rows between the first START_TIME and last STOP_TIME that fall outside
    any annotation segment receive the label ``"null"``.
    """
    label_df = label_df.with_columns(
        pl.col("START_TIME").str.to_datetime("%m/%d/%Y %H:%M:%S%.f", strict=False),
        pl.col("STOP_TIME").str.to_datetime("%m/%d/%Y %H:%M:%S%.f", strict=False),
    )

    first_label = label_df["START_TIME"][0]
    last_label = label_df["STOP_TIME"][-1]

    # Trim to label time range
    accel_df = accel_df.filter(
        (pl.col("Timestamp") >= first_label) & (pl.col("Timestamp") <= last_label)
    )

    # Start with all "null"
    label_col = pl.lit("null")

    # Build a chained when/then for each label row (last match wins)
    for row in label_df.iter_rows(named=True):
        lbl = row[label_column]
        assigned = "null" if lbl in NULL_LABELS else lbl
        mask = (pl.col("Timestamp") >= row["START_TIME"]) & (
            pl.col("Timestamp") <= row["STOP_TIME"]
        )
        label_col = pl.when(mask).then(pl.lit(assigned)).otherwise(label_col)

    accel_df = accel_df.with_columns(label_col.alias("label"))

    return accel_df


def write_model_csv(accel_df, sbj_id, output_path):
    """Write model-format CSV: sbj_id, acc_1, acc_2, acc_3, label."""
    output = pl.DataFrame(
        {
            "sbj_id": [sbj_id] * len(accel_df),
            "acc_1": accel_df["Accelerometer X"],
            "acc_2": accel_df["Accelerometer Y"],
            "acc_3": accel_df["Accelerometer Z"],
            "label": accel_df["label"],
        }
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.write_csv(output_path)
    return len(output)


# ──────────────────────────────────────────────────────────────────────
# Step B helpers: LOSO JSON generation
# ──────────────────────────────────────────────────────────────────────

def parse_timestamp(ts_str):
    """Parse a timestamp string into a datetime object."""
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(str(ts_str).strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse timestamp: {ts_str}")


def collect_unique_labels(label_files, label_column):
    """Scan all label CSVs and return an ordered label_mapping (null → 0, then alphabetical)."""
    unique = set()
    for path in label_files:
        df = pl.read_csv(path)
        if label_column in df.columns:
            unique.update(df[label_column].drop_nulls().unique().to_list())

    unique -= NULL_LABELS
    mapping = {}
    for i, lbl in enumerate(sorted(unique), start=1):
        mapping[lbl] = i
    return mapping


def build_subject_annotations(label_path, label_column, label_mapping, fps):
    """Build annotation list for one subject from its label CSV.

    Returns (annotations_list, duration_seconds).
    """
    df = pl.read_csv(label_path)
    ref_time = parse_timestamp(df["START_TIME"][0])

    annotations = []
    for row in df.iter_rows(named=True):
        start_sec = (parse_timestamp(row["START_TIME"]) - ref_time).total_seconds()
        stop_sec = (parse_timestamp(row["STOP_TIME"]) - ref_time).total_seconds()
        label = row[label_column]
        if label is None or label in NULL_LABELS:
            label = "null"
        label_id = label_mapping.get(label, 0)

        length = stop_sec - start_sec
        annotations.append(
            {
                "segment": [start_sec, stop_sec],
                "segment (frames)": [start_sec * fps, stop_sec * fps],
                "label_id": label_id,
                "label": label,
                "length": length,
            }
        )

    duration = (
        parse_timestamp(df["STOP_TIME"][-1]) - ref_time
    ).total_seconds()
    return annotations, duration


def generate_loso_files(subjects_data, label_mapping, output_dir):
    """Generate one LOSO JSON per subject.

    ``subjects_data`` is a dict: sbj_id -> {annotations, duration, fps}.
    """
    os.makedirs(output_dir, exist_ok=True)
    subject_ids = sorted(subjects_data.keys())
    paths = []

    # label_dict excludes null (main.py prepends it when has_null=True)
    label_dict = {lbl: lid for lbl, lid in label_mapping.items()}

    for val_sbj in subject_ids:
        database = {}
        for sbj in subject_ids:
            info = subjects_data[sbj]
            database[sbj] = {
                "subset": "Validation" if sbj == val_sbj else "Training",
                "duration": info["duration"],
                "fps": info["fps"],
                "annotations": info["annotations"],
            }

        json_data = {
            "version": [],
            "taxonomy": [],
            "database": database,
            "label_dict": label_dict,
        }

        out_path = os.path.join(output_dir, f"loso_{val_sbj}.json")
        with open(out_path, "w") as f:
            json.dump(json_data, f, indent=2)
        paths.append(out_path)
        print(f"  Generated {os.path.basename(out_path)} (validation: {val_sbj})")

    return paths


# ──────────────────────────────────────────────────────────────────────
# Step C: YAML config generation
# ──────────────────────────────────────────────────────────────────────

def generate_yaml_config(anno_json_paths, num_classes, output_path):
    """Write a DeepConvContext YAML config for PAAWS."""
    config = {
        "name": "deepconvcontext",
        "dataset_name": "paaws",
        "has_null": True,
        "anno_json": [str(p) for p in anno_json_paths],
        "dataset": {
            "sens_folder": "./data/paaws/raw/inertial",
            "input_dim": 3,
            "sampling_rate": 80,
            "num_classes": num_classes,
            "window_size": 80,
            "window_overlap": 50,
            "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        },
        "model": {
            "conv_kernels": 64,
            "conv_kernel_size": 9,
            "lstm_units": 128,
            "lstm_layers": 1,
            "dropout": 0.5,
            "type": "lstm",
            "bidirectional": False,
            "attention_num_heads": 4,
            "transformer_depth": 3,
        },
        "train_cfg": {
            "lr": 0.0001,
            "lr_decay": 0.9,
            "lr_step": 10,
            "epochs": 30,
            "weight_decay": 0.000001,
            "weight_init": "xavier_normal",
            "weighted_loss": True,
        },
        "loader": {
            "train_batch_size": 100,
            "test_batch_size": 100,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Config written to {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def find_accel_file(ds_folder, sbj_id, accel_pattern):
    """Locate the accelerometer CSV inside a DS folder.

    Tries the explicit pattern first, then falls back to a glob search.
    """
    explicit = os.path.join(ds_folder, accel_pattern.format(id=sbj_id))
    if os.path.exists(explicit):
        return explicit

    # Fallback: look in accel/ subfolder
    alt = os.path.join(ds_folder, "accel", accel_pattern.format(id=sbj_id))
    if os.path.exists(alt):
        return alt

    # Glob fallback
    candidates = glob(os.path.join(ds_folder, "**", "*.csv"), recursive=True)
    for c in candidates:
        if "label" not in os.path.basename(c).lower() and c.endswith(".csv"):
            base = os.path.basename(c)
            if "Accelerometer" not in base and "LeftWrist" in base:
                return c
            if "LeftWrist" in base:
                return c

    return None


def find_label_file(ds_folder, sbj_id, file_tag="Lab"):
    """Locate the label CSV inside a DS folder."""
    explicit = os.path.join(ds_folder, "label", f"DS_{sbj_id}-{file_tag}-label.csv")
    if os.path.exists(explicit):
        return explicit

    # Glob fallback
    candidates = glob(os.path.join(ds_folder, "label", "*-label.csv"))
    if candidates:
        return candidates[0]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PAAWS data for DeepConvContext training."
    )
    parser.add_argument(
        "--paaws-dir",
        type=str,
        default="/mnt/storage/for_release",
        help="Path to PAAWS data directory (default: /mnt/storage/for_release)",
    )
    parser.add_argument(
        "--accel-pattern",
        type=str,
        default="DS_{id}-Lab-LeftWristTop.csv",
        help="Accelerometer filename pattern (default: DS_{id}-Lab-LeftWristTop.csv)",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="PA_TYPE",
        help="Column for activity labels (default: PA_TYPE)",
    )
    parser.add_argument(
        "--null-class",
        type=str,
        default="null",
        help="Null class label (default: null)",
    )
    args = parser.parse_args()

    paaws_dir = args.paaws_dir
    inertial_dir = os.path.join("data", "paaws", "raw", "inertial")
    annotations_dir = os.path.join("data", "paaws", "annotations")
    config_path = "configs/main_experiments/deepconvcontext/paaws_loso_lstm.yaml"

    # ── Discover subjects ──────────────────────────────────────────
    SUBDIRS = [
        # (directory_name, id_prefix, file_tag)
        ("PAAWS_FreeLiving", "fl", "Free"),
        ("PAAWS_SimFL_Lab", "lab", "Lab"),
    ]

    ds_entries = []  # list of (ds_folder_path, prefix, file_tag)
    for subdir_name, prefix, file_tag in SUBDIRS:
        subdir_path = os.path.join(paaws_dir, subdir_name)
        if os.path.isdir(subdir_path):
            for folder in sorted(glob(os.path.join(subdir_path, "DS_*"))):
                ds_entries.append((folder, prefix, file_tag))

    if not ds_entries:
        print(f"No DS_* folders found in subdirectories of {paaws_dir}")
        sys.exit(1)

    print(f"Found {len(ds_entries)} subject folder(s)")

    # ── Step A: Process each subject ────────────────────────────────
    print("\n=== Step A: Syncing accelerometer data with labels ===")
    processed = {}  # sbj_id -> {label_path, sampling_rate}

    for ds_folder, prefix, file_tag in tqdm(ds_entries, desc="Step A: Syncing subjects"):
        folder_name = os.path.basename(ds_folder)
        match = re.match(r"DS_(\d+)", folder_name)
        if not match:
            tqdm.write(f"  Warning: skipping {folder_name} (cannot extract ID)")
            continue
        sbj_num = match.group(1)
        sbj_id = f"sbj_{prefix}_{sbj_num}"

        accel_pattern = args.accel_pattern.replace("Lab", file_tag)
        accel_path = find_accel_file(ds_folder, sbj_num, accel_pattern)
        if accel_path is None:
            tqdm.write(f"  Warning: no accelerometer file for {folder_name}, skipping")
            continue

        label_path = find_label_file(ds_folder, sbj_num, file_tag)
        if label_path is None:
            tqdm.write(f"  Warning: no label file for {folder_name}, skipping")
            continue

        # Read accelerometer
        start_dt, sampling_rate, accel_df = read_actigraph(accel_path)

        # Read labels
        label_df = pl.read_csv(label_path)

        # Sync
        synced = sync_labels(accel_df, label_df, label_column=args.label_column)

        # Write CSV
        out_csv = os.path.join(inertial_dir, f"{sbj_id}.csv")
        n_rows = write_model_csv(synced, sbj_num, out_csv)
        tqdm.write(f"  {sbj_id}: {n_rows} rows -> {out_csv}")

        processed[sbj_id] = {
            "label_path": label_path,
            "sampling_rate": sampling_rate,
        }

    if not processed:
        print("No subjects processed successfully. Exiting.")
        sys.exit(1)

    # ── Step B: Generate LOSO annotations ───────────────────────────
    print("\n=== Step B: Generating LOSO JSON annotations ===")

    label_files = [info["label_path"] for info in processed.values()]
    fps = next(iter(processed.values()))["sampling_rate"]

    label_mapping = collect_unique_labels(label_files, args.label_column)
    num_activity_classes = len(label_mapping)  # excludes null
    # +1 for the null class
    num_classes = num_activity_classes + 1

    print(f"  Found {num_activity_classes} activity classes (+1 null = {num_classes} total):")
    for lbl, lid in sorted(label_mapping.items(), key=lambda x: x[1]):
        print(f"    {lid}: {lbl}")

    # Save label mapping
    os.makedirs(annotations_dir, exist_ok=True)
    mapping_path = os.path.join(annotations_dir, "label_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({"null": 0, **label_mapping}, f, indent=2)
    print(f"  Label mapping saved to {mapping_path}")

    # Build per-subject annotation data
    subjects_data = {}
    for sbj_id, info in processed.items():
        annotations, duration = build_subject_annotations(
            info["label_path"], args.label_column, label_mapping, fps
        )
        subjects_data[sbj_id] = {
            "annotations": annotations,
            "duration": duration,
            "fps": fps,
        }

    anno_paths = generate_loso_files(subjects_data, label_mapping, annotations_dir)

    if len(processed) < 2:
        print(
            f"\n  WARNING: Only {len(processed)} subject(s) found. "
            "LOSO cross-validation requires 2+ subjects for meaningful training."
        )

    # ── Step C: Generate YAML config ────────────────────────────────
    print("\n=== Step C: Generating training config ===")
    generate_yaml_config(anno_paths, num_classes, config_path)

    print("\n=== Done! ===")
    print(f"  Inertial CSVs:  {inertial_dir}/")
    print(f"  Annotations:    {annotations_dir}/")
    print(f"  Config:         {config_path}")
    print(f"\nTo train, run:")
    print(f"  python main.py --config {config_path} --seed 1 --ckpt-freq 10")


if __name__ == "__main__":
    main()
