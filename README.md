# DeepConvContext: A Multi-Scale Approach to Timeseries Classification in Human Activtiy Recognition 

![DeepConvContext overview](architecture.jpg)

## Abstract
Despite recognized limitations in modeling long-range temporal dependencies, Human Activity Recognition (HAR) has traditionally relied on a sliding window approach to segment labeled datasets. Deep learning models like DeepConvLSTM typically classify each window independently, thereby restricting learnable temporal context to within-window information. To address this constraint, we propose DeepConvContext, a multi-scale time series classification framework for HAR. Drawing inspiration from the vision-based Temporal Action Localization community, DeepConvContext models both intra- and inter-window temporal patterns by processing sequences of time-ordered windows. Unlike recent HAR models that incorporate attention mechanisms, DeepConvContext relies solely on LSTMs. Our ablation studies demonstrate the superior performance of LSTMs over attention-based variants for modeling inertial sensor data. Benchmark evaluations across six widely-used HAR datasets show that DeepConvContext achieves up to a 10% improvement in F1-score over the original DeepConvLSTM.

## Additional Results Material
Additional confusion matrices of all mentioned experiments can be found in the `confusion_matrices` folder.

## Installation
Please follow instructions mentioned in the [INSTALL.md](/INSTALL.md) file.

## Download
The datasets used for conducting experiments can be downloaded [here](https://uni-siegen.sciebo.de/s/oikTt0G8kIDZvaL).

## PAAWS Data Processing

To process raw PAAWS accelerometer data into model-ready format, use `prepare_paaws.py`. This script:

1. Syncs accelerometer data (80 Hz ActiGraph CSVs) with activity labels by timestamp
2. Generates per-subject CSVs in `data/paaws/raw/inertial/`
3. Generates LOSO JSON annotations in `data/paaws/annotations/`
4. Auto-generates a YAML training config

### Usage

```
python prepare_paaws.py --paaws-dir /mnt/storage/for_release
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--paaws-dir` | `/mnt/storage/for_release` | Path to PAAWS parent directory containing `PAAWS_FreeLiving/` and `PAAWS_SimFL_Lab/` |
| `--accel-pattern` | `DS_{id}-Lab-LeftWristTop.csv` | Accelerometer filename pattern (`{id}` is replaced per subject) |
| `--label-column` | `PA_TYPE` | Column in label CSV to use as activity class |
| `--null-class` | `null` | Label assigned to unlabeled time regions |

### Expected Input Structure

```
/mnt/storage/for_release/
├── PAAWS_FreeLiving/
│   └── DS_<id>/
│       ├── DS_<id>-Free-LeftWristTop.csv  # ActiGraph accelerometer (10-line header, 80 Hz)
│       └── label/
│           └── DS_<id>-Free-label.csv     # Activity labels (START_TIME, STOP_TIME, PA_TYPE)
└── PAAWS_SimFL_Lab/
    └── DS_<id>/
        ├── DS_<id>-Lab-LeftWristTop.csv
        └── label/
            └── DS_<id>-Lab-label.csv
```

Subjects are prefixed `fl_` or `lab_` to disambiguate overlapping IDs across directories. New subjects added to either subdirectory are picked up automatically on re-run.

### One-Step Prepare + Train

```bash
bash scripts/prepare_and_train_paaws.sh
```

This runs `prepare_paaws.py` followed by `main.py` with the generated config.

## Reproduce Experiments
Once having installed requirements, one can rerun experiments by running the `main.py` script:

````
python main.py --config ./configs/baseline/main_experiments/deepconvcontext/hangtime_loso_lstm.yaml --seed 1
````

Each config file represents one type of experiment. Each experiment was run three times using three different random seeds (i.e. `1, 2, 3`). To rerun the experiments without changing anything about the config files, please place the complete dataset download into a folder called `data` in the main directory of the repository. The folder `job_scripts` contains collections of commands of all experiments

## Recompute metrics and figures
To recreate confusion matrices as well as compute scoring metrics mentioned in the paper, please run `compute_metrics.py`.

## Logging using Neptune.ai
In order to log experiments to [Neptune.ai](https://neptune.ai) please provide `project` and `api_token` information in your local deployment (see `main.py`)

## Contact
Marius Bock (marius.bock(at)uni-siegen.de)
