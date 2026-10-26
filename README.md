**3rd Place Solution to the MoCha 2026 Challenge at ECCV 2026**

<a href="https://arxiv.org/abs/2609.10187" target="_blank">
  <img src="https://img.shields.io/badge/arXiv-2609.10187-b31b1b.svg" alt="arXiv">
</a>

## Requirements

- Python 3.10+
- PyTorch, NumPy, pandas, transformers
- CUDA is recommended for full training; CPU is supported but slow.

`CARE-PD-pseudo` must contain `llm_captions/` and `metadata.csv`.

## Assets

Download the [`assets` package from Google Drive](https://drive.google.com/file/d/1stI4quzlBF0w9PCUVaN2uN2qDtfmsD7U/view?usp=sharing),
extract it, and place the resulting `assets` directory directly inside this
`pipeline` directory. The required layout is:

```text
pipeline/
├── assets/
│   ├── Best_submission.zip
│   ├── clip-vit-base-patch32/
│   └── template/
├── train.sh
├── test.sh
└── make_submission.sh
```

## Dataset

Download [`CARE-PD-pseudo` from Google Drive](https://drive.google.com/file/d/1tBL_SbSwfJ4133FRRZB5wTwLQPXvoVIc/view?usp=sharing)
and extract it into `data/CARE-PD-pseudo`. This is the default dataset
location used by both `train.sh` and `test.sh`:

```text
pipeline/
└── data/
    ├── cache/
    └── CARE-PD-pseudo/
        ├── llm_captions/
        └── metadata.csv
```

You can still pass a different dataset path as the first argument.

### 1. Train

This trains the four sources (PD-GaM, BMCLab, T-SDU-PD, and 3DGait) separately
and merges them into the final model.

```bash
bash train.sh
```

Custom dataset path:

```bash
bash train.sh /path/to/CARE-PD-pseudo
```

### 2. Test

After training, evaluate true-label samples only:

```bash
bash test.sh
```

To evaluate both true-label and pseudo-label samples:

```bash
bash test.sh data/CARE-PD-pseudo all
```

### 3. Build the submission

```bash
bash make_submission.sh
```

## Method and sources

- MoCha solution: [Language-Aligned Motion Representations for Domain-Generalizable UPDRS-Gait Severity Estimation](https://arxiv.org/abs/2609.10187)
- SCORE merging: [Bridging Domains through Subspace-Aware Model Merging](https://arxiv.org/abs/2603.05768)
- CARE-PD project: [CARE-PD official project page](https://neurips2025.care-pd.ca/)
- CARE-PD dataset: [Hugging Face](https://huggingface.co/datasets/vida-adl/CARE-PD/tree/main)


