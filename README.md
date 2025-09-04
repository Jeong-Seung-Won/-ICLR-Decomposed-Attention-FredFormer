This repository provides an experimental pipeline to train, validate, and test multiple model configurations registered in `CFG_REGISTRY`, and to compare their performance (MSE/MAE). The README is written for reproducibility and clarity for conference or paper submissions.

## Overview
- `main.py`:
  - Instantiates each model class registered in `MODEL_ZOO` and `CFG_REGISTRY`, and prints parameter counts.
  - Runs training (`train_epoch`), validation (`eval_epoch`), and early stopping.
  - Saves the best checkpoint as `best_{model_key}_5B.pt`.
  - Reloads the best checkpoint and evaluates on the test set (`eval_epoch`).
  - Prints per-model metrics and a final summary.
- Data loaders and train/eval loops are implemented in `Foundation_Model_utils.py`.

## Requirements
- Python ≥ 3.9 (3.10/3.11 recommended)
- PyTorch ≥ 2.0, CUDA (optional but recommended)
- Others: `numpy`, `tqdm`, optionally `torchvision`, etc.

Example setup:
```bash
# conda (recommended)
conda create -n fm-env python=3.10 -y
conda activate fm-env

# PyTorch with CUDA 12.x wheels (pick the correct URL for your CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# additional Python packages as needed
pip install numpy tqdm
```

## Project Structure
- `main.py`: Experiment driver (train/val/test loop)
- `config.py`: Model registry (`MODEL_ZOO`) and configuration registry (`CFG_REGISTRY`)
- `Foundation_Model_utils.py`: Data loaders (`build_*_loader`) and train/eval functions
- `Foundation_Model_ours_tensor_train.py`: Model implementation example (actual class is referenced via `MODEL_ZOO`)

## Data Preparation
Implement/configure these functions in `Foundation_Model_utils.py` for your dataset:
- `build_train_loader()`
- `build_val_loader()`
- `build_test_loader()`

Recommendations:
- Clearly document and fix your preprocessing/normalization pipeline for reproducibility.
- If you need denormalized evaluation (`eval_epoch_denorm`), keep the statistics and paths versioned.

## Configurations
Models and configs are registered in `config.py`:
- `MODEL_ZOO`: `{key: ModelClass}`
- `CFG_REGISTRY`: `{key: config}` (hyperparameters, architecture options, etc.)

To add a new model:
1. Implement your model class and make it importable.
2. Register: `MODEL_ZOO["YourModelKey"] = YourModelClass`
3. Add config: `CFG_REGISTRY["YourModelKey"] = your_cfg`

## Run
```bash
python main.py
```

Example console output:
```text
Sanity-checking model instantiation
  ModelA          →  12,345,678 parameters
  ModelB          →  9,876,543 parameters
Train batches : XXX
Val   batches : XXX
Test  batches : XXX
[ModelA] Ep 01/50 | Train MSE 0.12345678 | Val MSE 0.10000000 | 12.34s
...
early stop (ModelA)
Finished ModelA         → Test MSE: 0.09876543  |  MAE: 0.01234567
...
Summary (lowest val MSE):
ModelA           Val MSE 0.10000000 | Test MSE 0.09876543  MAE 0.01234567
...
Best model: {"model": "ModelA", "val": 0.1000, "test_mse": 0.0988, "test_mae": 0.0123}
```

## Training Setup and Hyperparameters
Defaults in `main.py`:
- Optimizer: `Adam(lr=1e-4)`
- Scheduler: `CosineAnnealingLR(T_max=len(train_loader)*EPOCHS)`
- Epochs: `50`
- Early Stopping Patience: `5`
- Loss: `MSELoss` (report `L1Loss` as MAE)
- AMP: Enabled when CUDA is available (`GradScaler`)

Tune constants `LR`, `EPOCHS`, `EARLY_PATIENCE` in `main.py` as needed.

## Checkpoints and Logs
- Best weights per model key are saved at project root: `best_{key}_5B.pt`
- When loading multi-GPU checkpoints (`nn.DataParallel`), the `"module."` prefix is stripped automatically.
- Logs are printed to stdout; redirect to a file if desired.

## GPU and Multi-GPU
- `main.py` currently sets `CUDA_VISIBLE_DEVICES` in-code; prefer setting it from the shell:
  - Example: `CUDA_VISIBLE_DEVICES=0 python main.py`
- If multiple GPUs are visible, `nn.DataParallel` is used automatically.

## Reproducibility
Seeding is not fully enforced by default. For strict reproducibility, add:
```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Reporting (for submissions)
- Summarize per-model `Val/Test MSE/MAE` in a table.
- Select the best model by lowest `Val MSE` and report its `Test` performance.
- Document data splits and preprocessing; include hardware/software environment (GPU, CUDA, PyTorch, driver/OS).

## FAQ
- Q: Why is the checkpoint file named `best_{key}_5B.pt`?
  - A: It’s the current default. Change the filename string in `main.py` if needed.
- Q: Where are data paths configured?
  - A: Inside `Foundation_Model_utils.py` `build_*_loader` functions or via configs added in `config.py`.
- Q: How does AMP work here?
  - A: It’s enabled automatically with CUDA (`GradScaler(enabled=DEVICE.startswith("cuda"))`); disabled on CPU.

## Citation
If you use this repository, please cite:
```bibtex
@misc{foundation_model_training_2025,
  title        = {Foundation Model Training & Evaluation},
  author       = {Your Name and Co-authors},
  year         = {2025},
  howpublished = {GitHub repository},
  note         = {\url{https://github.com/your-repo}}
}
```

## Contact
- Maintainer: Your Name <your.email@domain>
- Issues/Bugs: via GitHub Issues or email

## License
- See `LICENSE` for the distribution license (e.g., MIT).
