# NASA CMAPSS Jet Engine Simulated Data

This project uses the FD001 dataset from the NASA CMAPSS turbofan engine degradation dataset.

## How to download:
1. Visit the Kaggle mirror: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
2. Download the archive.
3. Extract `train_FD001.txt` and `test_FD001.txt` and `RUL_FD001.txt`.
4. Place them directly in this `data/` directory.

The training script `models/train_models.py` will expect the file `data/train_FD001.txt` to exist.
