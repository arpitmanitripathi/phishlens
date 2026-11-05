"""Load the dataset and split it into train / validation / test.

The split is GROUPED BY WEBSITE (the `domain` column): all URLs of one site land in the same
part. Otherwise the model could "recognise" a site it saw in training (e.g. 3 URLs of the same
phishing kit) and the test score would look better than it really is.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "dataset.csv"
SEED = 42


def load_dataset(path: Path = DATASET) -> pd.DataFrame:
    return pd.read_csv(path)


def split_dataset(df: pd.DataFrame, seed: int = SEED):
    """70 % train, 15 % validation, 15 % test (approximately: whole websites are moved).

    train      -> the models learn from it
    validation -> used to make choices (voting weight, early stopping, which model is best)
    test       -> touched ONCE at the very end, to report the final, honest score
    """
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, rest_idx = next(first.split(df, groups=df["domain"]))
    train, rest = df.iloc[train_idx], df.iloc[rest_idx]

    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(second.split(rest, groups=rest["domain"]))
    val, test = rest.iloc[val_idx], rest.iloc[test_idx]

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
