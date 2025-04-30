"""
dataloader.py
Fully self-contained loader for 5-D channel cubes
(user, Rx, Subcarrier, Tx, time).

New normalisation methods added:
  • global_max
  • sqrt_min_max
  • clip_db
Author: <you>
"""

import os
import random
from typing import Tuple, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# -----------------------------------------------------------
# 1)  Low-level helpers
# -----------------------------------------------------------

def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def _print_range(name: str, arr: np.ndarray) -> None:
    print(f"{name}: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

# -----------------------------------------------------------
# 2)  Main Dataset
# -----------------------------------------------------------

class ChannelSequenceDataset(Dataset):
    """
    Returns   inp : (Rx, Sub, Tx, seq_len)
              out : (Rx, Sub, Tx)
    for one user / one starting time index.
    """

    def __init__(
        self,
        file_prefix: str,
        file_extension: str,
        device: torch.device,
        seq_len: int = 64,
        normalization: str = "min_max",
        per_user: bool = False,
        clip_db_floor: float = -100.0,
        clip_db_ceil: float = -30.0,
        eps: float = 1e-9,
    ):
        """
        Args
        ----
        file_prefix   path *without* extension ('.mat' or '.npy' will be appended)
        file_extension  'mat' | 'npy'
        seq_len         input window length
        normalization   see table above
        per_user        apply scaling per user if True, else on the whole cube
        clip_db_floor   only for 'clip_db'
        clip_db_ceil    only for 'clip_db'
        eps             small constant to avoid log(0)
        """
        assert normalization in {
            "min_max",
            "z_score",
            "log_min_max",
            "robust",
            "global_max",
            "sqrt_min_max",
            "clip_db",
        }, f"unknown normalisation {normalization}"

        self.device        = device
        self.normalization = normalization
        self.per_user      = per_user
        self.seq_len       = seq_len
        self.clip_db_floor = clip_db_floor
        self.clip_db_ceil  = clip_db_ceil
        self.eps           = eps

        # ---------- load raw complex cube ----------
        file_path = file_prefix + file_extension
        if file_extension == "npy":
            raw = np.load(file_path, mmap_mode="r")          # (U,R,S,T,L)
        elif file_extension == "mat":
            with h5py.File(file_path, "r") as f:
                if "channel_matrix" in f:
                    grp = f["channel_matrix"]
                    real = np.array(grp["real"])
                    imag = np.array(grp["imag"])
                else:                                        # fallback
                    real = np.array(f["real"])
                    imag = np.array(f["imag"])
            raw = real + 1j * imag
        else:
            raise ValueError("file_extension must be 'mat' or 'npy'")

        # ---------- inspect ----------
        self.num_users, self.n_rx, self.n_sub, self.n_tx, self.time_len = raw.shape
        print("Loaded raw cube :", raw.shape)

        # ---------- magnitude ----------
        mag = np.abs(raw).astype(np.float32)

        # ---------- apply chosen scaling ----------
        self.data = (
            self._scale_per_user(mag) if per_user else self._scale_global(mag)
        )
        _print_range("After scaling", self.data)

        # ---------- info for caller ----------
        self.samples_per_user = self.time_len - (self.seq_len + 1)

    # ---------------------------------------------------------
    # 2-A  normalisation helpers
    # ---------------------------------------------------------

    def _scale_per_user(self, mag: np.ndarray) -> np.ndarray:
        out = np.empty_like(mag, dtype=np.float32)
        for u in range(self.num_users):
            out[u] = self._apply_scaler(mag[u])
        return out

    def _scale_global(self, mag: np.ndarray) -> np.ndarray:
        return self._apply_scaler(mag)

    def _apply_scaler(self, x: np.ndarray) -> np.ndarray:
        """Return x scaled to [0,1] (or z-scored) according to self.normalization."""
        if self.normalization == "min_max":
            return self._min_max(x)
        if self.normalization == "z_score":
            return self._z_score(x)
        if self.normalization == "log_min_max":
            return self._log_min_max(x)
        if self.normalization == "robust":
            return self._robust(x)
        if self.normalization == "global_max":
            return x / (x.max() + self.eps)
        if self.normalization == "sqrt_min_max":
            return self._min_max(np.sqrt(x))
        if self.normalization == "clip_db":
            return self._clip_db(x)
        raise RuntimeError("unreachable")

    # ---- concrete scalers ----
    def _min_max(self, x):
        a, b = x.min(), x.max()
        if b == a:
            return np.zeros_like(x)      # silent channel
        return (x - a) / (b - a)

    def _z_score(self, x):
        mu, sigma = x.mean(), x.std() + self.eps
        return (x - mu) / sigma

    def _log_min_max(self, x):
        logx = np.log10(x + self.eps)
        return self._min_max(logx)

    def _robust(self, x):
        q25, q75 = np.percentile(x, [25, 75])
        iqr      = q75 - q25 if (q75 - q25) > self.eps else self.eps
        return np.clip((x - q25) / iqr, 0, 1)

    def _clip_db(self, x):
        y = 20.0 * np.log10(x + self.eps)
        y = np.clip(y, self.clip_db_floor, self.clip_db_ceil)
        return (y - self.clip_db_floor) / (self.clip_db_ceil - self.clip_db_floor)

    # ---------------------------------------------------------
    # 2-B  PyTorch dataset API
    # ---------------------------------------------------------

    def __len__(self) -> int:
        return self.num_users * self.samples_per_user

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        user   = idx // self.samples_per_user
        t0     = idx %  self.samples_per_user
        inp = self.data[user, :, :, :, t0 : t0 + self.seq_len]
        out = self.data[user, :, :, :, t0 + self.seq_len]
        return _to_tensor(inp, self.device), _to_tensor(out, self.device)

    # ---------------------------------------------------------
    # 2-C  expose original stats (optional)
    # ---------------------------------------------------------

    def get_stats(self) -> Dict[str, float]:
        """handy for inverse-transform later"""
        return {
            "normalization": self.normalization,
            "per_user": self.per_user,
        }

# -----------------------------------------------------------
# 3)  Convenience loader for your 3 datasets
# -----------------------------------------------------------

def split_dataset(dataset: Dataset, split_ratio: float = 0.8):
    n = len(dataset)
    n_train = int(n * split_ratio)
    n_test  = n - n_train
    return random_split(dataset, [n_train, n_test])

def _load_one(
    path_prefix: str,
    batch_size : int,
    device     : torch.device,
    **ds_kwargs,
):
    ds  = ChannelSequenceDataset(path_prefix, "mat", device, **ds_kwargs)
    trn, tst = split_dataset(ds, 0.8)
    return (
        trn,
        tst,
        DataLoader(trn, batch_size, shuffle=True,  drop_last=True),
        DataLoader(tst, batch_size, shuffle=False, drop_last=False),
    )

def get_all_datasets(
    data_dir: str,
    batch_size: int = 16,
    dataset_id = 1,
    **ds_kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names  = {
        1: "umi_fixed_compact_8Tx_2Rx.",
        2: "umi_fixed_dense_8Tx_2Rx.",
        3: "umi_fixed_standard_8Tx_2Rx.",
    }

    def maybe_load(i):
        if dataset_id in (i, "all"):
            path = os.path.join(data_dir, names[i])
            print(f"→ Loading Dataset {i}: {path}mat")
            return _load_one(path, batch_size, device, **ds_kwargs)
        return (None,) * 4

    return (*maybe_load(1), *maybe_load(2), *maybe_load(3))

# -----------------------------------------------------------
# 4)  Quick self-test
# -----------------------------------------------------------

if __name__ == "__main__":
    ddir = "/path/to/outputs"
    _, _, train_loader, _, = get_all_datasets(
        ddir,
        batch_size=8,
        dataset_id=1,
        normalization="clip_db",
        per_user=True,
    )[:4]
    x, y = next(iter(train_loader))
    print("batch_inp :", x.shape, " batch_out :", y.shape)
