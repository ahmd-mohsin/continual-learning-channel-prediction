

import os
import random
from typing import Tuple, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def _print_range(name: str, arr: np.ndarray) -> None:
    print(f"{name}: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

"""
dataloader.py
Refactored with automatic power-of-10 scaling so that
the largest magnitude is brought into [1, 10).

author: <you>
"""

import os
import random
from typing import Tuple, Dict, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def _print_range(tag: str, arr: np.ndarray):
    print(f"{tag}: min={arr.min():.4e}  max={arr.max():.4e}  mean={arr.mean():.4e}")


class ChannelSequenceDataset(Dataset):
    """
    Sample returned by __getitem__ :
        inp : (Rx, Sub, Tx, seq_len)
        out : (Rx, Sub, Tx)
    """

    AVAILABLE_NORMALISERS = {
        "min_max", "z_score", "log_min_max",
        "robust",  "global_max", "sqrt_min_max", "clip_db"
    }

    def __init__(
        self,
        file_prefix: str,               # path without extension
        file_extension: str,            # 'mat' | 'npy'
        device: torch.device,
        seq_len: int          = 64,
        normalization: str    = "min_max",
        per_user: bool        = False,  # True ⇒ normalise each user separately
        clip_db_floor: float  = -100.0, # only for 'clip_db'
        clip_db_ceil: float   = -30.0,  # only for 'clip_db'
        eps: float            = 1e-12,  # for log/denoms
    ):
        assert normalization in self.AVAILABLE_NORMALISERS, \
            f"unknown normalisation '{normalization}'"

        self.device        = device
        self.normalization = normalization
        self.per_user      = per_user
        self.seq_len       = seq_len
        self.clip_db_floor = clip_db_floor
        self.clip_db_ceil  = clip_db_ceil
        self.eps           = eps

        # ---------------- load raw complex cube ----------------
        file_path = file_prefix + file_extension
        if file_extension == "npy":
            raw = np.load(file_path, mmap_mode="r")           # (U,R,S,Tx,T)
        elif file_extension == "mat":
            with h5py.File(file_path, "r") as f:
                if "channel_matrix" in f:                     # preferred layout
                    grp = f["channel_matrix"]
                    real = np.array(grp["real"])
                    imag = np.array(grp["imag"])
                else:                                         # fallback
                    real = np.array(f["real"])
                    imag = np.array(f["imag"])
            raw = real + 1j * imag
        else:
            raise ValueError("file_extension must be 'mat' or 'npy'")

        # ---------------- shapes & raw stats ------------------
        self.num_users, self.n_rx, self.n_sub, self.n_tx, self.time_len = raw.shape
        print("Loaded raw cube :", raw.shape)

        mag = np.abs(raw).astype(np.float32)
        _print_range("Raw magnitude", mag)

        # ---------------- auto-scale --------------------------
        mag_scaled = self._auto_scale(mag)          # now max ≈ [1,10)
        _print_range("After auto-scale", mag_scaled)

        # ---------------- normalise ---------------------------
        self.data = (
            self._scale_per_user(mag_scaled)
            if self.per_user else
            self._scale_global(mag_scaled)
        )
        _print_range("After normalisation", self.data)

        # stats for inverse-transform or debugging
        self.samples_per_user = self.time_len - (self.seq_len + 1)
    

    def _auto_scale(self, mag: np.ndarray) -> np.ndarray:
        """Bring max magnitude into [1,10) via power-of-10 gain."""
        if self.per_user:
            out            = np.empty_like(mag)
            self.gains: List[float] = []
            for u in range(self.num_users):
                g = self._calc_gain(mag[u].max())
                self.gains.append(g)
                out[u] = mag[u] * g
            print(f"Per-user auto gain : "
                  f"{min(self.gains):.1e} – {max(self.gains):.1e}")
            return out
        else:
            g              = self._calc_gain(mag.max())
            self.gain      = g
            print(f"Global auto gain  : {g:.1e}")
            return mag * g

    @staticmethod
    def _calc_gain(max_val: float) -> float:
        if max_val <= 0:
            return 1.0                             # all zeros → leave as is
        power = np.ceil(-np.log10(max_val))        # e.g. 7.9e-6 → +6 ⇒ 10^6
        return 10.0 ** power                       # ensures new max ∈ [1,10)

    def _scale_per_user(self, arr: np.ndarray) -> np.ndarray:
        out = np.empty_like(arr, dtype=np.float32)
        for u in range(self.num_users):
            out[u] = self._apply_scaler(arr[u])
        return out

    def _scale_global(self, arr: np.ndarray) -> np.ndarray:
        return self._apply_scaler(arr)

    # ---- concrete scalers -----------------------------------

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

    def _min_max(self, x):
        a, b = x.min(), x.max()
        if b == a:
            return np.zeros_like(x)
        return (x - a) / (b - a)

    def _z_score(self, x):
        mu, sigma = x.mean(), x.std() + self.eps
        return (x - mu) / sigma

    def _log_min_max(self, x):
        logx = np.log10(x + self.eps)
        return self._min_max(logx)

    def _robust(self, x):
        q25, q75 = np.percentile(x, [25, 75])
        iqr      = max(q75 - q25, self.eps)
        return np.clip((x - q25) / iqr, 0, 1)

    def _clip_db(self, x):
        y = 20.0 * np.log10(x + self.eps)
        y = np.clip(y, self.clip_db_floor, self.clip_db_ceil)
        return (y - self.clip_db_floor) / (self.clip_db_ceil - self.clip_db_floor)

    def __len__(self) -> int:
        return self.num_users * self.samples_per_user

    def __getitem__(self, idx):
        user   = idx // self.samples_per_user
        t0     = idx %  self.samples_per_user

        mag_seq = self.data[user, :, :, :, t0 : t0 + self.seq_len]      # (R,S,T,L)
        mag_tgt = self.data[user, :, :, :, t0 + self.seq_len]           # (R,S,T)

        mask_seq = (mag_seq > 0).astype(np.float32)
        mask_tgt = (mag_tgt > 0).astype(np.float32)

        # stack → channel dimension: 0=magnitude, 1=mask
        inp  = np.stack([mag_seq, mask_seq], axis=0)                    # (2,R,S,T,L)
        out  = np.stack([mag_tgt, mask_tgt], axis=0)                    # (2,R,S,T)

        return _to_tensor(inp,  self.device), _to_tensor(out, self.device)

def split_dataset(dataset: Dataset, split_ratio: float = 0.8):
    n = len(dataset)
    n_train = int(n * split_ratio)
    n_test  = n - n_train
    return random_split(dataset, [n_train, n_test])

def _load_one(path_prefix: str, batch_size: int, device: torch.device, **ds_kw):
    ds  = ChannelSequenceDataset(path_prefix, "mat", device, **ds_kw)
    trn, tst = split_dataset(ds, 0.8)
    return (
        trn, tst,
        DataLoader(trn, batch_size, shuffle=True,  drop_last=True),
        DataLoader(tst, batch_size, shuffle=False, drop_last=False),
    )

def get_all_datasets(
    data_dir: str,
    batch_size: int = 16,
    dataset_id      = 1,          # 1 | 2 | 3 | 'all'
    **ds_kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names  = {
        1: "umi_fixed_compact_8Tx_2Rx.",
        2: "umi_fixed_dense_8Tx_2Rx.",
        3: "umi_fixed_standard_8Tx_2Rx.",
    }

    def maybe(i):
        if dataset_id in (i, "all"):
            p = os.path.join(data_dir, names[i])
            print(f"→ Loading Dataset {i}: {p}mat")
            return _load_one(p, batch_size, device, **ds_kwargs)
        return (None,) * 4

    return (*maybe(1), *maybe(2), *maybe(3))
