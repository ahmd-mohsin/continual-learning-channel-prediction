import h5py
import logging
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# ——— logging setup —————————————————————————————————————————————————————————————————————
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def _to_tensor(
    array: np.ndarray,
    device: torch.device
) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _print_range(
    name: str,
    array: np.ndarray
) -> None:
    logger.info(
        f"{name}: min={array.min():.4e}  max={array.max():.4e}  mean={array.mean():.4e}"
    )


class ChannelSequenceDataset(Dataset):
    """
    Provides sequences of magnitude+mask pairs from a complex
    channel matrix cube stored in .mat or .npy format.

    __getitem__ returns:
        inp : (2, Rx, Sub, Tx, seq_len)
        out : (2, Rx, Sub, Tx)
    """

    AVAILABLE_NORMALISERS = {
        "min_max", "z_score", "log_min_max",
        "robust", "global_max", "sqrt_min_max", "clip_db"
    }

    def __init__(
        self,
        file_prefix: str,
        file_extension: str,            # 'mat' or 'npy'
        device: torch.device,
        seq_len: int = 64,
        normalization: str = "min_max",
        per_user: bool = False,
        clip_db_floor: float = -100.0,
        clip_db_ceil: float = -30.0,
        eps: float = 1e-12,
    ):
        if normalization not in self.AVAILABLE_NORMALISERS:
            raise ValueError(
                f"[{self.__class__.__name__}] unknown normalization '{normalization}'"
            )

        self.device = device
        self.normalization = normalization
        self.per_user = per_user
        self.seq_len = seq_len
        self.clip_db_floor = clip_db_floor
        self.clip_db_ceil = clip_db_ceil
        self.eps = eps

        # — load raw complex cube —
        path = file_prefix + file_extension
        raw = self._load_raw_cube(path, file_extension)

        self.num_users, self.n_rx, self.n_sub, self.n_tx, self.time_len = raw.shape
        logger.info(f"[{self.__class__.__name__}] Loaded raw cube: {raw.shape}")

        mag = np.abs(raw).astype(np.float32)
        _print_range("Raw magnitude", mag)

        # — auto‐scale →
        mag = self._auto_scale(mag)
        _print_range("After auto-scale", mag)

        # — normalise →
        self.data = (
            self._scale_per_user(mag) if self.per_user
            else self._scale_global(mag)
        )
        _print_range("After normalisation", self.data)

        # for __len__ / indexing
        self.samples_per_user = self.time_len - (self.seq_len + 1)

    def _load_raw_cube(
        self,
        path: str,
        ext: str
    ) -> np.ndarray:
        if ext == "npy":
            return np.load(path, mmap_mode="r")
        if ext == "mat":
            with h5py.File(path, "r") as f:
                if "channel_matrix" in f:
                    grp = f["channel_matrix"]
                    real = np.array(grp["real"])
                    imag = np.array(grp["imag"])
                else:
                    real = np.array(f["real"])
                    imag = np.array(f["imag"])
            return real + 1j * imag
        raise ValueError(
            f"[{self.__class__.__name__}] unsupported extension '{ext}'"
        )

    def _auto_scale(self, mag: np.ndarray) -> np.ndarray:
        """Scale so that max magnitude ∈ [1,10)."""
        if self.per_user:
            out = np.empty_like(mag)
            self.gains: List[float] = []
            for u in range(self.num_users):
                g = self._calc_gain(mag[u].max())
                self.gains.append(g)
                out[u] = mag[u] * g
            logger.info(
                f"[{self.__class__.__name__}] Per-user auto gain: "
                f"{min(self.gains):.1e} – {max(self.gains):.1e}"
            )
            return out
        g = self._calc_gain(mag.max())
        self.gain = g
        logger.info(f"[{self.__class__.__name__}] Global auto gain: {g:.1e}")
        return mag * g

    @staticmethod
    def _calc_gain(max_val: float) -> float:
        if max_val <= 0:
            return 1.0
        power = np.ceil(-np.log10(max_val))
        return 10.0 ** power

    def _scale_per_user(self, arr: np.ndarray) -> np.ndarray:
        out = np.empty_like(arr, dtype=np.float32)
        for u in range(self.num_users):
            out[u] = self._apply_scaler(arr[u])
        return out

    def _scale_global(self, arr: np.ndarray) -> np.ndarray:
        return self._apply_scaler(arr)

    def _apply_scaler(self, x: np.ndarray) -> np.ndarray:
        norm = self.normalization
        if norm == "min_max":
            return self._min_max(x)
        if norm == "z_score":
            return self._z_score(x)
        if norm == "log_min_max":
            return self._log_min_max(x)
        if norm == "robust":
            return self._robust(x)
        if norm == "global_max":
            return x / (x.max() + self.eps)
        if norm == "sqrt_min_max":
            return self._min_max(np.sqrt(x))
        if norm == "clip_db":
            return self._clip_db(x)
        # unreachable
        raise RuntimeError(f"[{self.__class__.__name__}] invalid normalisation")

    def _min_max(self, x: np.ndarray) -> np.ndarray:
        a, b = x.min(), x.max()
        return np.zeros_like(x) if b == a else (x - a) / (b - a)

    def _z_score(self, x: np.ndarray) -> np.ndarray:
        mu, sigma = x.mean(), x.std() + self.eps
        return (x - mu) / sigma

    def _log_min_max(self, x: np.ndarray) -> np.ndarray:
        return self._min_max(np.log10(x + self.eps))

    def _robust(self, x: np.ndarray) -> np.ndarray:
        q25, q75 = np.percentile(x, [25, 75])
        iqr = max(q75 - q25, self.eps)
        return np.clip((x - q25) / iqr, 0.0, 1.0)

    def _clip_db(self, x: np.ndarray) -> np.ndarray:
        y = 20.0 * np.log10(x + self.eps)
        y = np.clip(y, self.clip_db_floor, self.clip_db_ceil)
        return (y - self.clip_db_floor) / (self.clip_db_ceil - self.clip_db_floor)

    def __len__(self) -> int:
        return self.num_users * self.samples_per_user

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user = idx // self.samples_per_user
        t0 = idx % self.samples_per_user

        seq = self.data[user, ..., t0 : t0 + self.seq_len]
        tgt = self.data[user, ..., t0 + self.seq_len]

        mask_seq = (seq > 0).astype(np.float32)
        mask_tgt = (tgt > 0).astype(np.float32)

        inp = np.stack([seq, mask_seq], axis=0)
        out = np.stack([tgt, mask_tgt], axis=0)

        return (
            _to_tensor(inp, self.device),
            _to_tensor(out, self.device),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"users={self.num_users}, seq_len={self.seq_len}, "
            f"norm='{self.normalization}', per_user={self.per_user})"
        )

    __str__ = __repr__


def split_dataset(
    dataset: Dataset,
    split_ratio: float = 0.8
) -> Tuple[Dataset, Dataset]:
    n = len(dataset)
    n_train = int(n * split_ratio)
    return random_split(dataset, [n_train, n - n_train])


def _load_one(
    path_prefix: str,
    batch_size: int,
    device: torch.device,
    **ds_kwargs
) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    ds = ChannelSequenceDataset(path_prefix, "mat", device, **ds_kwargs)
    trn, tst = split_dataset(ds, 0.8)
    return (
        trn, tst,
        DataLoader(trn, batch_size, shuffle=True, drop_last=True),
        DataLoader(tst, batch_size, shuffle=False, drop_last=False),
    )


def get_all_datasets(
    data_dir: str,
    batch_size: int = 16,
    dataset_id: Union[int, str] = 1,  # 1|2|3|'all'
    **ds_kwargs
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[DataLoader], Optional[DataLoader], ...]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names: Dict[int, str] = {
        1: "umi_fixed_compact_8Tx_2Rx.",
        2: "umi_fixed_dense_8Tx_2Rx.",
        3: "umi_fixed_standard_8Tx_2Rx.",
    }

    def maybe(i: int):
        if dataset_id in (i, "all"):
            prefix = os.path.join(data_dir, names[i])
            logger.info(f"→ Loading Dataset {i}: {prefix}mat")
            return _load_one(prefix, batch_size, device, **ds_kwargs)
        return (None, None, None, None)

    return (*maybe(1), *maybe(2), *maybe(3))
