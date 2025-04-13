import torch
import torch.nn as nn
import math
import torch.nn.functional as F

###############################################################################
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

import torch
import torch.nn as nn
from nni.nas.nn.pytorch.layers import Linear  # NAS‑aware Linear

# ─── Embedder Config ─────────────────────────────────────────────────────────
@dataclass
class EmbedderConfig:
    input_dims: int = 3
    include_input: bool = True
    max_freq_log2: int = 9
    num_freqs: int = 10
    log_sampling: bool = True
    periodic_fns: List[Callable] = None

    def __post_init__(self):
        if self.periodic_fns is None:
            self.periodic_fns = [torch.sin, torch.cos]

# ─── Embedder ─────────────────────────────────────────────────────────────────
class Embedder(nn.Module):
    def __init__(self, config: EmbedderConfig):
        super().__init__()
        self.config = config
        self.embedding_fns: List[Callable] = []
        self.output_dims = 0
        self._create_embedding_fns()

    def _create_embedding_fns(self):
        d = self.config.input_dims

        if self.config.include_input:
            self.embedding_fns.append(lambda x: x)
            self.output_dims += d

        if self.config.log_sampling:
            freq_bands = 2.0 ** torch.linspace(
                0.0, self.config.max_freq_log2, steps=self.config.num_freqs
            )
        else:
            freq_bands = torch.linspace(
                1.0, 2.0 ** self.config.max_freq_log2, steps=self.config.num_freqs
            )

        for freq in freq_bands:
            for p_fn in self.config.periodic_fns:
                # bind freq & p_fn into the lambda
                self.embedding_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq)
                )
                self.output_dims += d

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # concatenate all fns along the last dim
        return torch.cat([fn(inputs) for fn in self.embedding_fns], dim=-1)

# ─── get_embedder helper ──────────────────────────────────────────────────────
def get_embedder(
    multires: int,
    input_dims: int = 3,
    include_input: bool = True,
) -> Tuple[Union[Embedder, nn.Identity], int]:
    if multires == -1:
        return nn.Identity(), input_dims

    cfg = EmbedderConfig(
        input_dims=input_dims,
        include_input=include_input,
        max_freq_log2=multires - 1,
        num_freqs=multires,
    )
    emb = Embedder(cfg)
    return emb, emb.output_dims


