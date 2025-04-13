import os
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
import nni
import nni.nas.nn.pytorch as nasnn
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment
from nni.nas.nn.pytorch.layers import Linear, Transformer
from dataloader import ChannelSequenceDataset
from dataclasses import dataclass
import torch
# import torch.nn as nn
import math
# import torch.nn.functional as F
from nni.nas.nn.pytorch.layers import Linear, Transformer  # NAS‑aware Transformer
# import logging
# logging.getLogger('websocket').setLevel(logging.INFO)

###############################################################################
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

import torch
import torch.nn as nn
from nni.nas.nn.pytorch.layers import Linear  # NAS‑aware Linear



# def get_embedder(multires, input_dims, include_input):
#     # returns (embed_fn, embed_dim)
#     # e.g. from your positionalembeder.py
#     raise NotImplementedError

def generate_square_subsequent_mask(sz1, sz2):
    # your mask function
    return torch.triu(torch.ones(sz1, sz2) * float('-inf'), diagonal=1)


from nni.nas.nn.pytorch.layers import Linear  # NAS‑aware Linear
class PositionalEncoding(nasnn.ModelSpace):
    """
    Wraps the frequency-based embedder (from positionalembber.py)
    so that it can replace the old sinusoidal PositionalEncoding.
    """

    def __init__(self, d_model, multires=6):
        """
        Args:
            d_model (int): The 'model dimension' that the Transformer uses.
            multires (int): Number of frequency bands (L) for the embedder.
                            Increase/decrease as you like.
        """
        super(PositionalEncoding, self).__init__()

        # Build the embedder configured to take in dimension = d_model
        self.embedder, embed_dim = get_embedder(
            multires,           # e.g. 6 or 10
            input_dims=d_model, # we treat each 'd_model' channel as an input
            include_input=True
        )

        # If embedder's output dimension differs from d_model,
        # define a linear layer to project it back to d_model
        # self.need_projection = (embed_dim != d_model)
        # if self.need_projection:
        #     self.proj = nn.Linear(embed_dim, d_model)
        self.proj = Linear(embed_dim, d_model)
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model),
            with the frequency-based positional encodings.
        """
        # Apply the multi-frequency embedder
        # shape after embedder: (batch, seq_len, embed_dim)
        encoded = self.embedder(x)

        # If embed_dim != c, project back down
        # if self.need_projection:
        encoded = self.proj(encoded)

        return encoded


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
class Embedder(nasnn.ModelSpace):
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




class TransformerModelSpace(nasnn.ModelSpace):
    def __init__(self,
                 out_channels=4, H=18, W=16, seq_len=16):
        super().__init__()
        # hyper‑param choices
        # self.dim_val          = nni.choice('dim_val',          [16, 32])
        self.dim_val          = 128
        self.n_heads         = nni.choice('n_heads',         [2, 4, 6  ])
        self.n_encoder_layers = nni.choice('n_encoder_layers', [1,   2])
        self.n_decoder_layers = nni.choice('n_decoder_layers', [1,   2  ])
        # self.dim_val          = nni.choice('dim_val',          [64, 128, 256])
        # self.n_heads         = nni.choice('n_heads',         [2,   4,   8  ])
        # self.n_encoder_layers = nni.choice('n_encoder_layers', [1,   2,   4,   6])
        # self.n_decoder_layers = nni.choice('n_decoder_layers', [1,   2,   3  ])
        # self.multires        = nni.choice('multires',        [4,   6,   8  ])
        self.multires = 8
        self.out_channels = out_channels
        self.H = H; self.W = W; self.seq_len = seq_len
        self.input_size = out_channels * H * W

        # shared modules
        # self.input_projection = nn.Linear(self.input_size, self.dim_val)
        self.input_projection = Linear(self.input_size, self.dim_val)
        # self.input_projection = nasnn.Linear(self.input_size, self.dim_val)
        self.pos_encoder      = PositionalEncoding(d_model=self.dim_val,
                                                   multires=self.multires)
        self.transformer      = Transformer(
            d_model=self.dim_val,
            nhead=self.n_heads,
            num_encoder_layers=self.n_encoder_layers,
            num_decoder_layers=self.n_decoder_layers,
            batch_first=True
        )
        self.fc_out = Linear(self.dim_val, self.input_size)

    def forward(self, x):
        B = x.size(0)
        # x: (B, out_ch, H, W, seq_len)
        # print("------------------------")
        # print(x.shape)
        x = x.permute(0,4,1,2,3).reshape(B, self.seq_len, -1)
        # print(x.shape)
        # print("------------------------")
        src = self.input_projection(x)
        src = self.pos_encoder(src)
        tgt = torch.zeros(B, 1, self.dim_val, device=x.device)
        tgt = self.pos_encoder(tgt)
        src_mask = generate_square_subsequent_mask(self.seq_len, self.seq_len).to(x.device)
        tgt_mask = generate_square_subsequent_mask(1, 1).to(x.device)
        out = self.transformer(src=src, tgt=tgt,
                               src_mask=src_mask, tgt_mask=tgt_mask)
        out = self.fc_out(out).squeeze(1)
        return out.view(B, self.out_channels, self.H, self.W)
    



# ─── 2) Define your evaluator ────────────────────────────────────────────────
# 

import nni.nas.strategy as strategy

# ─── 3) Launch NAS Experiment ────────────────────────────────────────────────
from nas_utils import evaluate_model
if __name__ == '__main__':
    model_space     = TransformerModelSpace()
    # print("----------------------------------")
    # model_space = nasnn.ModelSpace()
    # print("Model space created")
    # print(model_space)
    evaluator       = FunctionalEvaluator(evaluate_model)
    search_strategy = strategy.Random()  # dedup=False if deduplication is not wanted
    # print("----------------------------------")
    exp = NasExperiment(model_space, evaluator, search_strategy)
    # print("----------------------------------")

    # customize experiment settings
    exp.config.max_trial_number   = 3
    exp.config.trial_concurrency  = 2
    exp.config.trial_gpu_number   = 0
    exp.config.training_service.use_active_gpu = True

    # run on port 8081
    exp.run(8081)
    # print("Experiment started! Web UI at http://localhost:8081")