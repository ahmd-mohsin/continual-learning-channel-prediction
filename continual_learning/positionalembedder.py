# models/embedder.py

from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

import torch
from torch import nn


@dataclass
class EmbedderConfig:
    """Configuration for positional encoding embedder.

    Args:
        input_dims: Dimensionality of input features
        include_input: Whether to include raw input in embedding
        max_freq_log2: Log2 of max frequency
        num_freqs: Number of frequency bands
        log_sampling: Whether to sample frequencies in log space
        periodic_fns: List of periodic functions to use
    """

    input_dims: int = 3
    include_input: bool = True
    max_freq_log2: int = 9  # L-1
    num_freqs: int = 10  # L
    log_sampling: bool = True
    periodic_fns: List[Callable] = None

    def __post_init__(self):
        if self.periodic_fns is None:
            self.periodic_fns = [torch.sin, torch.cos]


class Embedder(nn.Module):
    """Positional encoding embedder module.

    Maps input features to a higher dimensional space using a series of
    sinusoidal functions at different frequencies.

    Args:
        config: Configuration object
    """

    def __init__(self, config: EmbedderConfig):
        super().__init__()
        self.config = config
        self.embedding_fns: List[Callable] = []
        self.output_dims = 0
        self._create_embedding_fn()

    def _create_embedding_fn(self):
        """Setup the embedding functions."""
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
                2.0**0.0, 2.0**self.config.max_freq_log2, steps=self.config.num_freqs
            )

        for freq in freq_bands:
            for p_fn in self.config.periodic_fns:
                self.embedding_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq)
                )
                self.output_dims += d

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input.

        Args:
            inputs: Input tensor to encode

        Returns:
            Encoded tensor with higher dimensionality
        """
        return torch.cat([fn(inputs) for fn in self.embedding_fns], dim=-1)


def get_embedder(
    multires: int,
    input_dims: int = 3,
    include_input: bool = True,
) -> Tuple[Union[Embedder, nn.Identity], int]:
    """Create an embedder module.

    Args:
        multires: Number of frequency bands (L in paper)
        input_dims: Dimensionality of input features
        include_input: Whether to include raw input

    Returns:
        Tuple of (embedder module, output dimensionality)
    """
    if multires == -1:
        return nn.Identity(), input_dims

    config = EmbedderConfig(
        input_dims=input_dims,
        include_input=include_input,
        max_freq_log2=multires - 1,
        num_freqs=multires,
    )

    embedder = Embedder(config)
    return embedder, embedder.output_dims