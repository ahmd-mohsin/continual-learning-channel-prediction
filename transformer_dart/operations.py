# import torch
# import torch.nn as nn

# OPS = {
#   'none' : lambda C, stride, affine: Zero(stride),
#   'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
#   'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
#   'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
#   'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
#   'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
#   'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
#   'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
#   'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
#   'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
#     nn.ReLU(inplace=False),
#     nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
#     nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
#     nn.BatchNorm2d(C, affine=affine)
#     ),
# }

# class ReLUConvBN(nn.Module):

#   def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#     super(ReLUConvBN, self).__init__()
#     self.op = nn.Sequential(
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
#       nn.BatchNorm2d(C_out, affine=affine)
#     )

#   def forward(self, x):
#     return self.op(x)

# class DilConv(nn.Module):
    
#   def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
#     super(DilConv, self).__init__()
#     self.op = nn.Sequential(
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
#       nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#       nn.BatchNorm2d(C_out, affine=affine),
#       )

#   def forward(self, x):
#     return self.op(x)


# class SepConv(nn.Module):
    
#   def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
#     super(SepConv, self).__init__()
#     self.op = nn.Sequential(
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
#       nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#       nn.BatchNorm2d(C_in, affine=affine),
#       nn.ReLU(inplace=False),
#       nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
#       nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#       nn.BatchNorm2d(C_out, affine=affine),
#       )

#   def forward(self, x):
#     return self.op(x)


# class Identity(nn.Module):

#   def __init__(self):
#     super(Identity, self).__init__()

#   def forward(self, x):
#     return x


# class Zero(nn.Module):

#   def __init__(self, stride):
#     super(Zero, self).__init__()
#     self.stride = stride

#   def forward(self, x):
#     if self.stride == 1:
#       return x.mul(0.)
#     return x[:,:,::self.stride,::self.stride].mul(0.)


# class FactorizedReduce(nn.Module):

#   def __init__(self, C_in, C_out, affine=True):
#     super(FactorizedReduce, self).__init__()
#     assert C_out % 2 == 0
#     self.relu = nn.ReLU(inplace=False)
#     self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#     self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
#     self.bn = nn.BatchNorm2d(C_out, affine=affine)

#   def forward(self, x):
#     x = self.relu(x)
#     out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
#     out = self.bn(out)
#     return out



import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Candidate Operations for Transformer Cells
###############################################################################

class NoneOp(nn.Module):
    """
    Represents a null operation. Returns a zero tensor having the same shape as input.
    """
    def __init__(self, d_model):
        super(NoneOp, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        return x.mul(0.)

class SelfAttentionOp(nn.Module):
    """
    A multi-head self-attention operation.
    Uses PyTorch's nn.MultiheadAttention with batch_first=True.
    """
    def __init__(self, d_model, n_heads=4):
        super(SelfAttentionOp, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        # Self-attention: use the input as query, key, and value.
        out, _ = self.self_attn(x, x, x)
        return out

class FFNOp(nn.Module):
    """
    A simple feed-forward network (FFN).
    Implements two linear layers with a ReLU in between.
    """
    def __init__(self, d_model, expansion=4):
        super(FFNOp, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.fc2 = nn.Linear(d_model * expansion, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class Conv1dOp(nn.Module):
    """
    A 1D convolution operation applied along the time (sequence) dimension.
    The input shape is permuted before and after the convolution.
    """
    def __init__(self, d_model):
        super(Conv1dOp, self).__init__()
        # Convolution parameters ensure the sequence length remains unchanged.
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Input: (batch, seq_len, d_model) -> Permute to (batch, d_model, seq_len)
        x_perm = x.permute(0, 2, 1)
        out = self.conv(x_perm)
        out = self.activation(out)
        # Permute back to (batch, seq_len, d_model)
        return out.permute(0, 2, 1)

# Identity mapping: simply pass through the input.
Identity = nn.Identity

###############################################################################
# OPS Dictionary
###############################################################################
# This dictionary maps primitive string names (as used in your genotype and 
# mixed operation implementations) to a lambda that instantiates the 
# corresponding module given the model dimension.
OPS = {
    "none": lambda d_model: NoneOp(d_model),
    "skip_connect": lambda d_model: Identity(),
    "self_attn": lambda d_model: SelfAttentionOp(d_model),
    "ffn": lambda d_model: FFNOp(d_model),
    "conv1d": lambda d_model: Conv1dOp(d_model)
}
