# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from operations import *
# from torch.autograd import Variable
# from genotypes import PRIMITIVES
# from genotypes import Genotype


# class MixedOp(nn.Module):

#   def __init__(self, C, stride):
#     super(MixedOp, self).__init__()
#     self._ops = nn.ModuleList()
#     for primitive in PRIMITIVES:
#       op = OPS[primitive](C, stride, False)
#       if 'pool' in primitive:
#         op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
#       self._ops.append(op)

#   def forward(self, x, weights):
#     return sum(w * op(x) for w, op in zip(weights, self._ops))


# class Cell(nn.Module):

#   def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
#     super(Cell, self).__init__()
#     self.reduction = reduction

#     if reduction_prev:
#       self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
#     else:
#       self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
#     self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
#     self._steps = steps
#     self._multiplier = multiplier

#     self._ops = nn.ModuleList()
#     self._bns = nn.ModuleList()
#     for i in range(self._steps):
#       for j in range(2+i):
#         stride = 2 if reduction and j < 2 else 1
#         op = MixedOp(C, stride)
#         self._ops.append(op)

#   def forward(self, s0, s1, weights):
#     s0 = self.preprocess0(s0)
#     s1 = self.preprocess1(s1)

#     states = [s0, s1]
#     offset = 0
#     for i in range(self._steps):
#       s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
#       offset += len(states)
#       states.append(s)

#     return torch.cat(states[-self._multiplier:], dim=1)


# class Network(nn.Module):

#   def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
#     super(Network, self).__init__()
#     self._C = C
#     self._num_classes = num_classes
#     self._layers = layers
#     self._criterion = criterion
#     self._steps = steps
#     self._multiplier = multiplier

#     C_curr = stem_multiplier*C
#     self.stem = nn.Sequential(
#       nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
#       nn.BatchNorm2d(C_curr)
#     )
 
#     C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
#     self.cells = nn.ModuleList()
#     reduction_prev = False
#     for i in range(layers):
#       if i in [layers//3, 2*layers//3]:
#         C_curr *= 2
#         reduction = True
#       else:
#         reduction = False
#       cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
#       reduction_prev = reduction
#       self.cells += [cell]
#       C_prev_prev, C_prev = C_prev, multiplier*C_curr

#     self.global_pooling = nn.AdaptiveAvgPool2d(1)
#     self.classifier = nn.Linear(C_prev, num_classes)

#     self._initialize_alphas()

#   def new(self):
#     model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
#     for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
#         x.data.copy_(y.data)
#     return model_new

#   def forward(self, input):
#     s0 = s1 = self.stem(input)
#     for i, cell in enumerate(self.cells):
#       if cell.reduction:
#         weights = F.softmax(self.alphas_reduce, dim=-1)
#       else:
#         weights = F.softmax(self.alphas_normal, dim=-1)
#       s0, s1 = s1, cell(s0, s1, weights)
#     out = self.global_pooling(s1)
#     logits = self.classifier(out.view(out.size(0),-1))
#     return logits

#   def _loss(self, input, target):
#     logits = self(input)
#     return self._criterion(logits, target) 

#   def _initialize_alphas(self):
#     k = sum(1 for i in range(self._steps) for n in range(2+i))
#     num_ops = len(PRIMITIVES)

#     self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
#     self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
#     self._arch_parameters = [
#       self.alphas_normal,
#       self.alphas_reduce,
#     ]

#   def arch_parameters(self):
#     return self._arch_parameters

#   def genotype(self):

#     def _parse(weights):
#       gene = []
#       n = 2
#       start = 0
#       for i in range(self._steps):
#         end = start + n
#         W = weights[start:end].copy()
#         edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
#         for j in edges:
#           k_best = None
#           for k in range(len(W[j])):
#             if k != PRIMITIVES.index('none'):
#               if k_best is None or W[j][k] > W[j][k_best]:
#                 k_best = k
#           gene.append((PRIMITIVES[k_best], j))
#         start = end
#         n += 1
#       return gene

#     gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
#     gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

#     concat = range(2+self._steps-self._multiplier, self._steps+2)
#     genotype = Genotype(
#       normal=gene_normal, normal_concat=concat,
#       reduce=gene_reduce, reduce_concat=concat
#     )
#     return genotype




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Assumes that operations.py defines OPS for transformer operations
from operations import OPS
from genotypes import PRIMITIVES
from genotypes import Genotype
from positional_encoder import get_embedder
###############################################################################
# Transformer Mixed Operation
###############################################################################
class TransformerMixedOp(nn.Module):
    """
    Similar to DARTS’ MixedOp but for transformer operations.
    Each candidate op should operate on an input tensor of shape 
    (batch, seq_len, d_model) and output the same shape.
    """
    def __init__(self, d_model):
        super(TransformerMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            # Each op is assumed to accept (d_model) as its only size parameter.
            op = OPS[primitive](d_model)
            self._ops.append(op)

    def forward(self, x, weights):
        # weights is a vector (or scalar list) applied to each op’s output.
        return sum(w * op(x) for w, op in zip(weights, self._ops))


###############################################################################
# Transformer Cell
###############################################################################
class TransformerCell(nn.Module):
    """
    A Transformer cell analogous to a DARTS cell.
    Receives two states s0 and s1 (each of shape (batch, seq_len, d_model)),
    and builds a directed acyclic graph (DAG) with a set number of steps.
    Each edge in the DAG is a TransformerMixedOp.
    The outputs of the last few nodes (as determined by multiplier)
    are concatenated and projected back to d_model.
    """
    def __init__(self, steps, multiplier, d_model):
        super(TransformerCell, self).__init__()
        self.steps = steps
        self.multiplier = multiplier

        # Build a list of mixed operations – each edge is a candidate op.
        self._ops = nn.ModuleList()
        # In the DARTS cell, the number of incoming edges for node i is (2+i)
        for i in range(self.steps):
            for j in range(2 + i):
                op = TransformerMixedOp(d_model)
                self._ops.append(op)
                
        # After concatenating last `multiplier` states (each of size d_model),
        # project them back to d_model.
        self.proj = nn.Linear(multiplier * d_model, d_model)

    def forward(self, s0, s1, weights):
        # s0 and s1 are assumed to be token sequences: (batch, seq_len, d_model)
        states = [s0, s1]
        offset = 0
        for i in range(self.steps):
            # For node i, combine all previous states using their corresponding ops.
            s = sum(self._ops[offset + j](h, weights[offset + j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        # Concatenate the last 'multiplier' states along the feature (d_model) dimension.
        out = torch.cat(states[-self.multiplier:], dim=-1)
        out = self.proj(out)
        return out


###############################################################################
# Transformer Network (akin to the DARTS Network)
###############################################################################
class TransformerNetwork(nn.Module):
    """
    A Transformer-based network for time-series channel prediction.
    Instead of a convolutional stem, we use a linear input projection
    to map the flattened spatial features (channels, H, W) to d_model.
    
    The network expects an input tensor of shape:
      (batch, out_channels, H, W, seq_len)
    and outputs a prediction reshaped to:
      (batch, out_channels, H, W)
    
    Architecture search is enabled through learnable architecture parameters,
    which are used in each cell (sharing a common set of weights).
    """
    def __init__(self, d_model, num_classes, layers, criterion,
                 steps=4, multiplier=4, input_dims=None, seq_len=16, multires=6):
        super(TransformerNetwork, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes  # For regression (e.g. channel prediction),
                                         # num_classes could be equal to out_channels*H*W.
        self.layers = layers
        self.steps = steps
        self.multiplier = multiplier
        self.seq_len = seq_len

        # input_dims is a tuple: (out_channels, H, W)
        if input_dims is None:
            input_dims = (4, 18, 16)
        self.out_channels, self.H, self.W = input_dims
        self.input_size = self.out_channels * self.H * self.W

        self._criterion = criterion

        # 1) Linear projection from flattened spatial (per time step) to d_model:
        self.input_projection = nn.Linear(self.input_size, d_model)
        
        # 2) Frequency-based positional encoding:
        self.pos_encoder = PositionalEncoding(d_model, multires=multires)
        
        # 3) Stack of TransformerCells:
        self.cells = nn.ModuleList()
        for i in range(layers):
            cell = TransformerCell(steps, multiplier, d_model)
            self.cells.append(cell)
            
        # 4) Final projection: project last token from d_model back to input_size.
        self.fc_out = nn.Linear(d_model, self.input_size)

        self._initialize_alphas()

    def new(self):
        model_new = TransformerNetwork(
            self.d_model, self.num_classes, self.layers, self._criterion,
            steps=self.steps, multiplier=self.multiplier,
            input_dims=(self.out_channels, self.H, self.W),
            seq_len=self.seq_len
        ).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, out_channels, H, W, seq_len)
        Returns:
            Tensor of shape (batch, out_channels, H, W)
        """
        batch_size = x.size(0)
        # Permute and reshape:
        # -> (batch, seq_len, out_channels, H, W)
        # -> then flatten spatial dimensions: (batch, seq_len, input_size)
        # print("=========================")
        # print(x.shape)
        # print("=========================")
        x = x.permute(0, 4, 1, 2,3).reshape(batch_size, self.seq_len, -1)
        
        # Project to d_model:
        s = self.input_projection(x)  # shape: (batch, seq_len, d_model)
        s = self.pos_encoder(s)
        
        # Initialize two states; for the first cell both are the same.
        s0 = s1 = s
        for cell in self.cells:
            # Here we use a single set of learnable architecture weights (alphas_normal)
            weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
            
        # Use the last token from the final cell as the representation.
        out = self.fc_out(s1[:, -1])  # shape: (batch, input_size)
        out = out.view(batch_size, self.out_channels, self.H, self.W)
        out = out.squeeze()
        return out

    def _loss(self, input, target):
        logits = self(input)
        # print("=========================")
        # print(logits.shape)
        # print(target.shape)
        # print("=========================")
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        # Count total number of edges in a cell.
        k = sum(1 for i in range(self.steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        # Similar to the DARTS genotype extraction.
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self.steps):
                end = start + n
                # Get a copy of weights for this node’s incoming edges.
                W = weights[start:end].copy()
                # Choose the best two edges (ignoring the 'none' op).
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x]))
                                        if k != PRIMITIVES.index('none'))
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        concat = list(range(2 + self.steps - self.multiplier, self.steps + 2))
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=[], reduce_concat=concat  # No separate reduction operations in this transformer version
        )
        return genotype


###############################################################################
# Positional Encoding
###############################################################################
class PositionalEncoding(nn.Module):
    """
    Frequency-based positional encoding.
    This wraps a multi-resolution Fourier embedder.
    """
    def __init__(self, d_model, multires=6):
        super(PositionalEncoding, self).__init__()
        # get_embedder should return a callable (the embedder) and its output dimension.
        self.embedder, embed_dim = get_embedder(
            multires,           # number of frequency bands
            input_dims=d_model, # treat each d_model dimension as input
            include_input=True
        )
        self.need_projection = (embed_dim != d_model)
        if self.need_projection:
            self.proj = nn.Linear(embed_dim, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor of shape (batch, seq_len, d_model) with frequency-based embeddings.
        """
        encoded = self.embedder(x)  # (batch, seq_len, embed_dim)
        if self.need_projection:
            encoded = self.proj(encoded)
        return encoded
