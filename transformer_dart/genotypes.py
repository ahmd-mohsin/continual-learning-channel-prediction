# from collections import namedtuple

# Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# PRIMITIVES = [
#     'none',
#     'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect',
#     'sep_conv_3x3',
#     'sep_conv_5x5',
#     'dil_conv_3x3',
#     'dil_conv_5x5'
# ]

# NASNet = Genotype(
#   normal = [
#     ('sep_conv_5x5', 1),
#     ('sep_conv_3x3', 0),
#     ('sep_conv_5x5', 0),
#     ('sep_conv_3x3', 0),
#     ('avg_pool_3x3', 1),
#     ('skip_connect', 0),
#     ('avg_pool_3x3', 0),
#     ('avg_pool_3x3', 0),
#     ('sep_conv_3x3', 1),
#     ('skip_connect', 1),
#   ],
#   normal_concat = [2, 3, 4, 5, 6],
#   reduce = [
#     ('sep_conv_5x5', 1),
#     ('sep_conv_7x7', 0),
#     ('max_pool_3x3', 1),
#     ('sep_conv_7x7', 0),
#     ('avg_pool_3x3', 1),
#     ('sep_conv_5x5', 0),
#     ('skip_connect', 3),
#     ('avg_pool_3x3', 2),
#     ('sep_conv_3x3', 2),
#     ('max_pool_3x3', 1),
#   ],
#   reduce_concat = [4, 5, 6],
# )
    
# AmoebaNet = Genotype(
#   normal = [
#     ('avg_pool_3x3', 0),
#     ('max_pool_3x3', 1),
#     ('sep_conv_3x3', 0),
#     ('sep_conv_5x5', 2),
#     ('sep_conv_3x3', 0),
#     ('avg_pool_3x3', 3),
#     ('sep_conv_3x3', 1),
#     ('skip_connect', 1),
#     ('skip_connect', 0),
#     ('avg_pool_3x3', 1),
#     ],
#   normal_concat = [4, 5, 6],
#   reduce = [
#     ('avg_pool_3x3', 0),
#     ('sep_conv_3x3', 1),
#     ('max_pool_3x3', 0),
#     ('sep_conv_7x7', 2),
#     ('sep_conv_7x7', 0),
#     ('avg_pool_3x3', 1),
#     ('max_pool_3x3', 0),
#     ('max_pool_3x3', 1),
#     ('conv_7x1_1x7', 0),
#     ('sep_conv_3x3', 5),
#   ],
#   reduce_concat = [3, 4, 6]
# )

# DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
# DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

# DARTS = DARTS_V2



from collections import namedtuple

# Define the genotype type with discrete operations and concatenation indices.
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# For the transformer model we use a different set of operations.
# Here the PRIMITIVES are chosen to represent candidate operations for transformer cells.
PRIMITIVES = [
    'none',
    'skip_connect',
    'self_attn',
    'ffn',
    'conv1d'
]

# An example genotype for a Transformer-based architecture.
# In this setting:
#   - 'normal' encodes the chosen operations for each edge in the transformer cell.
#   - 'normal_concat' defines which of the intermediate node outputs are concatenated to form
#      the cell's output.
#
# Since we have a single branch (no separate reduction cell in our transformer design),
# the "reduce" branch is kept identical or empty.
Transformer_V1 = Genotype(
    normal=[
        ('self_attn', 0),
        ('ffn', 1),
        ('conv1d', 0),
        ('skip_connect', 2),
        ('ffn', 1),
        ('self_attn', 0),
        ('conv1d', 2),
        ('skip_connect', 1)
    ],
    normal_concat=[2, 3, 4],
    reduce=[
        ('self_attn', 0),
        ('ffn', 1),
        ('conv1d', 0),
        ('skip_connect', 2),
        ('ffn', 1),
        ('self_attn', 0),
        ('conv1d', 2),
        ('skip_connect', 1)
    ],
    reduce_concat=[2, 3, 4]
)

# For ease of use you can set a default transformer genotype.
Transformer = Transformer_V1
