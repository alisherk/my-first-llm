"""
The term "multi-head attention" refers to a mechanism in transformer models where multiple attention heads operate in parallel, allowing the model to jointly attend to information from different representation subspaces at different positions.

In this context, a sinlge causal attention we developed in causal_attention.py can be considered as one attention head

From one attention head, we can extend the mechanism to multiple heads, each with its own set of query, key, and value projections. The outputs of all heads are then concatenated and linearly transformed to produce the final output.
"""
# First, we will build a multi-head attention by stacking multiple causal attention heads in parallel.
# we will stack 2 attention heads in parallel as shown in multi_head_attention.png

import torch
from torch import nn


class CausalAttention(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout_rate: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length, dtype=torch.bool)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_token = x.shape[1]
        if num_token > self.mask.shape[0]:
            raise ValueError(
                f"Sequence length {num_token} exceeds context length {self.mask.shape[0]}"
            )

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(-2, -1)

        attn_scores = attn_scores.masked_fill(
            ~self.mask[None, :num_token, :num_token], -torch.inf
        )
        scale = keys.shape[-1] ** 0.5
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ values


class MultiHeadAttentionWrapper(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()

        self.heads = torch.nn.ModuleList(
            [
                CausalAttention(
                    d_in, d_out, context_length, dropout_rate=dropout, qkv_bias=qkv_bias
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],  # step     (x^6)
    ]
)


# we we use num_heads = 2, the output dimension will be 2 * d_out = 4
torch.manual_seed(0)  # for reproducibility, optional
batch = torch.stack([inputs, inputs])


context_length = batch.shape[1]
d_in, d_out = 3, 2
num_heads = 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, dropout=0.1, num_heads=num_heads
)
context_vecs = mha(batch)
"""
context_vecs output: torch.Size([2, 6, 4])
tensor([[[-0.5626,  0.3908, -0.3945, -0.7289],
         [-0.7225,  0.4395, -0.1707, -0.8348],
         [-0.7751,  0.4515, -0.0948, -0.8670],
         [-0.6988,  0.4086, -0.0330, -0.7794],
         [-0.6812,  0.3533, -0.0463, -0.6941],
         [-0.4719,  0.2273, -0.0044, -0.7024]],

        [[-0.5626,  0.3908, -0.3945, -0.7289],
         [-0.7225,  0.4395, -0.1707, -0.8348],
         [-0.7751,  0.4515, -0.0948, -0.8670],
         [-0.6988,  0.4086, -0.0330, -0.7794],
         [-0.6812,  0.3533, -0.0595, -0.5166],
         [-0.6522,  0.3622, -0.0146, -0.5509]]], grad_fn=<CatBackward0>)
"""


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0.1,
        qkv_bias=False,
    ):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # Reduce the projection dim to match desired output dim

        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        """
        supposed d_in is 6 and d_out = 2 
        and token has 6 inputs [1, 2, 3, 4, 5, 6]

        W_key = torch.nn.Linear(6, 2, bias=False)
        W_key.weight:
            [1, 1, 1, 1, 1, 1] → head 0 key
            [2, 2, 2, 2, 2, 2] → head 1 key

        output 0 = 1×1 + 2×1 + 3×1 + 4×1 + 5×1 + 6×1 = 21
        output 1 = 1×2 + 2×2 + 3×2 + 4×2 + 5×2 + 6×2 = 42

        key = [21, 42]

        Both outputs use all 6 input numbers, but each uses its own row of 6 weights. 
        If there are 2 attention heads, head 0 receives [21] and head 1 receives [42].

        So each weight in the Linear object is applied to each token independently, producing the corresponding key for each head.
        """

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor):
        batch_size, num_tokens, _ = x.shape

        if num_tokens > self.mask.size(0):
            raise ValueError(
                f"Number of tokens ({num_tokens}) exceeds context length ({self.mask.size(0)})"
            )
       # Shape: (b, num_tokens, d_out) -> computes the keys for all heads at once, each of the 6 toens get 4 numbers [k1, k2, k3, k4]
       # assume num of heads = 2, head 0 numbers [k1, k2] and head 1 numbers [k3, k4] for each token
        keys = self.W_key(x) 
        """
        inputs = torch.tensor(
            [
                [0.43, 0.15, 0.89],  # Your     (x^1)
                [0.55, 0.87, 0.66],  # journey  (x^2)
                [0.57, 0.85, 0.64],  # starts   (x^3)
                [0.22, 0.58, 0.33],  # with     (x^4)
                [0.77, 0.25, 0.10],  # one      (x^5)
                [0.05, 0.80, 0.55],  # step     (x^6)
            ]
        )

        keys tensor([[
         [ 0.3830, -0.0107],
         [ 0.4763,  0.2750],
         [ 0.4537,  0.2543],
         [ 0.3028,  0.2335],
         [-0.0808, -0.1899],
         [ 0.5371,  0.4397]],

        [[ 0.3830, -0.0107],
         [ 0.4763,  0.2750],
         [ 0.4537,  0.2543],
         [ 0.3028,  0.2335],
         [-0.0808, -0.1899],
         [ 0.5371,  0.4397]]],
        """ 

        queries = self.W_query(x)
        values = self.W_value(x)

        # splits 4 numbers list [k1, k2, k3, k4] into 2 heads, each with 2 numbers [[k1, k2], [k3, k4]]) without moving any data in memory.
        # view() just updates the tensor's shape metadata so PyTorch knows to treat the first two numbers as Head 0 ([k1, k2]) and the next two as Head 1 ([k3, k4]).
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        """
        tensor([Start of Dimension 0: batch_size = 2 (holds 2 batch items)
                [ Batch 0 | Start of Dimension 1: num_tokens = 6 (holds 6 tokens)

                    [ Token 0 | Start of Dimension 2: num_heads = 2
                        [ 0.3830], Head 0 | Dimension 3: head_dim = 1 (1 number per head)
                        [-0.0107]  Head 1 | Dimension 3: head_dim = 1
                    ],
                    [ Token 1
                        [ 0.4763], Head 0
                        [ 0.2750]  Head 1
                    ],
                    [ Token 2
                        [ 0.4537], Head 0
                        [ 0.2543]  Head 1
                    ],
                    [ Token 3
                        [ 0.3028], Head 0
                        [ 0.2335]  Head 1
                    ],
                    [ Token 4
                        [-0.0808], Head 0
                        [-0.1899]  Head 1
                    ],
                    [ Token 5
                        [ 0.5371], Head 0
                        [ 0.4397]  Head 1
                    ]
                ], End of Batch Item 0

                [ Batch Item 1 | Start of Dimension 1: num_tokens = 6 (identical copy in this example)
                    [ Token 0 | Start of Dimension 2: num_heads = 2
                        [ 0.3830], Head 0 | Dimension 3: head_dim = 1
                        [-0.0107]  Head 1 | Dimension 3: head_dim = 1
                    ],
                    [ Token 1
                        [ 0.4763], Head 0
                        [ 0.2750]  Head 1
                    ],
                    [ Token 2
                        [ 0.4537], Head 0
                        [ 0.2543]  Head 1
                    ],
                    [ Token 3
                        [ 0.3028], Head 0
                        [ 0.2335]  Head 1
                    ],
                    [ Token 4
                        [-0.0808], Head 0
                        [-0.1899]  Head 1
                    ],
                    [ Token 5
                        [ 0.5371], Head 0
                        [ 0.4397]  Head 1
                    ]
                ], End of Batch Item 1
            ], grad_fn=<ViewBackward0>) End of Dimension 0

        """

        """
            keys:    [k0, k1, k2, k3] → [[k0, k1], [k2, k3]]
            queries: [q0, q1, q2, q3] → [[q0, q1], [q2, q3]]
            values:  [v0, v1, v2, v3] → [[v0, v1], [v2, v3]]
                                          head 0    head 1
        """

        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # the problem here is that our data grouped by [batch, num_tokens, num_heads, head_dim]
        # but for attention computation we need it grouped by [batch, num_heads, num_tokens, head_dim]
        # keys.transpose(1, 2) changes the shape to what we want
        keys = keys.transpose(1, 2)

        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # the problem here is that queries shape is [2, 2, 6, 2]
        # our keys shape is [2, 2, 6, 2]
        # ((6, 2) * (6, 2)) would not work for matrix multiplication
        # we need to transpose the last two dimensions of keys to get (6, 2) * (2, 6) -> (6, 6) for attention scores
        # so keys.transpose(2, 3) will turn into [2, 2, 2, 6] for this example so (6, 2) * (2, 6) works for matrix multiplication
        # shape of queries and keys for this example: [batch_size, num_heads, num_tokens, head_dim]
        # so if we permform matrix multiplication like this, there will be an error due to mismatched dimensions
        attn_scores = queries @ keys.transpose(2, 3)

        """
        queries:                 [batch, 2, 6, 2]
        keys.transpose(2, 3):    [batch, 2, 2, 6]
                                ----------------
        attn_scores:             [batch, 2, 6, 6]

        head 0 queries @ head 0 keys → head 0 scores [6, 6]
        head 1 queries @ head 1 keys → head 1 scores [6, 6]
        """ 

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        
        return self.out_proj(context_vec)  # optional projection

torch.manual_seed(42)
batch_size, context_length, d_in = batch.shape # 2, 6, 3
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, num_heads=2)
context_vectors = mha(batch)

"""
torch.Size([2, 6, 2])
tensor([[[-0.4281,  0.2988],
         [-0.4474,  0.2959],
         [-0.3399,  0.3295],
         [-0.3808,  0.2923],
         [-0.3633,  0.3207],
         [-0.3408,  0.3381]],

        [[-0.4281,  0.2988],
         [-0.3270,  0.3549],
         [-0.4001,  0.2701],
         [-0.3808,  0.2923],
         [-0.3864,  0.2979],
         [-0.3209,  0.3578]]], grad_fn=<UnsafeViewBackward0>)
"""