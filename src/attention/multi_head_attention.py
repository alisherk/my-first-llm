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
        self.head_dim = d_out // num_heads # 1 for d_out=2, num_heads=2

        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        """
        Example: d_in=3, d_out=2, num_heads=2, and bias=False.

        supposed my first 2 tokens are:
        "Your"    [0.43, 0.15, 0.89]
        "journey" [0.55, 0.87, 0.66]

        If the learned weights happened to be:
        W_key.weight (shape [d_out, d_in]):
            [1, 1, 1] → head 0 key
            [2, 2, 2] → head 1 key

       2 rows of weights will be applied to "journey" token embedding as follows:
       "journey" embedding: [0.55, 0.87, 0.66]

            head 0 key: 1*0.55 + 1*0.87 + 1*0.66 = 2.08
            head 1 key: 2*0.55 + 2*0.87 + 2*0.66 = 4.16

        key = [2.08, 4.16]
        If there are 2 attention heads, head 0 receives [2.08] and head 1 receives [4.16].
        
        The 2 set of weights will be applied to each token embedding to produce the corresponding keys for each head.

        We can assume here that we would need number of heads to match the number of rows in the weight matrix for proper splitting of the keys among the heads.
        """

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)  # Mix the concatenated head outputs.
        
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
        
        keys = self.W_key(x) 
        """
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

        # Split keys into heads and arrange them for multi-head attention
        # [batch, tokens, d_out] -> [batch, tokens, heads, head_dim].
        # In the example above, [0.3830, -0.0107] becomes [[0.3830], [-0.0107]]: one number for each head.
        # view() only groups the features; it does not calculate attention.
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        """
        shape.Size([2, 6, 2, 1])
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

        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        """
         Move the heads before tokens so each head can attend independently:
        [batch, tokens, heads, head_dim] -> [batch, heads, tokens, head_dim].

        • Dimension 0: batch
        • Dimension 1: tokens
        • Dimension 2: heads (where index 0 = head 0, and index 1 = head 1)
        • Dimension 3: features

        Before transpose(1, 2), keys are grouped by token:
        [batch, tokens, heads, features]

                            head 0      head 1
                        +---------+---------+
            token 0 ->  | [ 0.3830] | [-0.0107] |
            token 1 ->  | [ 0.4763] | [ 0.2750] |
            token 2 ->  | [ 0.4537] | [ 0.2543] |
            token 3 ->  | [ 0.3028] | [ 0.2335] |
            token 4 ->  | [-0.0808] | [-0.1899] |
            token 5 ->  | [ 0.5371] | [ 0.4397] |
                        +---------+---------+
                                |
                                |   .transpose(1, 2)
                                |   (Swaps tokens & heads)
                                v

            AFTER: [batch=1, heads=2, tokens=6, features=1]
            (Heads are ROWS, Tokens are COLUMNS)

                        token 0     token 1     token 2     token 3     token 4     token 5
                    +-----------+-----------+-----------+-----------+-----------+-----------+
            head 0 -> | [ 0.3830] | [ 0.4763] | [ 0.4537] | [ 0.3028] | [-0.0808] | [ 0.5371] |
            head 1 -> | [-0.0107] | [ 0.2750] | [ 0.2543] | [ 0.2335] | [-0.1899] | [ 0.4397] |
                    +-----------+-----------+-----------+-----------+-----------+-----------+
        """
        keys = keys.transpose(1, 2)

        # Split queries into heads and arrange them for multi-head attention
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        """
        Calculate scores independently for each head. 
        With batch_size=2, num_tokens=6, num_heads=2, and head_dim=1:

        Before transposing:
            queries: [batch=2, heads=2, tokens=6, head_dim=1]
            keys:    [batch=2, heads=2, tokens=6, head_dim=1]

        (2, 2, 6, 1) does not match (2, 2, 6, 1) because the last dimension of queries (head_dim) must match the third dimension of keys (tokens) for matrix multiplication.

        before calling keys.transpose(2, 3), keys are grouped by token:

        keys = torch.tensor([
            [
                # --- Head 0 (shape:) ---
                [
                    [ 0.3830],  # token 0
                    [ 0.4763],  # token 1
                    [ 0.4537],  # token 2
                    [ 0.3028],  # token 3
                    [-0.0808],  # token 4
                    [ 0.5371],  # token 5
                ],
                # --- Head 1 (shape:) ---
                [
                    [-0.0107],  # token 0
                    [ 0.2750],  # token 1
                    [ 0.2543],  # token 2
                    [ 0.2335],  # token 3
                    [-0.1899],  # token 4
                    [ 0.4397],  # token 5
                ],
            ],
            [
                # --- Head 0 (shape:) ---
                [
                    [ 0.1100], [ 0.2200], [ 0.3300], [ 0.4400], [ 0.5500], [ 0.6600]
                ],
                # --- Head 1 (shape:) ---
                [
                    [-0.1100], 
                    [-0.2200], 
                    [-0.3300], 
                    [-0.4400], 
                    [-0.5500], 
                    [-0.6600]
                ],
        ])
        # keys.shape -> torch.Size([2, 2, 6, 1]) before transposing

        keys_transposed = keys.transpose(2, 3) turns it into (2, 2, 1, 6)
        torch.tensor([
            # ================= BATCH 0 =================
            [
                # --- Head 0 (shape:) ---
                [
                    [ 0.3830,  0.4763,  0.4537,  0.3028, -0.0808,  0.5371]
                ],
                # --- Head 1 (shape:) ---
                [
                    [-0.0107,  0.2750,  0.2543,  0.2335, -0.1899,  0.4397]
                ],
            ],

            # ================= BATCH 1 =================
            [
                # --- Head 0 (shape:) ---
                [
                    [ 0.1100,  0.2200,  0.3300,  0.4400,  0.5500,  0.6600]
                ],
                # --- Head 1 (shape:) ---
                [
                    [-0.1100, -0.2200, -0.3300, -0.4400, -0.5500, -0.6600]
                ],
            ]
        ])
                # --- Head 0 (shape:) ---
                [
                    [ 0.1100,  0.2200,  0.3300,  0.4400,  0.5500,  0.6600]
                ],
                # --- Head 1 (shape:) ---
                [
                    [-0.1100, -0.2200, -0.3300, -0.4400, -0.5500, -0.6600]
                ],
            ]
        ])
        # keys_transposed.shape -> torch.Size([2, 2, 1, 6])

        so now we can muliply queries by keys to get attention scores.

        after multiplication with queries, the attention scores will have shape [batch, heads, tokens, tokens]
        attn_scores = torch.tensor([
            # ================= BATCH 0 =================
            [
                # --- Head 0 (shape: 6x6 score grid) ---
                [
                    # key 0    key 1    key 2    key 3     key 4    key 5
                    [ 0.3830,  0.4763,  0.4537,  0.3028, -0.0808,  0.5371],  # query 0 (1.0 * keys)
                    [ 0.7660,  0.9526,  0.9074,  0.6056, -0.1616,  1.0742],  # query 1 (2.0 * keys)
                    [ 0.1149,  0.1429,  0.1361,  0.0908, -0.0242,  0.1611],  # query 2 (e.g. 0.3 * keys)
                    [-0.1915, -0.2382, -0.2269, -0.1514,  0.0404, -0.2686],  # query 3 (e.g. -0.5 * keys)
                    [ 0.3830,  0.4763,  0.4537,  0.3028, -0.0808,  0.5371],  # query 4 (e.g. 1.0 * keys)
                    [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],  # query 5 (e.g. 0.0 * keys)
                ],
                # --- Head 1 (shape: 6x6 score grid) ---
                [
                    [-0.0107,  0.2750,  0.2543,  0.2335, -0.1899,  0.4397],  # query 0
                    [-0.0214,  0.5500,  0.5086,  0.4670, -0.3798,  0.8794],  # query 1
                    [ ...  6 scores for query 2 ... ],
                    [ ...  6 scores for query 3 ... ],
                    [ ...  6 scores for query 4 ... ],
                    [ ...  6 scores for query 5 ... ],
                ],
            ],

            # ================= BATCH 1 =================
            [
                # --- Head 0 (shape: 6x6 score grid) ---
                [
                    [ ... 6 rows x 6 columns ... ]
                ],
                # --- Head 1 (shape: 6x6 score grid) ---
                [
                    [ ... 6 rows x 6 columns ... ]
                ],
            ]
        ])
        """
        attn_scores = queries @ keys.transpose(2, 3)

        # The same causal mask broadcasts across batches and heads, blocking
        # future tokens independently in each head.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        """
        attention scores shape torch.Size([2, 2, 6, 6])
        tensor([
          [ 
            [
              [ 0.0541,    -inf,    -inf,    -inf,    -inf,    -inf],
              [ 0.0673,  0.2717,    -inf,    -inf,    -inf,    -inf],
              [ 0.0641,  0.2588,  0.2597,    -inf,    -inf,    -inf],
              [ 0.0428,  0.1727,  0.1733,  0.1000,    -inf,    -inf],
              [-0.0114, -0.0461, -0.0462, -0.0267, -0.0360,    -inf],
              [ 0.0759,  0.3064,  0.3074,  0.1775,  0.2396,  0.1778],
            ],
            [
              [-0.0033,    -inf,    -inf,    -inf,    -inf,    -inf],
              [ 0.0860,  0.0711,    -inf,    -inf,    -inf,    -inf],
              [ 0.0795,  0.0657,  0.0685,    -inf,    -inf,    -inf],
              [ 0.0730,  0.0604,  0.0629,  0.0191,    -inf,    -inf],
              [-0.0594, -0.0491, -0.0511, -0.0155, -0.0737,    -inf],
              [ 0.1375,  0.1137,  0.1184,  0.0359,  0.1708, -0.0047],
            ],
          ],
          [ 
            [ ... 6 rows x 6 columns for head 0 ... ],
            [ ... 6 rows x 6 columns for head 1 ... ],
          ],
        ], grad_fn=<MaskedFillBackward0>)
        """

        # Normalize over keys for each query in each head; dropout is active
        # during training and disabled in evaluation mode.
        scale = keys.shape[-1] ** 0.5 # 1
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Each head independently combines its own values using its own weights.
        # [batch, heads, tokens, head_dim] -> [batch, tokens, heads, head_dim].
        """
        attention weights shape torch.Size([2, 2, 6, 6])
        tensor([
            [ 
                [
                    [1.1111, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.4990, 0.6121, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.3238, 0.3935, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.2562, 0.2917, 0.2919, 0.2713, 0.0000, 0.0000],
                    [0.2271, 0.2194, 0.2193, 0.2237, 0.0000, 0.0000],
                    [0.1608, 0.2024, 0.0000, 0.1779, 0.1894, 0.1780],
                ],
                [
                    [1.1111, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.5597, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.3734, 0.3683, 0.3693, 0.0000, 0.0000, 0.0000],
                    [0.2831, 0.2795, 0.2802, 0.2682, 0.0000, 0.0000],
                    [0.2201, 0.2223, 0.2219, 0.2299, 0.2169, 0.0000],
                    [0.1928, 0.1883, 0.1892, 0.1742, 0.1993, 0.1673],
                ],
            ],
            [ 
                [
                    [1.1111, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.6121, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.3238, 0.3935, 0.3938, 0.0000, 0.0000, 0.0000],
                    [0.2562, 0.2917, 0.2919, 0.2713, 0.0000, 0.0000],
                    [0.2271, 0.2194, 0.2193, 0.2237, 0.2216, 0.0000],
                    [0.1608, 0.2024, 0.0000, 0.1779, 0.0000, 0.1780],
                ],
                [
                    [1.1111, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.5597, 0.5514, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.3734, 0.3683, 0.3693, 0.0000, 0.0000, 0.0000],
                    [0.2831, 0.2795, 0.2802, 0.2682, 0.0000, 0.0000],
                    [0.2201, 0.2223, 0.2219, 0.2299, 0.2169, 0.0000],
                    [0.1928, 0.1883, 0.1892, 0.1742, 0.1993, 0.1673],
                ],
            ],
        ], grad_fn=<MulBackward0>)

        values shape torch.Size([2, 2, 6, 1])
        tensor([
            [
                [
                    [0.4429],
                    [0.4864],
                    [0.4878],
                    [0.2311],
                    [0.3758],
                    [0.2370],
                ],
                [
                    [0.1077],
                    [0.3987],
                    [0.3865],
                    [0.2684],
                    [0.0570],
                    [0.3989],
                ],
            ],
            [ 
                [
                    [0.4429],
                    [0.4864],
                    [0.4878],
                    [0.2311],
                    [0.3758],
                    [0.2370],
                ],
                [
                    [0.1077],
                    [0.3987],
                    [0.3865],
                    [0.2684],
                    [0.0570],
                    [0.3989],
                ],
            ],
        ], grad_fn=<TransposeBackward0>)
        """
        context_vec = (attn_weights @ values).transpose(1, 2) 

        """
        context_vec shape torch.Size([2, 6, 2, 1])
        context_vec shape before view:  tensor([
        [
            [
                [0.4921],
                [0.1196]
            ],
            [
                [0.5187],
                [0.0603]
            ],
            [
                [0.3348],
                [0.3298]
            ],
            [
                [0.4604],
                [0.3222]
            ],

            [
                [0.3660],
                [0.2722]
            ],
            [
                [0.3241], head 0 -> for token "step"
                [0.2938]  head 1 -> for token "step"
            ]
        ],
        [
          [
            [0.4921],
            [0.1196]
          ],
          [
            [0.2977],
            [0.2801]
          ],
          [
            [0.5269],
            [0.3298]
          ],
          [
            [0.4604],
            [0.3222]
          ],
          [
            [0.4492],
            [0.2722]
          ],
          [
            [0.2530],
            [0.2938]
          ]
        ]], grad_fn=<TransposeBackward0>)
        """

        # Concatenate the head features for each token
        # this will combine the features from all heads for each token
        # reshaping it to [batch_size, num_tokens, d_out]
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)

        """
            context_vec tensor([[
                [0.4921, 0.1196],
                [0.5187, 0.0603],
                [0.3348, 0.3298],
                [0.4604, 0.3222],
                [0.3660, 0.2722],
                [0.3241, 0.2938]],

                [[0.4921, 0.1196],
                [0.2977, 0.2801],
                [0.5269, 0.3298],
                [0.4604, 0.3222],
                [0.4492, 0.2722],
                [0.2530, 0.2938]]], grad_fn=<ViewBackward0>)

        context_vec after view shape torch.Size([2, 6, 2])
        
        """
        # out_proj is a learned linear layer that mixes the concatenated head features for each token independently, keeping the width d_out.
        # For example, with two one-dimensional heads and input [0.3, 0.5]:
        # output 0 = 0.3 * weight[0, 0] + 0.5 * weight[0, 1] + bias[0]
        # output 1 = 0.3 * weight[1, 0] + 0.5 * weight[1, 1] + bias[1]
        # Both outputs can use both heads. The same weights and bias are applied to every token independently.
        return self.out_proj(context_vec)  # optional projection

torch.manual_seed(42)
batch_size, context_length, d_in = batch.shape # 2, 6, 3
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, num_heads=2)
context_vectors = mha(batch)

"""
context_vectors.shape
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

"""
CONCLUSION: WHAT HAVE WE LEARNED ABOUT MULTI-HEAD ATTENTION?

- Multi-head attention allows the model to focus on different parts of the input sequence simultaneously.

- Instead of creating separate nn.Linear layers per head, we combine them into one large linear projection (d_in -> d_out) for W_query, W_key, and W_value—and then  slice/split the output into heads via .view() and .transpose(1, 2). That is why splitting via .view() works!

- We can prepare each head to do computation independently, attending to different parts of the sequence.

- We merge the heads side-by-side back using .transpose(1, 2).contiguous().view(...), restoring the original [batch, tokens, d_out] shape (see notes below how this works).

- We can optionally apply another learned linear projection to mix the information from all heads for each token independently.
"""

""""
STEP 1: RIGHT AFTER (attn_weights @ values)
Shape: [batch=2, heads=2, tokens=6, head_dim=1]
(Each head has its own separate 6x1 column of updated token values)

  HEAD 0 (6 x 1)         HEAD 1 (6 x 1)
  +----------+           +----------+

  | [0.4921] | (tok 0)   | [0.1197] | (tok 0)
  | [0.5187] | (tok 1)   | [0.3541] | (tok 1)
  | [0.3348] | (tok 2)   | [0.3302] | (tok 2)
  | [0.4604] | (tok 3)   | [0.2768] | (tok 3)
  | [0.3660] | (tok 4)   | [0.2314] | (tok 4)
  | [0.3241] | (tok 5)   | [0.2735] | (tok 5)
  +----------+           +----------+

                       |
                       |  context_vec.transpose(1, 2)
                       |  (Put tokens back as rows, heads side-by-side)
                       v

STEP 2: AFTER .transpose(1, 2)
Shape: [batch=2, tokens=6, heads=2, head_dim=1]
(Each token row now holds two separate inner lists: [head_0] and [head_1])

              head 0       head 1
            +----------+ +----------+
token 0 ->  | [0.4921] | | [0.1197] |
token 1 ->  | [0.5187] | | [0.3541] |
token 2 ->  | [0.3348] | | [0.3302] |
token 3 ->  | [0.4604] | | [0.2768] |
token 4 ->  | [0.3660] | | [0.2314] |
token 5 ->  | [0.3241] | | [0.2735] |
            +----------+ +----------+

                       |
                       |  .contiguous().view(batch_size, num_tokens, self.d_out)
                       |  (Removes the inner divider between head 0 and head 1)
                       v

STEP 3: AFTER .contiguous().view(2, 6, 2)
Shape: [batch=2, tokens=6, d_out=2]
(Head 0 and Head 1 are glued together into one single feature vector per token)

            +--------------------+
token 0 ->  | [0.4921, 0.1197]   |
token 1 ->  | [0.5187, 0.3541]   |
token 2 ->  | [0.3348, 0.3302]   |
token 3 ->  | [0.4604, 0.2768]   |
token 4 ->  | [0.3660, 0.2314]   |
token 5 ->  | [0.3241, 0.2735]   |
            +--------------------+
"""