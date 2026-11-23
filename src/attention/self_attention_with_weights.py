"""
So far, our code in basic_attention.py takes 6 isolated token embeddings (inputs) and transforms them into 6 context-aware embeddings (all_context_vectors), where every token's vector is now a weighted blend of itself and all surrounding words in the sentence.

In original transformer GPT models, This self-attention mechanism is also called "scaled dot-product attention".

The most notable diff here from basic_attention.py is the introduction of weight matrices that are updated during training. They are crucial so that the model can learn to assign appropriate attention scores based on the relationships between tokens, rather than relying solely on fixed dot-product similarities.

These matrices are typically referred to as the query, key, and value weight matrices (W_Q, W_K, W_V). They are used to project the input embeddings into different subspaces for computing attention scores and context vectors.

The relationship between the query, key, and value matrices is as follows is shown in query_key_value_matrices.png.
"""

# NOW WE WILL IMPLEMENT SELF-ATTENTION WITH TRAINABLE WEIGHTS

import torch

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

# lets do it for the second input x2 vector (token) in the sequence of "Your journey starts with one step"
x_2 = inputs[1]  # second input vector (token) in the sequence
d_in = inputs.size(1)  # 3
d_out = 2

"""
we use Parameter objects because they allow the weight matrices to be treated as learnable parameters within the model. This means that during training, gradients will be computed for these matrices, and they will be updated to minimize the loss function, enabling the model to learn appropriate attention scores.

torch.nn.Parameter marks a tensor as part of a model’s trainable state

Parameter is automatically registered when assigned inside an nn.Module. This allows PyTorch to find it through:
model.parameters()
optimizer = torch.optim.Adam(model.parameters())
"""
# now lets initialize the weight matrices for query, key, and value projections
torch.manual_seed(123)  # for reproducibility
W_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
"""
tensor([[-0.6018, -0.0251],
        [-1.0263, -1.1756],
        [-2.0158,  3.1956]])
"""
W_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
"""
tensor([[-0.8854,  1.2414],
        [ 0.2085,  0.8696],
        [ 0.1234, -0.5188]])
"""
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
"""
tensor([[-0.3762,  0.3580],
        [-0.2253,  0.6488],
        [ 0.3887,  0.8627]])
"""

# NEXT WE COMPUTE THE QUERY, KEY, AND VALUE VECTORS FOR x_2
query_2 = x_2 @ W_query #tensor([-2.4348,  1.4151]). The output results in 2-dimensional vector as we set d_out = 2
key_2 = x_2 @ W_key #tensor([-1.3100, -0.4763])
value_2 = x_2 @ W_value #tensor([0.5391, 0.5146])

# even tho our goal is to compute attention for the second input x_2, we still need the key and value vectors for all inputs to calculate the attention scores correctly.
keys = inputs @ W_key
values = inputs @ W_value

"""
keys shape for all inputs: torch.Size([6, 2])
values shape for all inputs: torch.Size([6, 2])

we projected the 6 computed input tokens from 3-dimensional space to 2-dimensional embedding space.
"""

# NOW COMPUTE ALL ATTENTION SCORES FOR THE SECOND INPUT x_2
att_scores_2 = query_2 @ keys.T  # compute attention scores for x_2 against all keys
# Attention scores for x_2 against all keys: tensor([-0.2542,  1.0479,  1.0689,  0.6391,  1.1461,  0.4688])
# this line measures how strongly the second input x_2 attends to each of the keys.

# NOW WE WANT TO GO FROM ATTENTION SCORES TO ATTENTION WEIGHTS
d_k = keys.shape[1]  # dimension of the key vectors (d_k): 2
attn_weights_2 = torch.softmax(att_scores_2 / (d_k ** 0.5), dim=-1)  # apply softmax to get attention weights
# tensor([0.1369, 0.1841, 0.1846, 0.1605, 0.1754, 0.1585])
# this simply converts the attention scores into probabilities that sum to 1, so that "journey" knows exactly how much of each word's value vector to blend into it's final context aware representation.
"""
the reason we divide the attention scores by the square root of the key dimension (d_k) before applying softmax is to prevent the scores from becoming too large. Large scores can lead to very small gradients during backpropagation, which can slow down learning. By scaling the scores, we maintain a more stable gradient and improve training dynamics.
"""

# THE FINAL STEP IS TO COMPUTE THE ATTENTION OUTPUT FOR x_2
attn_output_2 = attn_weights_2 @ values  # weighted sum of the value vectors
# Attention output for x_2: tensor([0.2854, 0.4081])

"""
CONCLUSION OF THE ABOVE ATTENTION COMPUTATION: 
1. Started with the token embedding (x_2): Isolated representation of "journey" with zero context from the rest of the sentence.
2. Checked all Keys (query_2 @ keys.T): Used "journey"'s Query to ask every word in the sentence (via their Keys), "How relevant are you to me?"
3. Turned scores into percentages (softmax): Normalized those relevance scores into attention weights (attn_weights_2) summing to 100%.
4. Blended all Values (attn_weights_2 @ values): Used those percentages to mix every word's Value vector into a single context-aware output (tensor([0.2854, 0.4081])) that now represents "journey" inside the specific context of "Your journey starts with one step."

The visual mechanism what we did above is shown at query_key_value_tramsform.png.
"""

# so far we computed the attention output for the second input x_2, which represents the word "journey" in the context of the entire sentence. Now we will generalize this process to compute attention outputs for all input tokens using self attention Python class
class SelfAttentionV1(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.randn(d_in, d_out))
        self.W_key = torch.nn.Parameter(torch.randn(d_in, d_out))
        self.W_value = torch.nn.Parameter(torch.randn(d_in, d_out))

        """"
        Parameter object stores weights by shape (d_in, d_out) = 3,2
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        scale = keys.shape[1] ** 0.5
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        return attn_weights @ values

class SelfAttentionV2(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        """
        Linear object stores wights by shape (d_out, d_in) = 2,3
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries: torch.Tensor = self.W_query(x)
        keys: torch.Tensor = self.W_key(x)
        values: torch.Tensor = self.W_value(x)

        attn_scores = queries @ keys.T
        scale = keys.shape[1] ** 0.5
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        return attn_weights @ values





