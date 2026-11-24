"""
for many LLM tasks, we want the self-attention mechanism to be causal, meaning that each token can only attend to previous tokens and itself, but not to future tokens. This ensures that the model generates text in a left-to-right manner, respecting the autoregressive property of language modeling.

It is called "causal" attention because it strictly obeys the law of cause and effect: an earlier word (the cause) can influence a later word (the effect), never the other way around.

In physics and statistics, causality means that an event at time t can only depend on events that happened at or before time t—never on events from the future (t + 1, t + 2, ...).
"""

# WE WILL MODIFY STANDARD SELF-ATTENTION TO INCORPORATE A CAUSAL ATTENTION MECHANISM BY MASKING FUTURE TOKENS

# one way to obtain the masked attention weight matrix in causal self attention is to apply the softfax function
# to the attention scores, zeroing out the elements above the digonal and normalizing the remaining resulting matrix

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


class SelfAttentionV2(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        self.W_query: torch.nn.Linear = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key: torch.nn.Linear = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value: torch.nn.Linear = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries: torch.Tensor = self.W_query(x)
        keys: torch.Tensor = self.W_key(x)
        values: torch.Tensor = self.W_value(x)

        attn_scores = queries @ keys.T
        scale = keys.shape[1] ** 0.5
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        return attn_weights @ values


sa_v2 = SelfAttentionV2(d_in=3, d_out=2)

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
values = sa_v2.W_value(inputs)
attn_scores = queries @ keys.T
scale = keys.shape[1] ** 0.5
attn_weights = torch.softmax(attn_scores / scale, dim=-1)
"""
output of attn_weights:
tensor([[0.1739, 0.1489, 0.1499, 0.1763, 0.1875, 0.1635],
        [0.1676, 0.1527, 0.1538, 0.1745, 0.1901, 0.1614],
        [0.1678, 0.1527, 0.1537, 0.1745, 0.1897, 0.1615],
        [0.1655, 0.1608, 0.1614, 0.1701, 0.1786, 0.1636],
        [0.1721, 0.1559, 0.1564, 0.1725, 0.1777, 0.1655],
        [0.1633, 0.1606, 0.1613, 0.1704, 0.1824, 0.1621]])
"""

# NOW WE CAN IMPLEMENT CAUSAL SELF-ATTENTION BY MASKING FUTURE TOKENS USING PYTORCH TRIL FUNC 
context_length = inputs.shape[0] # 6
mask_simple = torch.tril(torch.ones(context_length, context_length))
"""
output of mask_simple:
tensor([[1., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1.]])
"""

# NOW WE CAN MULTIPLY THE ATTENTION SCORES BY THE MASK TO ZERO OUT FUTURE TOKENS
attn_scores_masked_simple = attn_scores * mask_simple
"""
output of attn_scores_masked_simple:
tensor([[ 0.0676, -0.0000, -0.0000, -0.0000, -0.0000,  0.0000],
        [ 0.2174, -0.0368, -0.0000, -0.0000, -0.0000,  0.0000],
        [ 0.2223, -0.0362, -0.0430, -0.0000, -0.0000,  0.0000],
        [ 0.1117, -0.0223, -0.0264, -0.0199, -0.0000,  0.0000],
        [ 0.2488, -0.0145, -0.0174, -0.0404, -0.0649, -0.0000],
        [ 0.0718, -0.0297, -0.0352, -0.0144, -0.1241,  0.0378]],
       grad_fn=<MulBackward0>)
"""

# but we can implement it more effeciently with negative infinity values before applying softmax
mask = torch.tril(torch.ones(context_length, context_length))
masked = attn_scores.masked_fill(mask == 0, -torch.inf)
"""
output of masked:
tensor([[-0.1114,    -inf,    -inf,    -inf,    -inf,    -inf],
        [ 0.0075,  0.1188,    -inf,    -inf,    -inf,    -inf],
        [ 0.0084,  0.1225,  0.1219,    -inf,    -inf,    -inf],
        [ 0.0202,  0.0499,  0.0498,  0.0286,    -inf,    -inf],
        [ 0.0206,  0.1536,  0.1529,  0.0953,  0.0969,    -inf],
        [ 0.0116,  0.0196,  0.0196,  0.0106,  0.0135,  0.0116]],
       grad_fn=<MaskedFillBackward0>)

• The old way:
    - In math e^0 = 1.
	- Instead of turning a future word's weight into 0, putting a 0 score gives that future word a positive share of 1 in the softmax pool—so the model would still pay attention to future words!
    - so there are additional calculations needed to renormalize the attention weights after masking.

• The -inf way (before softmax):
	• In math, e^{-infty} = 0.
	• When softmax exponentiates -inf, those future positions automatically turn into exact 0.0s, and softmax divides only by the valid past/current tokens in one single step
"""

# NOW WE CAN APPLY SOFTMAX TO THE MASKED ATTENTION SCORES TO GET THE FINAL ATTENTION WEIGHTS
attn_weights_masked = torch.softmax(masked / scale, dim=-1)
"""
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.4949, 0.5051, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3288, 0.3357, 0.3355, 0.0000, 0.0000, 0.0000],
        [0.2491, 0.2506, 0.2506, 0.2497, 0.0000, 0.0000],
        [0.1973, 0.2013, 0.2012, 0.2009, 0.1993, 0.0000],
        [0.1665, 0.1672, 0.1672, 0.1664, 0.1660, 0.1668]],
       grad_fn=<SoftmaxBackward0>)
"""

"""
We can now compute the context vectors via context_vectors = attn_weights_masked @ values

But before, let's explore optimization that we can add by masking additional attention weights with dropout.
Dropout is a technique in deep learning where randomly selected hidden layers are ignored during training to prevent overfitting and improve generalization.
This technique ensures that the model does not become overly reliant on any particular hidden unit, promoting robustness and better generalization to unseen data.
Dropout is used in training only and is typically disabled during evaluation or inference to use the full capacity of the model.

Dropout is applied in 2 places in training: 
 - On the attention weights, to randomly ignore some attention connections during training.
 - or after applying the attention weights to the value vectors, to randomly ignore some of the resulting context information during training.
"""

# WE WILL APPLY DROPOUT RATE AFTER COMPUTING THE ATTENTION WEIGHTS
torch.manual_seed(42)  # for reproducibility
dropout = torch.nn.Dropout(p=0.5)  # 50% dropout rate
example = torch.ones(6, 6)
"""
output of example
tensor([[1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.]]) 
"""
dropout(example)
"""
as we can see about 50% of the elements are set to 0 due to dropout. To compensate for the reduction in active elements in the matrix, the remaining elements are scaled by a factor of 1/0.5 = 2. This scaling ensures that the expected sum of the elements remains consistent before and after applying dropout.
tensor([[2., 2., 2., 2., 0., 2.],
        [0., 0., 2., 2., 2., 2.],
        [0., 0., 2., 0., 2., 0.],
        [0., 2., 2., 0., 2., 2.],
        [2., 2., 0., 2., 2., 2.],
        [2., 2., 2., 2., 0., 0.]])
"""

# now let's apply dropout to the attention weights
attn_weights_masked_dropout = dropout(attn_weights_masked)
"""
output of attn_weights_masked_dropout
tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [1.0268, 0.9732, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5139, 0.0000, 0.5073, 0.4719, 0.0000, 0.0000],
        [0.4188, 0.0000, 0.3980, 0.0000, 0.3915, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.3111, 0.0000, 0.0000]],
       grad_fn=<MulBackward0>)
"""

""" 
linear = torch.nn.Linear(3, 2)
output = linear(inputs)
print("shape at -1", output.shape[-1]) # output 2 
"""

# WE WILL NOW DEVELOP CAUSAL ATTENTION CLASS
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

        # this code ensures that this buffer automatically moves to the correct device along with the model and is not considered a learnable parameter. Avoiding device mismatch errors.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length, dtype=torch.bool)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_token = x.shape[1] # output 6
        if num_token > self.mask.shape[0]: 
            raise ValueError(
                f"Sequence length {num_token} exceeds context length {self.mask.shape[0]}"
            )

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        """
        we transpose here by swapping the last two dimensions of the keys tensor, 
        because our keys have shape of [2, 6, 3] and we need it to be [2, 3, 6] for the matrix multiplication with queries of shape [2, 6, 3]
        """
        attn_scores = queries @ keys.transpose(-2, -1)

        """
        Code:
            attn_scores = attn_scores.masked_fill(
                ~self.mask[None, :num_token, :num_token], -torch.inf
            )
        What Each Piece Does (Inside-Out):

        1. [:num_token, :num_token]  -->  TRIMS THE MASK TO FIT
        Slices the top-left corner of the pre-built mask to match the exact number of tokens in the current sentence (in case the sentence is shorter than max context_length).

      ([[[ True, False, False, False, False, False],
         [ True,  True, False, False, False, False],
         [ True,  True,  True, False, False, False],
         [ True,  True,  True,  True, False, False],
         [ True,  True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True,  True]]])

        2. [None, ...]  -->  ADDS A BATCH DIMENSION
        Turns the 2D mask (6, 6) into a 3D mask (1, 6, 6) so PyTorch can automatically broadcast (apply) it across all sentences in the batch (e.g., shape (2, 6, 6)).

        3. ~  -->  FLIPS TRUE AND FALSE (BOOLEAN NOT)
        self.mask has True for allowed words and False for future words. Because masked_fill() overwrites wherever the condition is True, ~ flips the booleans so ONLY future words become True.
      ([[[False,  True,  True,  True,  True,  True],
         [False, False,  True,  True,  True,  True],
         [False, False, False,  True,  True,  True],
         [False, False, False, False,  True,  True],
         [False, False, False, False, False,  True],
         [False, False, False, False, False, False]]])

        4. .masked_fill(..., -torch.inf)  -->  BLOCKS FUTURE TOKENS
        Replaces every future token's score (where the flipped mask is True) with negative infinity (-inf), guaranteeing softmax turns them into exact 0.0 weights.
        [[ 0.1782,    -inf,    -inf,    -inf,    -inf,    -inf],
         [ 0.1597,  0.4725,    -inf,    -inf,    -inf,    -inf],
         [ 0.1620,  0.4693,  0.4523,    -inf,    -inf,    -inf],
         [ 0.0653,  0.2513,  0.2409,  0.1773,    -inf,    -inf],
         [ 0.1586,  0.2794,  0.2717,  0.1723,  0.0567,    -inf],
         [ 0.0496,  0.3000,  0.2866,  0.2190, -0.0362,  0.3573]]],

        """
        attn_scores = attn_scores.masked_fill(
            ~self.mask[None, :num_token, :num_token], -torch.inf
        )
        scale = keys.shape[-1] ** 0.5
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ values

batch = torch.stack((inputs, inputs))
"""
batch: torch.Size([2, 6, 3])
tensor([
         [[0.4300, 0.1500, 0.8900],
         [0.5500, 0.8700, 0.6600],
         [0.5700, 0.8500, 0.6400],
         [0.2200, 0.5800, 0.3300],
         [0.7700, 0.2500, 0.1000],
         [0.0500, 0.8000, 0.5500]],

        [[0.4300, 0.1500, 0.8900],
         [0.5500, 0.8700, 0.6600],
         [0.5700, 0.8500, 0.6400],
         [0.2200, 0.5800, 0.3300],
         [0.7700, 0.2500, 0.1000],
         [0.0500, 0.8000, 0.5500]]
    ])
    
  shape size: torch.Size([2, 3, 6]) 
  tensor([
    [
        [0.4300, 0.5500, 0.5700, 0.2200, 0.7700, 0.0500],
        [0.1500, 0.8700, 0.8500, 0.5800, 0.2500, 0.8000],
        [0.8900, 0.6600, 0.6400, 0.3300, 0.1000, 0.5500],
    ],
    [
        [0.4300, 0.5500, 0.5700, 0.2200, 0.7700, 0.0500],
        [0.1500, 0.8700, 0.8500, 0.5800, 0.2500, 0.8000],
        [0.8900, 0.6600, 0.6400, 0.3300, 0.1000, 0.5500],
    ],
])

"""

"""
TRICK FOR MATCHING QUERIES & KEYS IN 3D (AND 4D): "COVER THE BATCH, KISS THE MIDDLE"

Please remember the output of one matrix needs to match the input.shape of the next matrix it will be multiplied with.
• Row of Matrix A: [0.20, 0.50, 0.30] (contains 3 numbers)
• Column of Matrix B: [
                       10.0, 
                       -4.0, 
                       2.5 ] (contains 3 numbers)

Calculation 
 ([0.20, 0.50, 0.30]) @ ([10.0, -4.0, 2.5]) = 0.20*10.0 + 0.50*(-4.0) + 0.30*2.5 = 2.0 - 2.0 + 0.75 = 0.75

 if there is mismatch in shape for exampes rows != columms or values in the matrices do not align, the matrix multiplication will not be valid and PyTorch will raise an error.


Whenever you multiply 3D tensors in PyTorch of shape [Batch, Tokens, Features], follow these 3 steps:

1. COVER THE FIRST NUMBER (THE BATCH)
Put your thumb over index 0 (Batch) on both shapes. PyTorch treats everything before the last two dimensions as independent parallel copies, 
so only the last two numbers ever multiply.

  queries: [2, 6, 3]  -->  focus on (6, 3)
  keys:    [2, 6, 3]  -->  focus on (6, 3)

2. CHECK THE INNER NEIGHBORS
Write the remaining 2D shapes side-by-side. The two numbers touching in the middle MUST be identical (columns of the first must match rows of the second):

  - Before transpose: (6, 3) @ (6, 3)  -->  Middle numbers are 3 and 6 (CLASH!)
  - After transpose:  (6, 3) @ (3, 6)  -->  Middle numbers are 3 and 3 (MATCH!)

3. Keep the outer numbers (the batch) aligned
After ensuring the middle numbers match, make sure the first numbers (the batch dimension) are the same for both tensors. This ensures that each batch element only attends to keys within the same batch.
rop the matching middle numbers (3), keep the outer numbers (6 and 6), and put the batch size (2) back in front:
(2, 6, 3) @ (2, 3, 6) --> (2, 6, 6)

Please note (2, 6, 6) is an intermideate tensor because it then multiplies with the values tensor of shape (2, 6, 3) to produce the final context vectors of shape (2, 6, 3).
(2, 6, 6) @ (2, 6, 3) --> (2, 6, 3)
 

BONUS SHORTCUT: ALWAYS USE keys.transpose(-2, -1)
Instead of hardcoding (1, 2) -- which breaks later when you add a 4th dimension for Multi-Head Attention [Batch, Heads, Tokens, Features] -- always write:

  attn_scores = queries @ keys.transpose(-2, -1)

Using (-2, -1) tells PyTorch: "No matter how many batch or h
"""


torch.manual_seed(42)
context_length = batch.shape[1] # output: 6
dim = batch.shape[-1] # output: 3
ca = CausalAttention(
    d_in=dim,
    d_out=dim,
    context_length=context_length,
    dropout_rate=0.1
)
context_vecs = ca(batch) # context_vecs.shape: torch.Size([2, 6, 3])
"""
output: tensor([[[-0.3673,  0.3073, -0.4233],
         [-0.1671,  0.1398, -0.1926],
         [-0.2631,  0.0983, -0.5868],
         [-0.2163,  0.0703, -0.5236],
         [-0.2185,  0.0444, -0.5159],
         [-0.1464,  0.0369, -0.3738]],

        [[-0.3673,  0.3073, -0.4233],
         [-0.2857,  0.1468, -0.5498],
         [-0.2631,  0.0983, -0.5868],
         [-0.2163,  0.0703, -0.5236],
         [-0.2185,  0.0444, -0.5159],
         [-0.1850,  0.0383, -0.4887]]], grad_fn=<UnsafeViewBackward0>)
"""

"""
CONCLUSION: WHY A CAUSAL MASK IS ESSENTIAL

1. PREVENTS "CHEATING" DURING TRAINING (NO DATA LEAKAGE)
During training, we feed the entire sentence ("Your journey starts with one step") into the model at once so it can learn in parallel. Without a causal mask, the token "journey" could look ahead at "starts" and simply copy the answer instead of learning to predict it from past context.

2. FORCES AUTOREGRESSIVE (LEFT-TO-RIGHT) GENERATION
By masking future positions with -inf (which becomes 0.0 after softmax), each token can ONLY attend to itself and earlier tokens:
  - "Your"    --> predicts "journey" (sees only "Your")
  - "journey" --> predicts "starts"  (sees "Your journey")
  - "starts"  --> predicts "with"    (sees "Your journey starts")

3. MATCHES REAL-WORLD INFERENCE
When generating new text in the real world, the model has no future words to look at—only the prompt typed so far. The causal mask ensures the model trains under the exact same one-way constraint it faces during live text generation.

4. Trivial Memorization Instead of Language Learning

During training, your data loader gives the model an input sequence ("Your journey starts with one") and a target sequence shifted by one word ("journey starts with one step"):

• At position 1 ("journey"), the model's job is to predict "starts".
• Without a causal mask, "journey"'s Query vector can attend directly to "starts"'s Key and Value vectors at position 2.
• Instead of learning grammar, semantics, or reasoning to guess what comes after "Your journey", the self-attention layer simply learns a shortcut: "copy whatever token is sitting one spot to my right."


5. Total Collapse During Real-World Generation (Inference)

When you actually deploy the model and ask it to finish the prompt "Your journey":

• There is no token sitting one spot to the right yet—the future doesn't exist.
• Because the model spent all of training relying on future tokens that aren't there during inference, its predictions break down into repetitive gibberish or random guesses.

"""