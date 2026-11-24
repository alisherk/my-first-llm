# when multiplying matrices it is important that inner dimensions match of the matrices macth

## Example 1: Switching Order of Multiplication

Suppose we have matrix `inputs` with shape (6, 3) and `attn_weights` with shape (6, 6)

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
attn_weights = torch.tensor(
    [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
    ]
)

all_context_vectors = inputs @ attn_weights  
if we try to multiply `inputs` with `attn_weights` as they are, we get an error:      
RuntimeError: mm: mat1 and mat2 shapes cannot be multiplied (6x3 and 6x6)
the columns of the first matrix must match the rows of the second matrix.

The error happens because the number of columns in `inputs` does not match the number of rows in `attn_weights`.
What we need to do is either transpose `attn_weights` and multiply it with `inputs`, or adjust the shapes so that the inner dimensions match.

Or, in this case we can simply switch the order of multiplication:
all_context_vectors = attn_weights @ inputs so that our shapes match: (6, 6) @ (6, 3) -> (6, 3)


## Example 2: Transposing Keys to Match Shapes
1. WHY (6, 1) @ (6, 1) FAILS:

  queries shape = [6, 1]
   queries = torch.tensor([
       [0.3830],
       [0.4763],
       [0.4537],
       [0.3028],
       [-0.0808],
       [0.5371]
   ])

  keys shape = [6, 1]
   keys = torch.tensor([
       [0.1413],
       [0.5704],
       [0.5724],
       [0.3304],
       [0.4462],
       [0.3310]
   ])

   (6, 1) @ (6, 1)  # Fails because inner dimensions do not match

   QUERIES (6 x 1)        KEYS (6 x 1)
   +---------+            +---------+

   |  0.3830 |            |  0.1413 |
   |  0.4763 |            |  0.5704 |
   |  0.4537 |     @      |  0.5724 |   <-- ERROR: 1 col != 6 rows
   |  0.3028 |            |  0.3304 |
   | -0.0808 |            |  0.4462 |
   |  0.5371 |            |  0.3310 |
   +---------+            +---------+


2. HOW TRANSPOSING KEYS FIXES IT — (6, 1) @ (1, 6) = (6, 6):
   "1 col matches 1 row! Every Query row multiplies across every Key column."

   QUERIES (6 x 1)             KEYS.T (1 x 6)
   +---------+                 +--------+--------+--------+--------+--------+--------+

   |  0.3830 |  (tok 0)        | 0.1413 | 0.5704 | 0.5724 | 0.3304 | 0.4462 | 0.3310 |
   |  0.4763 |  (tok 1)        +--------+--------+--------+--------+--------+--------+
   |  0.4537 |  (tok 2)   @      tok 0    tok 1    tok 2    tok 3    tok 4    tok 5
   |  0.3028 |  (tok 3)
   | -0.0808 |  (tok 4)
   |  0.5371 |  (tok 5)
   +---------+
                                
                                |  Row 0: 0.3830 * [all 6 keys] -> GET 0.0541, 0.2185, 0.2192, 0.1265, 0.1709, 0.1268
                                |  Row 1: 0.4763 * [all 6 keys] -> GET 0.0673, 0.2717, 0.2726, 0.1574, 0.2125, 0.1577
                                         
                                         |  
                                         v

   RESULT: ATTN_SCORES (6 x 6)
   +--------+--------+--------+--------+--------+--------+

   | 0.0541 | 0.2185 | 0.2192 | 0.1265 | 0.1709 | 0.1268 |  <- tok 0's scores for all keys
   | 0.0673 | 0.2717 | 0.2726 | 0.1574 | 0.2125 | 0.1577 |  <- tok 1's scores for all keys
   | 0.0641 | 0.2588 | 0.2597 | 0.1499 | 0.2024 | 0.1502 |  <- tok 2's scores for all keys
   | 0.0428 | 0.1727 | 0.1733 | 0.1000 | 0.1351 | 0.1002 |  <- tok 3's scores for all keys
   |-0.0114 |-0.0461 |-0.0462 |-0.0267 |-0.0361 |-0.0267 |  <- tok 4's scores for all keys
   | 0.0759 | 0.3064 | 0.3074 | 0.1775 | 0.2397 | 0.1778 |  <- tok 5's scores for all keys
   +--------+--------+--------+--------+--------+--------+
     key 0    key 1    key 2    key 3    key 4    key 5

## Example 3 (success when shapes match)
attention weights shape torch.Size([2, 2, 6, 6])
        tensor([
            [
              [
                [1.1111, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.4990, 0.6121, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.3238, 0.3935, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.2562, 0.2917, 0.2919, 0.2713, 0.0000, 0.0000],
                [0.2271, 0.2194, 0.2193, 0.2237, 0.0000, 0.0000],
                [0.1608, 0.2024, 0.0000, 0.1779, 0.1894, 0.1780]
              ],
              [
                [1.1111, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.5597, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.3734, 0.3683, 0.3693, 0.0000, 0.0000, 0.0000],
                [0.2831, 0.2795, 0.2802, 0.2682, 0.0000, 0.0000],
                [0.2201, 0.2223, 0.2219, 0.2299, 0.2169, 0.0000],
                [0.1928, 0.1883, 0.1892, 0.1742, 0.1993, 0.1673]
              ]
            ],
            [
              [
                [1.1111, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.6121, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.3238, 0.3935, 0.3938, 0.0000, 0.0000, 0.0000],
                [0.2562, 0.2917, 0.2919, 0.2713, 0.0000, 0.0000],
                [0.2271, 0.2194, 0.2193, 0.2237, 0.2216, 0.0000],
                [0.1608, 0.2024, 0.0000, 0.1779, 0.0000, 0.1780]
            ],
              [
                [1.1111, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.5597, 0.5514, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.3734, 0.3683, 0.3693, 0.0000, 0.0000, 0.0000],
                [0.2831, 0.2795, 0.2802, 0.2682, 0.0000, 0.0000],
                [0.2201, 0.2223, 0.2219, 0.2299, 0.2169, 0.0000],
                [0.1928, 0.1883, 0.1892, 0.1742, 0.1993, 0.1673]
              ]
            ]
          ]
        grad_fn=<MulBackward0>)

      and we have values tensor of shape [2, 2, 6, 1] as follows:
          tensor([
          [
            [
              [0.4429],
              [0.4864],
              [0.4878],
              [0.2311],
              [0.3758],
              [0.2370]
            ],
            [
              [0.1077],
              [0.3987],
              [0.3865],
              [0.2684],
              [0.0570],
              [0.3989]
            ],
          ],
          [
            [
              [0.4429],
              [0.4864],
              [0.4878],
              [0.2311],
              [0.3758],
              [0.2370]
            ],
            [
              [0.1077],
              [0.3987],
              [0.3865],
              [0.2684],
              [0.0570],
              [0.3989]
            ]
          ]
        ], grad_fn=<TransposeBackward0>)

In this example, we demonstrate how the attention weights matrix (6 x 6) is multiplied with the values matrix (6 x 1) to produce a context-aware representation for each token. So shapes match for matrix multiplication and we can successfully compute the resulting context vectors.

ATTN_WEIGHTS: (6 rows, 6 cols)        VALUES: (6 rows, 1 col)
                                              1 column
                                             +--------+
Row 0: [ w0,  w1,  w2,  w3,  w4,  w5 ]   @   |  v0    |  <- Row 0
       <--------- 6 columns --------->       |  v1    |  <- Row 1

                                             |  v2    |  <- Row 2
                                             |  v3    |  <- Row 3
                                             |  v4    |  <- Row 4
                                             |  v5    |  <- Row 5
                                             +--------+

Inner dimentions must match for matrix multiplication to be valid. In this case, the number of columns in the attention weights (6) must match the number of rows in the values (6).

How does Math works here 
3. WEIGHTED BLEND WITH VALUES — (6, 6) @ (6, 1) = (6, 1):
   "6 cols match 6 rows! Each row of weights blends all 6 value rows into 1 final vector."

   ATTN_WEIGHTS (6 x 6)                                         VALUES (6 x 1)
   +--------+--------+--------+--------+--------+--------+      +--------+

   | 1.1111 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |      | 0.4429 | (tok -> Your)
   | 0.4990 | 0.6121 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |      | 0.4864 | (tok -> journey)
   | 0.3238 | 0.3935 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  @   | 0.4878 | (tok -> starts)
   | 0.2562 | 0.2917 | 0.2919 | 0.2713 | 0.0000 | 0.0000 |      | 0.2311 | (tok -> with)
   | 0.2271 | 0.2194 | 0.2193 | 0.2237 | 0.0000 | 0.0000 |      | 0.3758 | (tok -> one)
   | 0.1608 | 0.2024 | 0.0000 | 0.1779 | 0.1894 | 0.1780 |      | 0.2370 | (tok -> step)
   +--------+--------+--------+--------+--------+--------+      +--------+
     tok 0    tok 1    tok 2    tok 3    tok 4    tok 5

        |  Row 0: (1.1111*0.4429) + (0*0.4864) + ...                    -> GET 0.4921               
        |  Row 1: (0.4990*0.4429) + (0.6121*0.4864) + (0*0.4878) + ...  -> GET 0.5187
        |  Row 2: (0.3238*0.4429) + (0.3935*0.4864) + (0*0.4878) + ...  -> GET 0.3348
        |  ...
        v

  RESULT: CONTEXT_VEC (6 x 1) for Head 0
   +--------+

   | 0.4921 |  <- tok Your -> new context-aware value
   | 0.5187 |  <- tok journey -> new context-aware value
   | 0.3348 |  <- tok starts -> new context-aware value
   | 0.4604 |  <- tok with -> new context-aware value
   | 0.3660 |  <- tok one -> new context-aware value
   | 0.3241 |  <- tok step -> new context-aware value
   +--------+


A Quick Memory Trick
Write the shapes next to each other and check the "inner neighbors":
• (6, 6) @ (6, 3) -> Inner numbers are 6 and 6 (Match! Outer numbers give the output shape: (6, 3)).
• (6, 3) @ (6, 6) -> Inner numbers are 3 and 6 (Mismatch!).

# Rows need to match the columns or columns need to match the rows for matrix multiplication to be valid.

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
After ensuring the middle numbers match, make sure the first numbers (the batch dimension) are the same for both tensors. This ensures that each batch element only attends to keys within the same batch. Drop the matching middle numbers (3), keep the outer numbers (6 and 6), and put the batch size (2) back in front:
(2, 6, 3) @ (2, 3, 6) --> (2, 6, 6)

Please note (2, 6, 6) is an intermideate tensor because it then multiplies with the values tensor of shape (2, 6, 3) to produce the final context vectors of shape (2, 6, 3).
(2, 6, 6) @ (2, 6, 3) --> (2, 6, 3)