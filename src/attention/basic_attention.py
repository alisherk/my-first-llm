import torch

# BASIC SELF-ATTENTION BASED ON ONE INPUT VECTOR ATTENDING TO ALL OTHERS IN THE SEQUENCE
"""
Sequence of input vectors (x^1, x^2, ..., x^6) corresponding to the tokens in the sentence "Your journey starts with one step."
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

# here we will consider the second input vector in the sequence "journey"
second_input = inputs[
    1
]  # x^2 corresponding to "journey" second_input: tensor([0.5500, 0.8700, 0.6600])

attn_scores_2 = torch.empty(
    inputs.size(0)
)  # attention scores for the second input against all inputs

for i in range(inputs.size(0)):
    attn_scores_2[i] = torch.dot(second_input, inputs[i])

# print(attn_scores_2) # computed attention scores for the second input against all inputs: tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

# in the next step, we normalize each of the attention scores. The main goal
# is to convert them into a probability distribution, so that the sum of all attention scores for a given input equals 1.
attn_weights_2_tmp = torch.softmax(attn_scores_2, dim=0)
# print("attn_weights_2_tmp:", attn_weights_2_tmp) # tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# print("Sum of attn_weights_2_tmp:", attn_weights_2_tmp.sum()) # tensor(1.)

# Now that we computed the normalized attention weights for the second input, we are ready for the final step
# to calculate the context vector z(x^2) for the second input by multiplying
# each input vector by its corresponding attention weight and summing them up.
context_vector_2 = torch.zeros_like(second_input)
for i in range(inputs.size(0)):
    context_vector_2 += attn_weights_2_tmp[i] * inputs[i]
# print("context_vector_2:", context_vector_2) #tensor([0.4419, 0.6515, 0.5683])

# THE NEXT STEP IS TO EXTEND THIS COMPUTATION TO ALL INPUT VECTORS, NOT JUST THE SECOND ONE.

# This involves calculating attention scores, normalizing them, and computing context vectors for each input in the sequence.
attn_scores = torch.empty(
    inputs.size(0), inputs.size(0)
)  # attention scores for all inputs against all inputs

attn_scores = (
    inputs @ inputs.T
)  # compute attention scores for all inputs against all inputs using matrix multiplication

attn_weights = torch.softmax(
    attn_scores, dim=-1
)  # normalize attention scores along the last dimension

"""
by setting dim=-1 in the softmax function, we ensure that softmax is applied along the last dimension of the attention scores matrix, if attn_scores is a 2D matrix, this means it will normalize accross thwe columns so that values in each row sum to 1.

last_row_sum = attn_weights[-1].sum()  # sum of attention weights for the last input
print("last_row_sum:", last_row_sum)  # should print 1.0 since attention weights are normalized
"""

"""
print("attn_weights:", attn_weights)
tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
        [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
        [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
        [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
        [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
        [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
"""

all_context_vectors = attn_weights @ inputs

"""
print("all_context_vectors:", all_context_vectors)
tensor([[0.4421, 0.5931, 0.5790],
        [0.4419, 0.6515, 0.5683],
        [0.4431, 0.6496, 0.5671],
        [0.4304, 0.6298, 0.5510],
        [0.4671, 0.5910, 0.5266],
        [0.4177, 0.6503, 0.5645]])

if we flip the order of multiplication to inputs @ attn_weights, we get RunTimeError because the dimensions do not align for matrix multiplication. The correct order is attn_weights @ inputs to obtain the context vectors for each input.

all_context_vectors = inputs @ attn_weights            
RuntimeError: mm: mat1 and mat2 shapes cannot be multiplied (6x3 and 6x6)
the columns of the first matrix must match the rows of the second matrix.


A Quick Memory Trick
Write the shapes next to each other and check the "inner neighbors":
• (6, 6) @ (6, 3) -> Inner numbers are 6 and 6 (Match! Outer numbers give the output shape: (6, 3)).
• (6, 3) @ (6, 6) -> Inner numbers are 3 and 6 (Mismatch!).

If it important to note that each row from att_weights is multiplyed by each corresponding column in the input matrix so 
value 0.2098 is multiplied by the corresponding column in the input matrix [0.43, 0.15, 0.89] resulting in
[0.4421, 0.5931, 0.5790] 

Each row in attn_weights holds percentages summing to 1.0. So value 0.2098 represents the contribution of the first input vector to the context vector for the first input.
"""

"""
CONCLUSION:
In summary, our code takes 6 isolated token embeddings (inputs) and transforms them into 6 context-aware embeddings (all_context_vectors), where every token's vector is now a weighted blend of itself and all surrounding words in the sentence.
"""
