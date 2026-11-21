"""
for the sake of simplicity, suppose we have a small vocab of only 6
words/tokens (instead of the 50000+ in a real LLM)

We want to create embedding of size 3 (in GPT-3, the embedding size is 12.288 dimensions)

Using the vocab size and output_dim, we can create the embedding layer in PyTorch as follows:
"""

import torch
from torch import nn

vocab_size = 6
output_dim = 3
torch.manual_seed(123)  # for reproducibility
embedding_layer = nn.Embedding(vocab_size, output_dim)
# print(embedding_layer.weight)

"""
output:
Parameter containing:
tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]], requires_grad=True)

The weight matrix of the embedding layers contains random values. These values will be learned and updated during the training process of the model. Moreover, we can see that the shape of the weight matrix is (vocab_size, output_dim), which corresponds to the number of tokens in the vocabulary and the size of the embedding vector for each token.
"""

#now lets apply it to a token id to obtain its embedding vector
token_id = torch.tensor([3])  # example token id
embedding_vector = embedding_layer(token_id)
# print(embedding_vector)

"""
output:
tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)
if we compare this embedding vector with the corresponding row in the weight matrix of the embedding layer, we can see that it macthes to the row with index 3.
so embedding layer essentially a lookup table that maps a token id to a specific row in the weight matrix, which represents the embedding vector for that token.
"""

# lets apply that to all 4 token ids
token_ids = torch.tensor([0, 1, 2, 3])
embedding_vectors = embedding_layer(token_ids)
# print(embedding_vectors)

"""
tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)

each row in the output corresponds to the embedding vector of the token id at the same position in the input tensor.
having now created embedding vector form token ids, nwext we'll add positional encodings to these embeddings to incorporate the order information of the tokens.
"""
