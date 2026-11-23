"""
why do we need positional encodings?
because the embedding vectors created so far only represent the identity of the tokens, but they do not contain any information about the order of the tokens in the sequence.
positional encodings are added to the token embeddings to provide the model with information about the relative or absolute positions of the tokens in the sequence.

absolute positioning describes where a token appears in the full sequence:
    position:  0    1    2
    token:    The  cat  sat
The token "cat" is at absolute position 1.

relative positioning describes the distance and direction between two tokens.
for example, "quietly" is two positions to the right of "cat":
    The  cat  sat  quietly

absolute positions answer "where is this token?"
relative positions answer "how far apart are these tokens?"
"""

from pathlib import Path

import torch
from text_tokenization import create_dataloader_v1
from torch import nn

vocab_size = 50257
output_dim = 256
token_embedding_layer = nn.Embedding(vocab_size, output_dim)
# output Embedding(50257, 256)

file_path = Path(__file__).with_name("the-verdict.txt")

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()


"""
if we sample data from the data loader, we embed each batch with 4 tokens into 256-dimensional vectors using the token embedding layer. So if we have a batch of 8 with 4 tokens each, the resulting tensor will have the shape (8, 4, 256).
"""
max_length = 4
data_loader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=False,
)

data_iter = iter(data_loader)
inputs, targets = next(data_iter)

# lets now use the embedding layer to imbed these tokens IDs into 256-dimensional vectors
embedded_inputs = token_embedding_layer(inputs)
# print(embedded_inputs.shape)  # output torch.Size([8, 4, 256])
"""
batch = [
    # batch 0
    [
        [256 values],  # Token position 0
        [256 values],  # Token position 1
        [256 values],  # Token position 2
        [256 values],  # Token position 3
    ],
    #batch 1
    [
        [256 values],
        [256 values],
        [256 values],
        [256 values],
    ],
    # ...batch 2 through 7...
]
# torch.Size([8, 4, 256])

print(embedded_inputs[0].shape)
# torch.Size([4, 256]) — all tokens in batch item 0

print(embedded_inputs[0, 2].shape)
# torch.Size([256]) — token 2 in batch item 0
"""

# for gpt model absolute embedding approach, we just need to create another embedding layer for positions
position_embedding_layer = nn.Embedding(max_length, output_dim)
position_embeddings = position_embedding_layer(torch.arange(max_length))
# print(position_embeddings.shape)  # output torch.Size([4, 256])
"""
we can see that position embeddings consists of 4 position vectors, each of 256 dimensions.
PyTorch can add 4 * 256-dimensional position vectors to the token embeddings to incorporate positional information.
"""
input_with_position = embedded_inputs + position_embeddings
# print(input_with_position.shape)  # output torch.Size([8, 4, 256])
"""
PyTorch adds the position embeddings using broadcasting:
    embedded_inputs       shape: [8, 4, 256]
    position_embeddings   shape: [4, 256]
    input_with_position   shape: [8, 4, 256]

The same position vectors are used for every sequence in the batch, so a token
at position 0 receives different positional information than the same token at
position 2. This is element-wise addition, not concatenation, so each token
still has 256 features and the resulting shape remains [8, 4, 256].

aha moment:
the position vector does not contain information about the meaning of a token.
It only adds information about the token's position in the sequence.

For example:
    token embedding("cat") + position embedding(0)
    = a combined representation of "cat" at position 0

The token embedding carries token-specific information, while the position
embedding carries position-specific information. For example: what does it mean for a token to be at position 0 versus position 2 in the sequence?

What happens at the linear algebra level: 

Think of the 256-dimensional space as a 3D room:
1. Token Subspace: The flat floor (2D).
2. Position Subspace: The vertical height (1D).
3. Where the Combined Vector Lives: Somewhere floating in the 3D room (e.g., 3 feet East, 1 foot North, and 2 feet up in the air).

Once the vector is floating in that combined space x_i = token_embedding_i + position_embedding_i, the Transformer's next layer (Self-Attention) multiplies {x}_{i} by weight matrices W_{Q}, W_{K}, (W_{V}). Depending on what a specific attention head needs to do, those matrices can interact with the subspaces in three ways:
• Read Only Meaning: Zero out the vertical (position) coordinates to project the vector back down onto the floor—useful for finding synonyms regardless of where they appear.

• Read Only Position: Zero out the floor (meaning) coordinates to look purely at vertical height—useful for an attention head whose only job is "always look at the immediately preceding word."

• Mix Both Together: Tilt the projection plane diagonally through the room so that meaning and position interact—for example, identifying whether "apple" appears at the start of a sentence (likely the subject) versus the end (likely the object).

Refer to token_position_interaction.png for a visual representation of how token and position subspaces interact in the combined vector space.

"""