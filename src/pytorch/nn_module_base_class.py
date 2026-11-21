"""
with PyTorch you can define neural network models by subclassing the nn.Module base class.
You need to implement the __init__ method to define the layers
"""

""" 
The following nn has 10 input features 

1st hidden layer has 30 neurons
2nd hidden layer has 20 neurons
output layer has num_outputs neurons

How the Math Works: During the forward pass, PyTorch transposes the weight matrix W^T, which becomes [50, 30]) to multiply it with your input x ([batch_size, 50]), and then adds the bias vector ([30]) via broadcasting:

output = x @ W^T + b

• our Input x: has shape (2, 50). Each row is one data sample with 50 columns (features).
• Stored Weights w: Has shape (30, 50). It has 30 rows and 50 columns.
• The Mismatch: You cannot directly multiply (batch_size, 50) by (30, 50) because 50 ≠ 30.
• The Fix W^T: Transposing W flips its dimensions to (50, 30)—giving it 50 rows and 30 columns. Now the inner dimensions match: (2, 50) @ (50, 30) -> (2, 30)


What Happens During That Multiplication

• Row-by-Column Dot Product: Each sample's 50 input values (a row in x) are multiplied element-by-element against a neuron's 50 weights (now a column in W^T) and summed together.

• Result: Doing this across all 30 columns produces 30 output values per sample, ready for the 30 biases to be added.

the number of rows given originally by the batch size (2) stays the exact same from start to finish, while only the number of columns changes at each linear layer (50 -> 30 -> 20 -> 3).

PyTorch does not loop over Row 0 and Row 1 one by one—it multiplies the entire [2, 50] matrix against the layer's weights all at once.

Step-by-Step Math for Layer 1 (nn.Linear(50, 30))

1. Matrix Multiplication x @ W^T
	• Your input x has shape [2, 50].
	• The transposed weight matrix W^T has shape [50, 30].
	• Multiplying [2, 50] by [50, 30] produces a [2, 30] matrix:
		• Row 0 of x (50 numbers) is multiplied against all 30 columns of W^T to form Row 0 of the output (30 numbers).
		• Row 1 of x (50 numbers) is multiplied against those exact same 30 columns of W^T to form Row 1 of the output (30 numbers).

2. Adding the Bias (+ b):
	• The bias vector b has shape [30] (one bias per output neuron).
	• PyTorch uses broadcasting to automatically add those exact same 30 bias numbers to both Row 0 and Row 1.

3. Applying ReLU:
	• ReLU checks all 60 numbers in the [2, 30] table at once, turning any negative number into 0 while keeping the [2, 30] shape unchanged.

How It Continues Through the Rest of the Network

• Layer 2 (nn.Linear(30, 20) + ReLU): Multiplies [2, 30] by [30, 20], adds the [20] biases to both rows, and applies ReLU, resulting in [2, 20].
• Output Layer (nn.Linear(20, 3)): Multiplies [2, 20] by [20, 3] and adds the [3] biases to both rows, giving your final [2, 3] output.

Because both rows use the exact same weights and biases independently, passing a [2, 50] matrix gives the exact same answers as passing each [1, 50] row separately—just much faster!
"""
from torch import manual_seed, nn, randn


class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()

        self.layers = nn.Sequential(
            # 1st hidden layer 
            nn.Linear(num_inputs, 30),
            nn.ReLU(),

            # 2nd hidden layer has to match the output of the 1st hidden layer
            nn.Linear(30, 20),
            nn.ReLU(),
            
            # output layer: the output layer has to match number of outputs of the last layer
            nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


model = NeuralNetwork(num_inputs=50, num_outputs=3)
# print the number of trainable parameters
# num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(num_trainable_params)

# print trainable paramaters of the first layer
# print(model.layers[0].weight)

# print the shape of the first layer's weight
#print(model.layers[0].weight.shape) # torch.Size([30, 50]) -> 30 rows, 50 columns

# lets access the bias of the first layer
#print(model.layers[0].bias)

# print the shape of the first layer's bias
#print(model.layers[0].bias.shape) # torch.Size([30]) -> 30 elements

## let's see how this simple NN can be used in a forward pass
manual_seed(123)  # for reproducibility
x = randn(1, 50)
logits = model(x) 
#print(logits) # output tensor([[-0.0502, -0.3148,  0.0149]], grad_fn=<AddmmBackward0>)
# this corresponds to the 3 output neurons
# this AddmmBackward0 indicates that the output was computed using the matrix multiplication and addition operation in the forward pass. It reprsents the last used function to compute a var in the computation graph.
# if we want to use this network for predications, we can just use, constructing computational grpah can waste resources
# w e can wrap the forward pass in torch.no_grad() to avoid constructing the computational graph
# with torch.no_grad():
#     predictions = model(x)

# in PyTorch, its a common practice to code models such that they return the raw logits without applying any activation like softmax at the end. This allows for more numerical stability and flexibility when using different loss functions.

# so if we want to compute probabilities from the logits, we can apply a softmax function
# with torch.no_grad():
#     probabilities = torch.softmax(logits, dim=1) # these values can be interpreted as class probabilities that sum to 1 along the specified dimension