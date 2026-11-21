# PyTorch has automatic differentiation capabilities through the autograd package 
# It allows you to compute gradients of tensors with respect to some scalar value
# In the context of deep learning, a computational graph has the sequence of operations need to compute
# the output a neural network. We need this to compute gradients for backpropagation.


# simple logistic linear regression forward pass
import torch
import torch.nn.functional as F

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
z = w1 * x1 + b
pred = torch.sigmoid(z)
loss = F.binary_cross_entropy(pred, y)

loss.backward()

""""
print(w1.grad)
print(b.grad)
outputs are: 
tensor([-0.0898])
tensor([-0.0817])
"""
