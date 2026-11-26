# Setting up efficient data loaders for PyTorch models

"""
PyTorch implements a DataSet and DataLoader abstraction to efficiently handle and iterate over large datasets. The DataSet class represents a dataset, while the DataLoader handles how the data is shuffled and assembled into batches during training or evaluation.
"""

import torch
import torch.nn.functional as F
from nn_module_base_class import NeuralNetwork
from torch import manual_seed, optim, tensor
from torch.utils.data import DataLoader, Dataset

# lets create a small toy dataset
# review how these points map on 2D plane in toy_dataset.png
X_train = tensor([
    [-1.2, 3.1], 
    [-0.9, 2.9], 
    [-0.5, 2.6], 
    [2.3, -1.1], 
    [2.7, -1.5]
]) 

# review how y train seperates these classes in points_classification.png
y_train = tensor([0, 0, 0, 1, 1])

x_test = tensor([
    [-1.8, 2.8], 
    [2.6, -1.6]
]) # so this is to give the network some unseen examples to evaluate on

y_test = tensor([0, 1]) # and this checks the network's predictions against the true labels for the test set

# defining a custom dataset class for PyTorch
    

class ToyDataset(Dataset):
    def __init__(self: ToyDataset, X: tensor, y: tensor):
        self.features: tensor = X
        self.labels: tensor = y

    def __len__(self: ToyDataset):
        return self.labels.shape[0]

    def __getitem__(self: ToyDataset, idx: int):
        one_x: tensor = self.features[idx]
        one_y: tensor = self.labels[idx]
        return one_x, one_y

# The purpose of the ToyDataset class is to provide a structured way to access the features and labels of our toy dataset. It allows PyTorch's DataLoader to efficiently batch and shuffle the data during training and evaluation.
train_dataset = ToyDataset(X_train, y_train)
test_dataset = ToyDataset(x_test, y_test)

# now we can create DataLoader instances for our training and test datasets
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

# after instantiating the DataLoader instances, we can now iterate over the batches of data during training or evaluation.
""" for batch_idx, (features, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}:")
    print("Features:", features)
    print("Labels:", labels) """

"""
Batch 0:
Features: tensor([[ 2.7000, -1.5000],[-1.2000,  3.1000]])
Labels: tensor([1, 0])

Batch 1:
Features: tensor([[-0.5000,  2.6000],[ 2.3000, -1.1000]])
Labels: tensor([0, 1])

Batch 2:
Features: tensor([[-0.9000,  2.9000]])
Labels: tensor([0])

as we can see based on the printed batches, the DataLoader is correctly batching and shuffling the training data.
It visits each example in the dataset exactly once per epoch, ensuring that all data points are used for training.
However, if you iterate over the DataLoader multiple times within the same epoch, the order of the batches may change due to shuffling. This is desired behavior to prevent NN models from overfitting to the order of the training data.

As you can see last batch only contains 1 example because 5 is not divisible by the batch size of 2.
In Practice having a smaller batch as the last batch in training epoch can disturb the convergence of the model, especially if the batch size is very small compared to the rest of the batches.
so we can drop the last incomplete batch by setting the `drop_last` parameter to `True` in the DataLoader. This ensures that all batches have the same size, which can help maintain stable training dynamics.
"""


# Lets now train a NN on the toy dataset

manual_seed(42)
device = torch.device("mps")
model = NeuralNetwork(num_inputs=2, num_outputs=2).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.005)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)

        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    model.eval()

# after we traing the model, we can evaluate its performance on the test dataset
model.eval()
with torch.no_grad():
    outputs = model(X_train.to(device))

torch.set_printoptions(precision=2, sci_mode=False)
probabilities = torch.softmax(outputs, dim=1)
#print(probabilities)

"""
output probabilities after the forward pass on the training data
tensor([[0.90, 0.10],
        [0.89, 0.11],
        [0.87, 0.13],
        [0.02, 0.98],
        [0.01, 0.99]])

this means that that first data point has 90% probability of belonging to class 0 and 10% probability of belonging to class 1.
similarly, the second data point has 89% probability of belonging to class 0 and 11% probability of belonging to class 1, and so on for the rest of the data points.
etc
"""

# we can also convert these values into class label predictions by taking the argmax along the class dimension
predicted_labels = torch.argmax(probabilities, dim=1)
#print(predicted_labels) # tensor([0, 0, 0, 1, 1])

# we can also compare the predicted labels with the true labels to evaluate the model's performance

# print(predicted_labels == y_train.to(device)) #tensor([True, True, True, True, True])
