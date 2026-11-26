import torch

tensor0d = torch.tensor([1, 2, 3])  # stores integers as int64 by default
# print(tensor0d.dtype)

floatvec = torch.tensor([1.0, 2.0, 3.0])  # stores floats as float32 by default
# print(floatvec.dtype)

# change tensor0d to float32
tensor0d = tensor0d.to(torch.float32)
# print(tensor0d.dtype)

# common torch operations
tensor2d = torch.tensor([[1, 2], [3, 4], [5, 6]])  # 2D tensor
# print(tensor2d.shape) # tensors has shape (2, 2) which means 2 rows and 2 columns

# pytorch tensor also has view for reshaping
# tensor2d_view = tensor2d.view(1, 4) # reshape to 1D tensor with 4 elements
# print(tensor2d_view.shape) # should print torch.Size([1, 4])
# what is the diff between view and reshape
# view requires the new shape to be compatible with the original shape, otherwise it will raise an error

# for example, if we try to view a tensor with incompatible shape, it will raise an error
# tensor2d.view(3, 2) # this will raise an error because the original shape is (2, 2)

# reshape returns a new tensor with the same data but different shape, but may copy the data if needed
# for example, we can reshape tensor2d to (3, 2) without error
tensor2d_reshape = tensor2d.reshape(
    1, 6
)  # this will work, but may copy the data if needed
# print(tensor2d_reshape.shape) # should print torch.Size([3, 2])

# what is transpose operation
# transpose swaps the dimensions of a tensor by turning rows into columns and columns into rows
tensor2d_transpose = tensor2d.t()  # transpose the 2D tensor
# print(tensor2d_transpose)
# print(tensor2d_transpose.shape) # should print torch.Size([2, 3])

# common way to multiply tensors
# element-wise multiplication
tensor2d_mul = tensor2d * tensor2d
# print(tensor2d_mul)
# matrix multiplication
tensor2d_matmul = torch.matmul(tensor2d, tensor2d.t())
# print(tensor2d_matmul)


# taking dot product of two tensors
# view -1 flattens the tensor into a 1D tensor with all elements
# then torch.dot computes the dot product of the two 1D tensors
tensor2d_dot = torch.dot(tensor2d.view(-1), tensor2d.view(-1))


# quick trick to distinguish tensor types
# 0D Scalar: 0 opening brackets
t0 = torch.tensor(5)

# 1D Vector: 1 opening bracket
t1 = torch.tensor([1, 2])

# 2D Matrix: 2 opening brackets
t2 = torch.tensor([[1, 2, 2, 5]])

# 3D Tensor (e.g., Batch=2, Rows=1, Cols=3): 3 opening brackets
t3 = torch.tensor([[[1, 2, 3]], [[4, 5, 6]]])

# 4D Tensor (e.g., Image batch: Batch=1, Channels=3, Height=2, Width=2): 4 opening brackets
t4 = torch.zeros((1, 3, 2, 2))
"""
([
    [
        [
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.]
        ]
    ]
])
"""
