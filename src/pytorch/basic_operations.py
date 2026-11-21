""""
lets consider 2D basis matric B = [3 -1
                                   1  2]

Each column of B represents a basis vector in 2D space as show in basic_matrix.png

This matrix acts as a geomatric transformer and a coordinate translator in 2D space.

if you feed the corner points of a shape (like a unit square) into B 
it transforms the entire shape into a larger, tilted parallelogram:

• Rotation: It turns points counterclockwise around the origin.
• Shearing: Because the columns are not perpendicular(3(-1) + 1(2) = -1 != 0)), it slants 90-degree corners into acute and obtuse angles.


Multiplying a vector by this matrix applies the combined transformation described above: rotation, shearing, and scaling along the basis vectors.

suppose we pick v = [2, 3]. It can be calculated manually as:

B * v = [3 -1
         1  2] * [2
                    3] = [3*2 + (-1)*3
                           1*2 + 2*3] = [6 - 3
                                          2 + 6] = [3
                                                     8]
"""

import torch

B = torch.tensor([[3, -1],
                  [1,  2]])

v = torch.tensor([2, 3])

result = B @ v

# print(result) # result of B @ v, should be [3, 8]

"""
Please note there is a diff between matrix like this: 

B = torch.tensor([[3, -1],
                  [1,  2]])


and data_set like this:

data_set = torch.tensor([[3, -1],
                         [1,  2]])

1. Datasets (data_set): Rows = Observations, Columns = Features
   In machine learning, datasets are organized like spreadsheets:
• Each row is a separate observation (a sample or data point).
• Each column is a feature (the i or j) coordinate.


2. Transformation Matrices (B): Columns = Basis Vectors
   In linear algebra, a matrix like B isn't a list of data points—it's a transformation rule that warps space.
• First column [3, 1]: Where the original i basis vector [1, 0] lands.
• Second column [-1, 2]: Where the original j basis vector [0, 1] lands.


"""