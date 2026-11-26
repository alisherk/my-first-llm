""" "
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

B = torch.tensor([[3, -1], [1, 2]])
v = torch.tensor([2, 3])
# row 0: 3*2 + (-1)*3 = 6 - 3 = 3 
# row 1: 1*2 + 2*3 = 2 + 6 = 8
result = B @ v # result should be [3, 8] which is the correct output
# you can review this transformation in transforming_vector_v.jpg


"""
Please note there is a diff between matrix like this: 

B = torch.tensor([[3, -1],
                  [1,  2]])

2. Transformation Matrices (B): Columns = Basis Vectors
   In linear algebra, a matrix like B isn't a list of data points—it's a transformation rule that warps space.
• First column [3, 1]: Where the original i basis vector [1, 0] lands.
• Second column [-1, 2]: Where the original j basis vector [0, 1] lands.


data_set = torch.tensor([[3, -1],
                         [1,  2]])

1. Datasets (data_set): Rows = Observations, Columns = Features
   In machine learning, datasets are organized like spreadsheets:
• Each row is a separate observation (a sample or data point).
• Each column is a feature (the i or j) coordinate.

"""
a = torch.tensor([[6, 1], [3, 1]])
# a shape torch.Size([2, 2])
b = torch.tensor([[2], [3]])
# b shape torch.Size([2, 1])

# Matrix multiplication using @ operator for similarity of 2D linear transformations or transorm, rotate, scale and shear vectors in space
c_dot_product_wise = a @ b
# row 0 of c_dot_product_wise: 6*2 + 1*3 = 12 + 3 = 15
# row 1 of c_dot_product_wise: 3*2 + 1*3 = 6 + 3 = 9.       
# c_dot_product_wise = [[15],[9]]

# matrix multiplication using * operator would broadcast b across the rows of a. use it to move transormed coordinates elementwise
c_elementwise = a * b
# row 0 of c_elementwise: 6*2, 1*3 = 12, 3
# row 1 of c_elementwise: 3*2, 1*3 = 6, 3

# c_elementwise = [[12, 3],
#                  [ 6, 3]]

"""

Why @ Transforms Space While * Cannot

Think of a point (x, y) in 2D space:

• With *, x only ever multiplies with x, and y only ever multiplies with y. You can stretch or shrink the grid horizontally or vertically, but you can never rotate it because x and y never mix.

• With @, the new x is a weighted mix of both the old x and old y (a*x + b*y). Mixing features across dimensions is what allows neural networks to rotate, bend, and reshape data into new representations.

You can review (@-vs-*).jpg for a visual explanation of the difference between matrix multiplication and element-wise multiplication.

+-----------------------------------+-----+---------------------------------------------+------------------------------------------+

| GOAL / SCENARIO                   | USE | WHY IT WORKS HERE                           | COMMON AI / PYTORCH EXAMPLE              |
+-----------------------------------+-----+---------------------------------------------+------------------------------------------+

| Measure Similarity                |  @  | Multiplies matching features & sums them    | scores = queries @ keys.T                |
|                                   |     | into a single alignment score.              | (Transformer Self-Attention)             |
+-----------------------------------+-----+---------------------------------------------+------------------------------------------+

| Project / Change Dimensions       |  @  | Mixes all input features together to map    | out = x @ weights.T                      |
|                                   |     | data into a new shape or feature space.     | (Inside nn.Linear layers)                |
+-----------------------------------+-----+---------------------------------------------+------------------------------------------+
| Compute Weighted Averages         |  @  | Multiplies values by weights and sums       | context = attention_weights @ values     |
|                                   |     | them up in a single matrix step.            | (Attention pooling)                      |
+-----------------------------------+-----+---------------------------------------------+------------------------------------------+

| Mask or Mute Data                 |  *  | Multiplies unwanted positions by 0 and      | masked = tokens * attention_mask         |
|                                   |     | kept positions by 1 without mixing neighbors| (Padding / Dropout)                      |
+-----------------------------------+-----+---------------------------------------------+------------------------------------------+

| Control Information Flow (Gating) |  *  | Acts like a dimmer switch (0.0 to 1.0)      | hidden = gate * candidate                |
|                                   |     | for each individual neuron in place.        | (LSTMs, GRUs, SwiGLU)                    |
+-----------------------------------+-----+---------------------------------------------+------------------------------------------+

| Scale Brightness or Magnitude     |  *  | Stretches or shrinks values along existing  | darker_img = img * 0.5                   |
|                                   |     | axes without changing vector direction.     | (Image preprocessing / normalization)    |
+-----------------------------------+-----+---------------------------------------------+------------------------------------------+

QUICK DECISION RULE:
  Do features/dimensions need to mix and add together?
    [ YES ] ---> Use @  (Blends rows & columns)
    [ NO  ] ---> Use *  (Scales or masks each element in its own lane)
"""
