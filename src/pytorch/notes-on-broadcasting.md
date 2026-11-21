Broadcasting is a feature in PyTorch (and NumPy) that automatically expands a smaller tensor to match the shape of a larger tensor during math operations without copying data in memory.

Normally, you can only add two matrices if they have the exact same shape (for example, (2, 30) + (2, 30)). Broadcasting lets you add a 1D list of 30 numbers directly to a 2D table of 2 rows and 30 columns.

How Broadcasting Works Here

Suppose after multiplying x @ W^T, you have a (2, 30) matrix, and you want to add your bias vector b of shape (30,):

- Step 1 (Align Dimensions): PyTorch compares the shapes from right to left. It sees (2, 30) and (30,) and treats the bias vector as (1, 30) (1 row, 30 columns).

- Step 2 (Virtual Stretching): Because the bias has size 1 in the row dimension while your matrix has size 2, PyTorch virtually "stretches" that single row of 30 biases across both rows of your data.

- Step 3 (Element-by-Element Addition): PyTorch adds b to Row 0, and then adds the exact same b to Row 1.


Simple Visual Example (Using 3 Columns Instead of 30)

Imagine your matrix multiplication gave you a (2, 3) output, and your bias vector has 3 numbers:

```text
[10  20  30]   [1  2  3]
[40  50  60] +
```

Broadcasting automatically behaves as if the bias vector were duplicated into two rows:

```text
[10  20  30]   [1  2  3]   [11  22  33]
[40  50  60] + [1  2  3] = [41  52  63]
```
