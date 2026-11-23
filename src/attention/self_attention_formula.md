
1. THE CODE LINE
------------------------------------------------------------------------
attn_weights = torch.softmax(attn_scores / (keys.shape[1] ** 0.5), dim=-1)

2. THE MATHEMATICAL FORMULA
------------------------------------------------------------------------
Compact Form: Attention Weights = softmax( (Q @ K^T) / sqrt(d_k) )

3. PIECE-BY-PIECE MAPPING
------------------------------------------------------------------------
* attn_scores  -->  Q @ K^T
  Raw dot-product similarity scores (query @ keys.T).

* keys.shape[1]  -->  d_k
  The dimension (number of columns) of each Key vector (e.g., 2).

* ** 0.5  -->  sqrt(...)
  Python shortcut for taking the square root (raising to the 1/2 power).

* keys.shape[1] ** 0.5 -->  / sqrt(d_k)
  Scaling step: Divides raw scores by sqrt(d_k) so variance stays at 1
  and gradients do not vanish during training.

* torch.softmax(..., dim=-1)  -->  e^(z_i) / sum_j(e^(z_j))
  Normalization step: Exponentiates every score (making them positive)
  and divides by the row sum (dim=-1) so each row sums to 1.0 (100%).

* attn_weights  -->  A
  Final percentage weights ready to blend the values matrix
  (attn_weights @ values).


4. QUICK TAKEAWAYS
------------------------------------------------------------------------
* x ** 0.5 = sqrt(x): Raising a number to the 0.5 (1/2) power takes its
  square root without needing 'import math' or converting an int to a
  PyTorch tensor.

Mathematically, raising a number to the 0.5 power equals its square root because of the exponent multiplication 
rule (x^a)^b = x^{a * b}.

Match the definition of a square root:
	• By definition, the square root of x written sqrt{x} or math.sqrt(x) is the positive number that gives you x when you square it: sqrt{x}^2 = x.

	• Since x^{0.5}^2 also gives you x, x ** 0.5 and math.sqrt(x) must be the exact same value.
    
(Note: This pattern extends to all roots—for example, x ** (1/3) is the cube root sqrt[3]{x}, and x ** 0.25 is the fourth root sqrt[4]{x}).

* Why Scale by sqrt(d_k)? As vector size d_k grows, dot products spread
  out by sqrt(d_k). Dividing by sqrt(d_k) keeps scores in a balanced
  range so softmax produces smooth, learnable weights instead of
  extreme outputs like [1.0, 0.0, 0.0].