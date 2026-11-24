# why do we use softmax in attention mechanisms

Softmax is used in attention mechanisms to convert raw attention scores into a probability distribution. This ensures that the attention weights for each token sum to 1, allowing the model to focus on the most relevant tokens while maintaining a normalized weighting scheme. By applying softmax, the model can effectively weigh the importance of different tokens when computing context vectors.


1. Why raw scores fail without Softmax

Suppose token "journey" has raw attention scores [0.766, 0.953, 0.907, 0.606, -0.162, 1.074] across 6 words. If you multiplied those raw numbers directly against the values vectors:

• Scale explosion: The sum of those scores is 4.14 (414%), which would artificially blow up the size of "journey"'s vector every layer—especially as sentences grow from 6 words to 1,000 words.

• Negative flipping: A negative score like -0.162 would actually subtract or invert another word's meaning rather than simply ignoring it.


2. What Softmax fixes (Three Jobs at Once)

When you pass those scores through torch.softmax(..., dim=-1), it outputs [0.1369, 0.1841, 0.1846, 0.1605, 0.1754, 0.1585]:

• Creates a 100% budget (sum = 1.0): Every word gets a slice of a fixed 100% pie (13.7% + 18.4% + 18.5% + 16.1% + 17.5% + 15.9% = 100%). No matter whether a sentence has 6 words or 6,000 words, the final blended vector stays on a stable scale.

• Eliminates negative numbers e^{x}: Because Softmax raises e to the power of each score e^score, even a negative score like -0.162 becomes a small positive weight (0.1754)—meaning "pay low attention", not "flip the vector backward". (And masked future tokens set to -inf become e^{-infty} = 0.0000, completely silencing them.)

• Sharpens winners: Because e^{x} grows exponentially, higher scores get noticeably bigger slices of the pie than lower scores, helping the model focus on the most relevant words while still keeping a tiny bit of background context from the rest.


3. Why divide by d_k ** 0.5 right before Softmax?

When d_k (the vector size) is large, dot products naturally grow into huge numbers (like 50 vs 10). If you feed huge numbers into Softmax, e⁵⁰ is so overwhelmingly larger than e¹⁰ that one word gets 0.9999 (100%) and every other word gets 0.0000—freezing learning (vanishing gradients). Dividing by sqrt{d_{k}} keeps the scores in a healthy range so Softmax produces smooth, usable percentages.

# Mathematical Application Example 
Softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j from 1 to K)

• z_{i}: The raw score (logit) for item i.

• e^{z_{i}} (Numerator): Applies the exponential function (e ≈ 2.718) to score z_{i}. This forces every number to be strictly positive (even if z_{i} is negative) and amplifies larger scores.

• sum_{j=1}^{K} e^{z_j} (Denominator): Sums the exponential values of all K items in the vector. Dividing by this total normalizes the output so all probabilities add up to 1.0 (100%).

Suppose your raw scores are = [1.0, 2.0, 0.0]:

1. Exponentiate each score:
	• e^{1.0} \approx 2.72
	• e^{2.0} \approx 7.39
	• e^{0.0} = 1.00

2. Sum the exponentiated scores:
	• 2.72 + 7.39 + 1.00 = 11.11

3. Divide each exponentiated score by the sum to get Softmax probabilities:
	• 2.72 / 11.11 ≈ 0.245
	• 7.39 / 11.11 ≈ 0.665
	• 1.00 / 11.11 ≈ 0.090

4. Check these probabilities sum to 1: 0.245 + 0.665 + 0.090 = 1.000
