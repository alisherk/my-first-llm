Chinchilla is a 70-billion-parameter large language model released by DeepMind in March 2022 (from the paper "Training Compute-Optimal Large Language Models").

• Why it's famous: Instead of building a giant, undertrained model like OpenAI's 175B GPT-3 or DeepMind's own 280B Gopher, researchers trained a smaller 70B model on 1.4 trillion tokens (4× more data) using the exact same compute budget.

• The breakthrough: Chinchilla beat those much larger models across benchmarks, proving that training data and model size must scale equally—which is where the name "Chinchilla scaling laws" comes from.


Chinchilla Scaling Laws Formulas
1. Equal Scaling Rule: N scales proportionally with D (if compute increases by 10x, scale parameters by ~3.16x and training tokens by ~3.16x).

2. Optimal Token-to-Parameter Ratio: D = 20 * N (roughly 20 training tokens per parameter).

3. Total Training Compute: C = 6 * N * D
   - C = Total training compute in FLOPs
   - N = Number of model parameters
   - D = Number of training tokens
   - 6 = 2 FLOPs per parameter (forward pass) + 4 FLOPs per parameter (backward pass)

Sample Calculation: 1B Parameter Model (N = 1,000,000,000 or 10^9)
1. Optimal Training Tokens (D):
   D = 20 * (1 * 10^9) = 20,000,000,000 tokens (20B tokens)

2. Total Training Compute (C):
   C = 6 * (1 * 10^9) * (20 * 10^9)
   C = 120 * 10^18 FLOPs
   C = 1.2 * 10^20 FLOPs (120 quintillion FLOPs)

3. Estimated Training Time on 1 NVIDIA H100 GPU:
   - Peak FP16 compute: ~1 PFLOP (10^15 FLOPs/sec)
   - Realistic utilization (45% MFU): 4.5 * 10^14 FLOPs/sec
   - Training time = (1.2 * 10^20 FLOPs) / (4.5 * 10^14 FLOPs/sec)
   - Training time = ~266,667 seconds = ~74 hours (~3.1 days)

Other Chinchilla-Optimal Reference Sizes
- 7B Model: 140B tokens | 5.9 * 10^21 FLOPs -> 5.9 quintillion
- 70B Model: 1.4T tokens | 5.9 * 10^23 FLOPs -> 590 quintillion
- 408B Model: ~8.2T tokens | 2.0 * 10^25 FLOPs -> 20,000 quintillion