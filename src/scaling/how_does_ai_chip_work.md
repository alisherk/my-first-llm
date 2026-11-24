# HOW DOES AN AI CHIP WORK?
An H100 doesn't calculate Row 0, then move to Row 1, then move to Row 2. Instead, it breaks the work across thousands of tiny parallel processors. 


1. All Batches & Heads Run at the Same Time (Streaming Multiprocessors)

Your tensor from attention/multi_head_attention.py has shape [batch=2, heads=2, 6, 6], which is 4 independent 6 × 6 grids.

• An H100 has 132 Streaming Multiprocessors (SMs) (independent mini-processors).
• It assigns (Batch 0, Head 0), (Batch 0, Head 1), (Batch 1, Head 0), and (Batch 1, Head 1) to different SMs so all 4 grids execute simultaneously.


2. All 6 Rows ("Your", "journey", "starts"...) Run at the Same Time (Tensor Cores)

Inside each SM, the H100 uses specialized hardware called Tensor Cores (528 total on an H100), which are physically wired to do matrix math (A @ B) in one clock cycle rather than one number at a time:

• Instead of computing 1.1111 * 0.4429 ("Your") and waiting before computing (0.4990 * 0.4429) + (0.6121 * 0.4864) ("journey"), a Tensor Core loads the entire ATTN_WEIGHTS tile and the entire VALUES tile into ultra-fast on-chip registers.

• Every row's dot product (0.4921, 0.5187, 0.3348, 0.4604, 0.3660, 0.3241) is calculated in parallel in the exact same hardware step.



SEQUENTIAL (CPU loop thinking):
  Step 1: Compute "Your"    -> 1.1111 * 0.4429                  = 0.4921
  Step 2: Compute "journey" -> 0.4990 * 0.4429 + 0.6121 * 0.4864 = 0.5187
  Step 3: Compute "starts"  -> 0.3238 * 0.4429 + 0.3935 * 0.4864 = 0.3348
  ... (waits one row at a time)


PARALLEL (NVIDIA H100 Tensor Core):
  +-------------------------------------------------------------------+

  |  [Batch 0, Head 0]  |  [Batch 0, Head 1]  |  [Batch 1, Head 0] ...|  <- Different SMs
  +-------------------------------------------------------------------+
            |
            v  (Inside one Tensor Core, ALL 6 rows fire in 1 burst)
  Row 0 ("Your")   : [1.1111, 0, ...]  @ [0.4429, 0.4864, ...] -> 0.4921  \
  Row 1 ("journey"): [0.4990, 0.61...] @ [0.4429, 0.4864, ...] -> 0.5187   | All 6 happen
  Row 2 ("starts") : [0.3238, 0.39...] @ [0.4429, 0.4864, ...] -> 0.3348   | at the EXACT
  Row 3 ("with")   : [0.2562, 0.29...] @ [0.4429, 0.4864, ...] -> 0.4604   | same time!
  Row 4 ("one")    : [0.2271, 0.21...] @ [0.4429, 0.4864, ...] -> 0.3660   |
  Row 5 ("step")   : [0.1608, 0.20...] @ [0.4429, 0.4864, ...] -> 0.3241  /


An SM (Streaming Multiprocessor) is the core building block—or "mini-CPU"—inside an NVIDIA GPU.
If you think of a standard computer CPU as having 8 to 16 big, complex cores, an NVIDIA GPU like the H100 is built out of 132 SMs, each packed with dozens of smaller, specialized math engines designed to run thousands of calculations at the exact same time.

What is inside a single H100 SM?

Each of the 132 SMs on an H100 contains its own local toolkit:

• 128 CUDA Cores: General-purpose math workers that handle basic arithmetic (addition, multiplication, Softmax exponentials, dropout masks).

• 4 Tensor Cores: Heavy-duty matrix engines physically wired to multiply whole tiles of numbers (A @ B) in a single clock cycle.

• Shared Memory (SRAM): A tiny, ultra-fast scratchpad right next to the cores so workers on the same SM can share intermediate numbers without waiting for slow main GPU memory (HBM).

• Warp Schedulers: Traffic controllers that group threads into teams of 32 (called warps) and keep all the cores fed with zero idle time.
