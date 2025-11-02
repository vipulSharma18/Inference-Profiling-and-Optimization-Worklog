# Inference-Profiling-Puzzles
> Note: Project is in the ideation stage, with the creation of puzzles in progress.

Explore commonly faced profiling scenarios and develop an intuition for how to break down profiling problems and find the bottleneck.

**Puzzle 1**: TorchAO's `Float8WeightOnlyConfig` on RTX 4090 and RTX 5090 is much slower than the eager baseline.      
**Puzzle 2**: More graphs, more problems: Understanding the impact of CUDAGraphs on GPU memory usage by profiling.      
**Puzzle 3**: What's my autotune up to?: Understand optimizations that the autotuner does by using GemLite's Triton GEMM kernels for RTX 4090 as an example.
