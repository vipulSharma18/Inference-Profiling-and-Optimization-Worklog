# Inference Profiling and Optimization Guide
> Note: Project is in the ideation stage, with the creation of problems in progress.

Explore commonly faced profiling scenarios and develop an intuition for how to break down profiling problems and find the bottleneck.

## Profilers used:
1. Torch Profiler
2. NVIDIA Nsight Systems
3. NVIDIA Nsight Compute

## Common bottlenecks:
1. Coarse-Grained Bottlenecks: The GPU (Streaming Multiprocessors) getting "stalled" (spending time in idle stage) while waiting for its dependencies:      
    a. CPU/Host side operations required by a GPU kernel.       
    b. Input/Output from/to the GPU global memory.      
    c. Network data movement (CPU to GPU buffer copies) in case of multi-GPU computation.       
    e. Launch overhead for each GPU kernel.     

2. Fine-Grained Bottlenecks: Latency caused at the instruction level in the GPU kernel:     
    a. Use of atomic instructions to prevent contention at the same memory location.        
    b. Inter-SM communication via the global memory.        
    c. Quantization at the level of warps/wave due to warp divergence, or at the level of data tiles due to partially filled tiles. -> Causes low goodput from the SMs.     
    d. Shared memory bank conflicts: Different data being requested from the same memory bank location.     
    e. Register spilling to the local memory.       
    f. L2 cache misses.     

## Principled approach to inference optimization:

Always go from top to bottom, or coarser profiling to fine-grained one, when optimizing inference workloads.

### Model Serving/User Request Batching Level:
* Batching of user requests as per the token budget and the type of operation (prefill v/s decode).
* Different dynamic batching algorithms.

### Multi-GPU:
* Types of parallelism within a model's forward pass given 1 input sample: EP, CP, SP, TP, PP.
* Types of distributed Attention: Ring Attention, Ulysses-Attention.
* Types of reductions across a network: Ring, Tree, Butterfly. 
* Types of network interfaces: P2P, InfiniBand, GPU-RDMA, NVLink Switch (NVLS SHARP), PCIe.
* Types of communication kernels: NCCL, NVSHMEM.

### Single GPU:
* Reducing CPU launch overhead by CUDA graphs and persistent kernels.
* Overlapping CPU work with GPU kernels via async launching.
* Cache hits: Global memory v/s L2 cache.
* Coalesced memory accesses from global memory.
* Async memory load from the global memory to overlap memory load with compute.
* Zero-copy movement of data from CPU to GPU.

### Cluster of Cooperative Thread Arrays (Multi-SM):
* Communication latency across SMs. Avoid using Global memory and directly be able to access each other's shared memory.

### Cooperative Thread Array (SM):
* Shared memory bank conflicts and swizzling of data to avoid it.

### Warp Scheduler level:
* Warp specialization and the implicit management of the dependencies between warps by using warp scheduler.

### Warp level:
* Warp divergence, wave and tile quantization.

### Thread level:
* Usage/load on the CUDA cores v/s Tensor Cores v/s Special Function Units. Avoiding contention between such compute resources.
* Avoiding contention between memory resources like registers by using tensor cores and TMEM.

## Problems (Curation in progress):

**Problem 1**: TorchAO's `Float8WeightOnlyConfig` on RTX 4090 and RTX 5090 is much slower than the eager baseline.      

**Problem 2**: More graphs, more problems: Understanding the impact of CUDAGraphs on GPU memory usage by profiling.      

**Problem 3**: What's my autotune up to?: Understand optimizations that the autotuner does by using GemLite's Triton GEMM kernels for RTX 4090 as an example.        

## References/Resources to learn the background:
1. Nsight Systems Docs: https://docs.nvidia.com/nsight-systems/index.html
2. Nsight Compute Docs: https://docs.nvidia.com/nsight-compute/index.html
3. CUDA Developer Tools Tutorials Playlist: https://www.youtube.com/playlist?list=PL5B692fm6--ukF8S7ul5NmceZhXLRv_lR
4. (Note: NASA has good HPC tutorials) NASA HECC Nsight Systems: https://www.nas.nasa.gov/hecc/support/kb/performance-analysis-of-your-gpu-applications-with-nsight-systems_701.html
5. NASA HECC Nsight Compute: https://www.nas.nasa.gov/hecc/support/kb/performance-analysis-of-your-gpu-cuda-kernels-with-nsight-compute-cli_706.html#url
6. (ptrblck's short tutorial) https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59
7. (Disclaimer: Paid) Optimizing CUDA Machine Learning Codes With Nsight Profiling Tools: https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-AC-03+V2
8. (Disclaimer: Paid) Nsight Analysis System: Build Custom Python Analysis Scripts: https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-AC-13+V1
9. (Disclaimer: Paid) Find the Bottleneck: Optimize AI Pipelines With Nsight Systems: https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-AC-14+V1
10. Different levels at which workload can be memory bound and how to resolve it: https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md
11. Understanding the memory chard and memory hierarchy by using Ampere as an example: https://developer.nvidia.com/gtc/2020/video/s21819-vid
12. Detailed walkthrough of the memory profiling capabilities of NCU: https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32089/
13. Hopper specific features and tuning guide overview: https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51119/
14. (Overview of the basics of nsys and ncu) What the Profiler is Telling You: How to Get the Most Performance out of Your Hardware: https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s22141/
15. (Detailed walkthrough of the whole flow) Introduction to Kernel Performance Analysis with NVIDIA Nsight Compute: https://www.youtube.com/watch?v=fsC3QeZHM1U