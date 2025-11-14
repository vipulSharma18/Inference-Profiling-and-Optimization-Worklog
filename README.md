# Inference Profiling and Optimization Worklog
A collection of inference profiling problems that I've discovered during my work. This repo contains worklogs for each of those problems and is meant as a guide to help develop intuition for systematically analyzing profiling problems and finding the root cause of slowdown.

Each problem has a docker container for reproducibility of the environment and the inference slowdown.

        
## Problems:

**Problem 1 - TorchAO Float8** [![Worklog](https://img.shields.io/badge/Worklog-76eec6)](./torchao_float8/README.md) [![Build](https://github.com/vipulSharma18/Inference-Profiling-and-Optimization-Guide/actions/workflows/build-torchao-float8.yml/badge.svg)](https://github.com/vipulSharma18/Inference-Profiling-and-Optimization-Guide/actions/workflows/build-torchao-float8.yml) **(In Progress)**: TorchAO's `Float8WeightOnlyConfig` is much slower than the eager baseline, and has much higher peak VRAM usage than eager. This was discovered during my work on a [survey of quantization formats](https://github.com/vipulSharma18/Survey-of-Quantization-Formats) (found on TorchAO-0.13.0 and resolved on versions 0.14.1 and above. Tested on RTX4090 and RTX5090).      

**Problem 2 - GemLite Autotune** [![Build](https://github.com/vipulSharma18/Inference-Profiling-and-Optimization-Guide/actions/workflows/build-gemlite-autotune.yml/badge.svg)](https://github.com/vipulSharma18/Inference-Profiling-and-Optimization-Guide/actions/workflows/build-gemlite-autotune.yml) **(Not Started)**: Understand optimizations that the autotuner does by using GemLite's Triton GEMM kernels for RTX 4090 as an example. An opportunity to use nsys and ncu trace difference functionalities, and to recognize different routes for kernel optimization that the autotuner might pick given different inputs and kernel parameters.        

**Problem 3 - TorchInductor CUDA Graph Memory** [![Build](https://github.com/vipulSharma18/Inference-Profiling-and-Optimization-Guide/actions/workflows/build-torchinductor-cudagraph.yml/badge.svg)](https://github.com/vipulSharma18/Inference-Profiling-and-Optimization-Guide/actions/workflows/build-torchinductor-cudagraph.yml) **(Not Started)**: CUDAGraphs can lead to high GPU memory usage even without input duplication (for shape alignment) when used with quantized inference. This problem aims to understand the impact of CUDAGraphs on GPU memory usage and execution performance.      

> Note: Details of the docker container and environment setup can be found in DOCKER_README.md.

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

Always go from top to bottom, or coarser profiling to fine-grained one, when optimizing inference workloads. Below is a list of typical bottlenecks and tricks that arise in inference optimization.

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

## References/Resources to learn the background:
1. Nsight Systems Docs: https://docs.nvidia.com/nsight-systems/index.html
2. Nsight Compute Docs: https://docs.nvidia.com/nsight-compute/index.html
3. (Detailed walkthrough of the whole flow) Introduction to Kernel Performance Analysis with NVIDIA Nsight Compute: https://www.youtube.com/watch?v=fsC3QeZHM1U
