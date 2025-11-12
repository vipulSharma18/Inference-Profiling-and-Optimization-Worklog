# Worklog for TorchAO's Float8WeightOnlyConfig Inference Slowdown Profiling
## The Problem:
TorchAO's Float8WeightOnlyConfig has a massive **decrease in inference throughput**, **low memory bandwidth utilization**, and **high peak VRAM** usage.          

TorchAO issue raised after I discovered this problem while working on a [survey of quantization formats](https://github.com/vipulSharma18/Survey-of-Quantization-Formats?tab=readme-ov-file#benchmarking-results): [TorchAO GH Issue](https://github.com/pytorch/ao/issues/3288).            

_Example:_ Meta-Llama-3.1-8B inference.

Model is 15.01GB in size in bf16. Computation in mixed precision with torch.bfloat16.      
GPU is RTX5090 with 34GB VRAM and peak theoretical memory bandwidth of 1792 GB/sec.       
Batch size is 1, so possibly a GEMV kernel is being used and the inference is memory bound.       
We follow GPT-Fast and only compile the decode stage of the inference.      

**No quantization**:        
tok/s=104.68, tok/s_decode=105.73, ttft=0.0186, mem/s=1571.23 GB/s, peak_mem=16.30 GB, model_size=15.01 GB         

**float8dq-tensor quantization**:       
tok/s=160.55, tok/s_decode=169.80, ttft=0.0675, mem/s=1204.97 GB/s, peak_mem= 9.21 GB, model_size= 7.51 GB           

**float8wo quantization**:      
**tok/s=7.44**, tok/s_decode=7.48, ttft=0.1442, **mem/s=55.91 GB/s**, **peak_mem=26.00 GB**, model_size= 7.51 GB         

**The 3 performance limitations:**
* Decrease in inference throughput: 7.44 tps v/s 105.73 tps of baseline.
* Low memory bandwidth utilization: 55.91 GB/s v/s peak theoretical 1792 GB/s.
* Peak VRAM usage: 26.00 GB v/s 16.30 of the baseline.

## The initial PyTorch Trace:
As a first, and possibly only step, we use the GPT-Fast benchmark provided by TorchAO to profile the memory and execution of the above 3 settings.

## Torch Memory Profile

> **TLDR**: The dequantization of weights in FP8WeightsOnly config is not fused with GEMV computations. This leads to spike in GPU VRAM usage.

The 0-th inference iteration is profiled using a CUDA memory snapshot. The snapshots are available at the following paths: `llama_benchmark/Meta-Llama-3.1-8B_None_torch_memory_profiler.pickle`, `llama_benchmark/Meta-Llama-3.1-8B_float8dq-tensor_torch_memory_profiler.pickle`, `llama_benchmark/Meta-Llama-3.1-8B_float8wo_torch_memory_profiler.pickle`.

<div align="center">
  <img src="figures/none_whole_timeline.png?v=2" alt="Baseline Whole Timeline" width="800">
  <p><strong>Figure 1:</strong> Baseline Whole Timeline</p>
</div>
<div align="center">
  <img src="figures/float8dq_whole_timeline.png?v=2" alt="FP8 Weights and Activations DQ Whole Timeline" width="800">
  <p><strong>Figure 2:</strong> FP8 Weights and Activations DQ Whole Timeline</p>
</div>
<div align="center">
  <img src="figures/float8wo_whole_timeline.png?v=2" alt="FP8 Weights Only Static Quantization Whole Timeline" width="800">
  <p><strong>Figure 3:</strong> FP8 Weights Only Static Quantization Whole Timeline</p>
</div>

> **Initial observation**: On first look, comparing the whole timelines in Figures 1, 2, and 3, we can notice that they all have some blocks of memory in the middle of the timeline (encircled). Also, Float8WO has multiple spikes in memory that are not present in the other two snapshots (marked by arrow).

For diving deeper, we can zoom-in on the blocks and the regions around them, and view their call stack. This reveals that most of the quantization, compilation, and inference activity actually happens in the narrow slice of memory at the top. Further, we can see different phases of inference in the memory timeline. For example, all the encircled rectangular blocks of memory have a call stack related to Torch Dynamo, Inductor, and CUDA Graph Trees, indicating that this memory was active during the compilation of the decode function.

Revisiting the GPT-Fast code in generate.py, we can see there are different phases of inference:
1. Quantization, except for baseline. Model's linear layers are replaced with quantized affine layers in subsequent steps.
2. Compilation of decode's forward call. Torch Dynamo wraps the forward in compile wrapper.
3.1. Dummy inference pass: Prefill with quantized model.
3.2. Dummy inference pass: Decode with frame evaluation by Torch Dynamo, compilation and lowering by Torch Inductor, and subsequent graph recording into a CUDAGraph.
4.1. Real inference passes: Prefill like 3.1.
4.2. Real inference passes: Decode with quantized model and replay of CUDAGraph Trees.

Note: Both decode are also followed by cloning of outputs for optional post-processing which takes up minor amount of memory.

When we try to map these phases to the memory timeline from PyTorch, we can see presence of 3.1, 3.2, 4.1, and 4.2. Figure 4 shows the phases by using the float8 weights only config as an example. The bug of excessive memory usage during inference actually helps us to see the phases more distinctly with the weights only config.

> **Second Observation**: We can see that the spikes in memory happen for 3.1, 3.2, and 4.1, but not for 4.2. 4.2 is decode with CUDAGraph being used, which forces the same memory locations to be used for graph replay. This must have avoided the memory spikes for 4.2. We can thus conclude that in any future inference pass, prefill-stage of inference will the one responsible for CUDA OOM bugs, or at least spikes in the memory.

<div align="center">
  <img src="figures/float8wo_marked_with_phases.png" alt="FP8 Weights Only Static Quantization's Timeline Annotated with Phases of Inference" width="800">
  <p><strong>Figure 4:</strong> FP8 Weights Only Static Quantization's Timeline Annotated with Phases of Inference</p>
</div>

> **Third Observation**: Looking at the call stack of the spikes in memory, the spikes can be attributed to the dequantization of the float8 weight tensors during the forward pass in TorchAO. In some cases, the spikes can even be a GB in size. Comparing this with our other Float8 config which uses static weights quantization and dynamic activations quantization, the spikes are absent. There are miniscule memory increases during the dynamic quantization of the activation tensors, but there aren't any memory allocations visible for dequantization of the weights.    

<div align="center">
  <img src="figures/float8wo_dequantize_callstack.png" alt="Call Stack of the Spikes in Memory for FP8 Weights Only Static Quantization" width="800">
  <p><strong>Figure 5:</strong> Call Stack of the Spikes in Memory for FP8 Weights Only Static Quantization</p>
</div>

> **Takeaway**: The FP8 Weights Only Config is not doing dequantization of weights tensor properly. Dequantization of the weights and the computation of product of weights and activations should be fused in the GEMV kernel instead of being done separately to avoid such memory spikes in the GPU VRAM.

## Torch Execution Trace
### Baseline

### FP8 Static Weights and Dynamic Activations Quantization

### FP8 Static Weights Only Quantization

If needed, we can proceed to gather more information via NSYS and NCU profiling (to be decided).