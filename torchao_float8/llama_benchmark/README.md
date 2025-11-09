```
# reference: https://github.com/pytorch/ao/blob/main/torchao/_models/llama/benchmarks.sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
```

# Setup:
```
export CHECKPOINT_PATH=/root/checkpoints
export MODEL_REPO=unsloth/Meta-Llama-3.1-8B
export MODEL_REPO=unsloth/Llama-3.2-3B
```

# Baseline with torch profile
```
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt --memory_profile torch_memory_profiler

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt --profile torch_execution_profiler

nsys python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --write_result benchmark_results.txt
```

# fp8 dynamic quantization for activations and weights: tensor-wise scaling
```
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization float8dq-tensor --write_result benchmark_results.txt --memory_profile torch_memory_profiler

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization float8dq-tensor --write_result benchmark_results.txt --profile torch_execution_profiler

nsys python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization float8dq-tensor --write_result benchmark_results.txt
```

# fp8 weights only -> massive slowdown in speed and increase in peak memory usage
```
python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization float8wo --write_result benchmark_results.txt --memory_profile torch_memory_profiler

python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization float8wo --write_result benchmark_results.txt --profile torch_execution_profiler

nsys python generate.py --checkpoint_path $CHECKPOINT_PATH/$MODEL_REPO/model.pth --compile --quantization float8wo --write_result benchmark_results.txt
```