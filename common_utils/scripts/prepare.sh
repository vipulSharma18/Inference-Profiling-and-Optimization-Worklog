# reference: https://github.com/pytorch/ao/blob/main/scripts/prepare.sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
python scripts/download.py --repo_id unsloth/Meta-Llama-3.1-8B
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/unsloth/Meta-Llama-3.1-8B
