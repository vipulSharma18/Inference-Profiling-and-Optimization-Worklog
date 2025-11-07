# reference: https://github.com/pytorch/ao/blob/main/scripts/prepare.sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
python scripts/download.py --repo_id unsloth/Meta-Llama-3.1-8B || {
    echo "ERROR: Failed to download model from unsloth/Meta-Llama-3.1-8B"
    echo "Continuing with next steps..."
}

python scripts/convert_hf_checkpoint.py --checkpoint_dir ~/checkpoints/unsloth/Meta-Llama-3.1-8B || {
    echo "ERROR: Failed to convert checkpoint at checkpoints/unsloth/Meta-Llama-3.1-8B"
    echo "Continuing with next steps..."
}

echo "[entrypoint] running sshd checks"

echo "[entrypoint] ensuring /var/run/sshd and /etc/ssh/ exists"
mkdir -p /var/run/sshd
mkdir -p /etc/ssh/

echo "[entrypoint] ensuring host keys exist in /etc/ssh:"
ls -l /etc/ssh/ || echo "[entrypoint] /etc/ssh/ does not exist, i.e., no host keys found"

echo "[entrypoint] generating host keys if any are missing..."
ssh-keygen -A || echo "[entrypoint] ERROR: ssh-keygen failed"

echo "[entrypoint] new keys in /etc/ssh:"
ls -l /etc/ssh/ || echo "[entrypoint] /etc/ssh/ does not exist, i.e., no host keys found"

echo "[entrypoint] Starting ssh service after generating keys..."
/usr/sbin/sshd || echo "[entrypoint] ERROR: failed to start ssh service"

echo "[entrypoint] Ensuring existence and listing contents of ~/.ssh/:"
mkdir -p ~/.ssh/
chmod 700 ~/.ssh
echo "[entrypoint] manually adding public ssh keys in the ssh_authkeys file as a failcheck"
cat /workspace/common_utils/scripts/ssh_publickeys >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
ls -l ~/.ssh/ || echo "[entrypoint] ~/.ssh/ does not exist, i.e., no user auth keys found"

echo "[entrypoint] entrypoint sshd checks complete"
