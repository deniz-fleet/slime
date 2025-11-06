#!/bin/bash
set -e

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi

# Pull SLIME image
docker pull slimerl/slime:latest

# Setup SSH access (add your public key)
mkdir -p /root/.ssh
cat >> /root/.ssh/authorized_keys <<'EOF'
-----BEGIN OPENSSH PRIVATE KEY-----

b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW

QyNTUxOQAAACDgh4DAVaWUciglfkpFkN0ySGYQjKTfYBO6b2/68ablhQAAAJgX+jtjF/o7

YwAAAAtzc2gtZWQyNTUxOQAAACDgh4DAVaWUciglfkpFkN0ySGYQjKTfYBO6b2/68ablhQ

AAAEAyo86s7oFEp6tlVMtRv3MNxgQwQtFOkK0mMe226M92buCHgMBVpZRyKCV+SkWQ3TJI

ZhCMpN9gE7pvb/rxpuWFAAAADmRlbml6QGZsZWV0LnNvAQIDBAUGBw==

-----END OPENSSH PRIVATE KEY-----
EOF

# Clone your repo to have it ready
mkdir -p /workspace
cd /workspace
git clone https://github.com/deniz-fleet/slime.git

# Start SLIME container with persistent name
docker run -d \
  --name slime-dev \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  -v /workspace:/workspace \
  --restart unless-stopped \
  slimerl/slime:latest \
  tail -f /dev/null

echo "Setup complete. SLIME container running as 'slime-dev'" > /root/setup.log