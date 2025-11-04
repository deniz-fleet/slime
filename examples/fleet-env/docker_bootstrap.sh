#!/usr/bin/env bash

set -euo pipefail

# Minimal bootstrap: install Docker on Linux if needed, pull image, run container with FLEET_API_KEY

IMAGE=${IMAGE:-slimerl/slime:v0.5.0rc0-cu126}
CONTAINER_NAME=${CONTAINER_NAME:-slime-fleet-$(date +%s)}

if [ -z "${FLEET_API_KEY:-}" ]; then
  echo "FLEET_API_KEY must be set in your environment (export FLEET_API_KEY=...)" >&2
  exit 1
fi

install_docker_linux() {
  if command -v docker >/dev/null 2>&1; then
    return 0
  fi
  echo "Installing Docker (Linux)..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER" || true
}

ensure_docker() {
  if command -v docker >/dev/null 2>&1; then
    return 0
  fi
  case "$(uname -s)" in
    Linux)
      install_docker_linux
      ;;
    Darwin)
      echo "Please install Docker Desktop for macOS from https://www.docker.com/products/docker-desktop/" >&2
      exit 1
      ;;
    *)
      echo "Unsupported OS $(uname -s). Please install Docker manually." >&2
      exit 1
      ;;
  esac
}

ensure_docker

echo "Pulling image: ${IMAGE}"
docker pull "${IMAGE}"

# GPU args if NVIDIA present
GPU_ARGS=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS="--gpus all"
fi

WORKDIR_MOUNT="-v $(pwd):/workspace"

echo "Starting container: ${CONTAINER_NAME}"
docker run -d --rm ${GPU_ARGS} --privileged --ipc=host --shm-size=64g \
  --tmpfs /tmp:exec,size=32g \
  ${WORKDIR_MOUNT} \
  -e FLEET_API_KEY="${FLEET_API_KEY}" \
  --name "${CONTAINER_NAME}" \
  "${IMAGE}" bash -lc 'sleep infinity'

echo "Container running: ${CONTAINER_NAME}"
echo
echo "Next step (inside container):"
echo "  docker exec -it ${CONTAINER_NAME} bash -lc 'bash examples/fleet-env/run_fleet_qwen3_vl_8b_rl.sh'"


