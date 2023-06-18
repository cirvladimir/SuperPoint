docker build . -f .devcontainer/gpu.Dockerfile -t superpoint
docker run -d --rm --gpus all -v $(pwd):/workspaces/SuperPoint superpoint /bin/bash -c "cd /workspaces/SuperPoint && .//train.sh"
