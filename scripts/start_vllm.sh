#!/bin/bash
# Require install docker and nvidia-docker2 as in README.md
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.8.0/install-guide.html

MODEL_PATH=Qwen/Qwen2.5-Coder-0.5B-Instruct
CUDA_VISIBLE_DEVICES=2
PORT=8080
echo "PORT=$PORT"
echo "MODEL_PATH=$MODEL_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

sudo docker run \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p $PORT:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.6.4 \
    --host 0.0.0.0 \
    --model $MODEL_PATH


