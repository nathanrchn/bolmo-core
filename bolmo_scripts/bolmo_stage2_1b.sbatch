#!/bin/bash
#SBATCH --job-name=bolmo-s2-1b
#SBATCH --partition=normal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=6:00:00
#SBATCH --output=/capstor/scratch/cscs/nathanrchn/bolmo/bolmo-checkpoints/logs/stage2_1b_%j.out
#SBATCH --error=/capstor/scratch/cscs/nathanrchn/bolmo/bolmo-checkpoints/logs/stage2_1b_%j.out
#SBATCH --environment=verl_swiss
#SBATCH --account=a0101

set -euo pipefail

export SCRATCH=/capstor/scratch/cscs/nathanrchn
export HF_HOME=$SCRATCH/.cache/huggingface

export VENV=$SCRATCH/bolmo/bolmo-core/.venv
# Use container's python (has torch/CUDA compiled for GH200), add venv packages via PYTHONPATH
export PYTHON=/usr/bin/python3
export PYTHONPATH=$SCRATCH/bolmo/bolmo-core/src:$VENV/lib/python3.12/site-packages:${PYTHONPATH:-}

mkdir -p $SCRATCH/bolmo/bolmo-checkpoints/logs

export WANDB_API_KEY=<your-wandb-api-key>

# NCCL configuration for GH200
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=262144
export FI_CXI_DEFAULT_TX_SIZE=256

HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_ADDR=$HEAD_NODE
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 4))

echo "=== BOLMo Stage 2 - 1B (no compression) ==="
echo "NODES=$SLURM_NNODES  WORLD_SIZE=$WORLD_SIZE"

export NAME=stage2_bolmo_1b

srun --export=ALL bash -c '
export LOCAL_RANK=$((SLURM_LOCALID))
export RANK=$((SLURM_NODEID * 4 + SLURM_LOCALID))
export LOCAL_WORLD_SIZE=4
export OLMO_SHARED_FS=1
export PYTHONUNBUFFERED=1

NAME='"$NAME"' \
NUM_WORKERS=24 \
SEQUENCE_LENGTH=4096 \
DATA_SOURCE='"$SCRATCH"'/bolmo/bolmo-core/data_sources.txt \
OLMO_ARCH=olmo2_1B_v2 \
LOCAL_MODEL_STYLE="hnet:xlstm" \
ADD_HASH_EMBEDDINGS=false \
ADD_EXPANDED_EMBEDDINGS=true \
LR_SCHEDULE=linear_with_warmup \
STAGE1_CKPT_PATH='"$SCRATCH"'/bolmo/bolmo-checkpoints/stage1_bolmo_1b/step75000/model_and_optim \
GLOBAL_MODEL_LEARNING_RATE=2.6e-5 \
SAVE_FOLDER='"$SCRATCH"'/bolmo/bolmo-checkpoints/'"$NAME"' \
'"$PYTHON"' '"$SCRATCH"'/bolmo/bolmo-core/src/examples/bolmo/train_stage2.py '"$NAME"' \
    train_module.optim.lr=5.2e-5 \
    data_loader.seed=1234 \
    data_loader.global_batch_size=1572864 \
    train_module.rank_microbatch_size=49152 \
    train_module.bolmo_config.losses=[ce,boundary] \
    train_module.bolmo_config.loss_weights=[1,4] \
    train_module.bolmo_config.teacher_force_boundaries=false \
    train_module.bolmo_config.do_alm_debiasing=false \
    train_module.bolmo_config.merge_boundary_loss=false \
    train_module.optim.weight_decay=0.1 \
    train_module.optim.betas=[0.9,0.95] \
    train_module.max_grad_norm=0.5 \
    model.block.attention.use_flash=true \
    model.local_encoder.n_layers=1 \
    model.local_decoder.n_layers=4 \
    model.local_decoder.hnet_smooth=false \
    model.local_decoder.hnet_modulate=false \
    model.local_encoder.boundary_predictor_lookahead=1 \
    model.local_decoder.add_in_projection=true \
    model.local_decoder.add_norm_onto_residual=false \
    model.local_decoder.add_projected_patch_residuals=false \
    model.local_encoder.block_config.feed_forward.hidden_size=2816 \
    model.local_decoder.block_config.feed_forward.hidden_size=2816 \
    model.local_encoder.d_model=2048 \
    model.local_decoder.d_model=2048 \
    trainer.callbacks.checkpointer.ephemeral_save_interval=1000 \
    trainer.callbacks.checkpointer.save_interval=30000 \
    trainer.callbacks.downstream_evaluator.eval_interval=150000 \
    trainer.max_duration.value=150000 \
    trainer.callbacks.wandb.entity=nathanrchn \
    trainer.callbacks.wandb.project=bolmo
'
