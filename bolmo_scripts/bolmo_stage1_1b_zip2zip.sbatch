#!/bin/bash
#SBATCH --job-name=bolmo-s1-1b-z2z
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/nathanrchn/bolmo/bolmo-checkpoints/logs/stage1_1b_zip2zip_%j.out
#SBATCH --error=/capstor/scratch/cscs/nathanrchn/bolmo/bolmo-checkpoints/logs/stage1_1b_zip2zip_%j.out
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

echo "=== BOLMo Stage 1 - 1B (zip2zip compression) ==="
echo "NODES=$SLURM_NNODES  WORLD_SIZE=$WORLD_SIZE"

export NAME=stage1_bolmo_1b_zip2zip

srun --export=ALL bash -c '
export LOCAL_RANK=$((SLURM_LOCALID))
export RANK=$((SLURM_NODEID * 4 + SLURM_LOCALID))
export LOCAL_WORLD_SIZE=4
export OLMO_SHARED_FS=1
export PYTHONUNBUFFERED=1

NAME='"$NAME"' \
SEQUENCE_LENGTH=4096 \
DTYPE=float32 \
DATA_SOURCE='"$SCRATCH"'/bolmo/bolmo-core/data_sources.txt \
OLMO_ARCH=olmo2_1B_v2 \
OLMO_CKPT_PATH='"$SCRATCH"'/bolmo/bolmo-ckpt/olmo2-1b-distcp \
TRAIN_MODE=stage_1 \
LOCAL_MODEL_STYLE="hnet:xlstm" \
ADD_HASH_EMBEDDINGS=false \
ADD_EXPANDED_EMBEDDINGS=true \
EMBEDDING_INIT_PATH="" \
SAVE_FOLDER='"$SCRATCH"'/bolmo/bolmo-checkpoints/'"$NAME"' \
COMPRESSION_ENABLED=true \
COMPRESSION_MAX_CODEBOOK_SIZE=4096 \
COMPRESSION_MAX_SUBTOKENS=4 \
COMPRESSION_VOCAB_SIZE=100352 \
'"$PYTHON"' '"$SCRATCH"'/bolmo/bolmo-core/src/examples/bolmo/train_stage1.py '"$NAME"' \
    train_module.bolmo_config.losses=[local_encoder,ce,local_decoder,boundary] \
    train_module.bolmo_config.loss_weights=[1,1,1,4] \
    train_module.bolmo_config.div_fn=kl \
    train_module.bolmo_config.binarization_temp=5.0 \
    train_module.bolmo_config.use_oracle_patch_reps=true \
    train_module.bolmo_config.teacher_force_boundaries=true \
    train_module.bolmo_config.encoder_loss_lookahead=4 \
    train_module.bolmo_config.encoder_loss_no_lookahead_weight=0.0 \
    train_module.bolmo_config.encoder_loss_lookahead_weights=[0.0,0.0,0.0,4.0] \
    train_module.bolmo_config.do_alm_debiasing=true \
    train_module.bolmo_config.merge_boundary_loss=false \
    train_module.optim.weight_decay=0.1 \
    train_module.max_grad_norm=0.5 \
    train_module.optim.lr=7e-4 \
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
    data_loader.global_batch_size=786432 \
    train_module.rank_microbatch_size=49152 \
    trainer.callbacks.checkpointer.ephemeral_save_interval=1000 \
    trainer.callbacks.checkpointer.save_interval=75000 \
    trainer.callbacks.downstream_evaluator.eval_interval=75000 \
    trainer.max_duration.value=75000 \
    trainer.callbacks.wandb.entity=nathanrchn \
    trainer.callbacks.wandb.project=bolmo
'
