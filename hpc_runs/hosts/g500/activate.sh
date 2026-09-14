#!/usr/bin/env bash
# Source this file on g500. Keep all generated data on the local scratch disk.
export INTRMOTIV_ROOT=/scratch/lin/IntrMotiv
export INTRMOTIV_SOURCE="$INTRMOTIV_ROOT/src/SF_hipposlam"
export INTRMOTIV_TRAIN_DIR="$INTRMOTIV_ROOT/train_dir"
export XDG_CACHE_HOME="$INTRMOTIV_ROOT/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$INTRMOTIV_ROOT/tools/python"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export WANDB_DIR="$INTRMOTIV_ROOT/logs/wandb"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_DATA_DIR="$INTRMOTIV_ROOT/artifacts/wandb"
export TMPDIR="$INTRMOTIV_ROOT/tmp"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$INTRMOTIV_ROOT/tools/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
source "$INTRMOTIV_ROOT/envs/SF_git/bin/activate"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
# Sample Factory's default train_dir and DMLab cache are relative to cwd.
cd "$INTRMOTIV_ROOT" || return 1
export PYTHONPATH="$INTRMOTIV_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
