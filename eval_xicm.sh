#!/bin/bash
#SBATCH --job-name=xicm_eval
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/xzha593/projects/X-ICM/slurm_logs/xicm_eval_%j.log
#SBATCH --error=/data/xzha593/projects/X-ICM/slurm_logs/xicm_eval_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=80G

set -euo pipefail

mkdir -p /data/xzha593/projects/X-ICM/slurm_logs
mkdir -p /data/xzha593/tmp/xicm_rgb_cache

source /data/xzha593/miniconda3/etc/profile.d/conda.sh
conda activate /data/xzha593/envs/xicm

export XICM=/data/xzha593/projects/X-ICM
export COPPELIASIM_ROOT=/data/xzha593/software/CoppeliaSim
export COPPELIASIM_REAL=$(readlink -f "$COPPELIASIM_ROOT")

export LLVM7_LIB=/data/xzha593/software/llvm-private/usr/lib64
export SYSROOT_LIB=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64
export DRI_DIR=$SYSROOT_LIB/dri

export LD_LIBRARY_PATH=$LLVM7_LIB:$SYSROOT_LIB:$COPPELIASIM_REAL:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_REAL/platforms
export QT_PLUGIN_PATH=$COPPELIASIM_REAL
export QT_QPA_PLATFORM=xcb
export QT_XCB_GL_INTEGRATION=xcb_glx
export QT_OPENGL=desktop
unset EGL_PLATFORM

export LIBGL_DRIVERS_PATH=$DRI_DIR
export LIBGL_ALWAYS_SOFTWARE=1
unset LIBGL_ALWAYS_INDIRECT
export MESA_LOADER_DRIVER_OVERRIDE=swrast
export GALLIUM_DRIVER=llvmpipe

export PATH=$CONDA_PREFIX/bin:$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/bin:$PATH
export PYTHONPATH=$XICM:$XICM/YARR:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

export SAVE_RGB_DIR=0
export XICM_RGB_CACHE_DIR=/data/xzha593/tmp/xicm_rgb_cache
export RLBENCH_SKIP_WAYPOINT_VALIDATION=1

cd "$XICM"

if [ -e "$COPPELIASIM_REAL/xcbglintegrations/libqxcb-egl-integration.so" ]; then
    echo "ERROR: EGL plugin still exists. Move it out first:"
    echo "$COPPELIASIM_REAL/xcbglintegrations/libqxcb-egl-integration.so"
    exit 1
fi

ln -sfn /data/xzha593/software/xkb-rhel8/usr/bin/xkbcomp /tmp/xkbcomp
rm -rf /tmp/xkb
ln -sfn /data/xzha593/software/xkb-rhel8/usr/share/X11/xkb /tmp/xkb

export XVFB8=/data/xzha593/software/xvfb-rhel8
export XVFB8_PATCHED=$XVFB8/usr/bin/Xvfb.patched
export XVFB8_LIB=$XVFB8/usr/lib64

export DISPLAY_NUM=$((100 + SLURM_JOB_ID % 1000))
export DISPLAY=:$DISPLAY_NUM

env LD_LIBRARY_PATH=$XVFB8_LIB:$CONDA_PREFIX/lib:$SYSROOT_LIB \
  $XVFB8_PATCHED $DISPLAY \
  -screen 0 1024x768x24 \
  +extension GLX +render +iglx -noreset \
  > /tmp/xvfb8_${SLURM_JOB_ID}.log 2>&1 &

export XVFB_PID=$!
sleep 3

if ! ps -p $XVFB_PID > /dev/null; then
    echo "Xvfb.patched failed to start."
    cat /tmp/xvfb8_${SLURM_JOB_ID}.log
    exit 1
fi

trap 'kill $XVFB_PID || true' EXIT

# echo "===== START lang_vis.out ====="
# bash scripts/eval_XICM.sh "0,50,99" 25 Qwen2.5.7B.instruct 18 0 "lang_vis.out"

# echo "===== DONE lang_vis.out ====="

echo "===== START random ====="
bash scripts/eval_XICM.sh "0,50,99" 25 Qwen2.5.7B.instruct 18 0 "random"

echo "===== DONE random ====="
echo "===== ALL DONE ====="