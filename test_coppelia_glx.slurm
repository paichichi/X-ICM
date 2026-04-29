#!/bin/bash
#SBATCH --job-name=test_glx
#SBATCH --time=00:20:00
#SBATCH --output=/data/xzha593/projects/X-ICM/slurm_logs/test_glx_%j.log
#SBATCH --error=/data/xzha593/projects/X-ICM/slurm_logs/test_glx_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail

mkdir -p /data/xzha593/projects/X-ICM/slurm_logs

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

export PYTHONPATH=$XICM:$XICM/YARR:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

cd "$XICM"

# XKB paths required by patched Xvfb.
ln -sfn /data/xzha593/software/xkb-rhel8/usr/bin/xkbcomp /tmp/xkbcomp
rm -rf /tmp/xkb
ln -sfn /data/xzha593/software/xkb-rhel8/usr/share/X11/xkb /tmp/xkb

# Start patched Xvfb.
export XVFB8=/data/xzha593/software/xvfb-rhel8
export XVFB8_PATCHED=$XVFB8/usr/bin/Xvfb.patched
export XVFB8_LIB=$XVFB8/usr/lib64

export DISPLAY_NUM=$((120 + SLURM_JOB_ID % 50))
export DISPLAY=:$DISPLAY_NUM

env LD_LIBRARY_PATH=$XVFB8_LIB:$CONDA_PREFIX/lib:$SYSROOT_LIB \
  $XVFB8_PATCHED $DISPLAY \
  -screen 0 1024x768x24 \
  +extension GLX +render +iglx -noreset \
  > /tmp/xvfb_test_${SLURM_JOB_ID}.log 2>&1 &

export XVFB_PID=$!
sleep 3

if ! ps -p $XVFB_PID > /dev/null; then
    echo "Xvfb.patched failed"
    cat /tmp/xvfb_test_${SLURM_JOB_ID}.log
    exit 1
fi

trap 'kill $XVFB_PID || true' EXIT

echo "===== ENV ====="
echo "DISPLAY=$DISPLAY"
echo "QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
echo "QT_XCB_GL_INTEGRATION=$QT_XCB_GL_INTEGRATION"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
cat /tmp/xvfb_test_${SLURM_JOB_ID}.log || true

echo "===== VisionSensor test ====="
python - <<'PY'
import os
import numpy as np

print("DISPLAY:", os.environ.get("DISPLAY"))
print("QT_QPA_PLATFORM:", os.environ.get("QT_QPA_PLATFORM"))
print("QT_XCB_GL_INTEGRATION:", os.environ.get("QT_XCB_GL_INTEGRATION"))

from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor

pr = PyRep()
pr.launch(headless=True)
print("CoppeliaSim launched")

vs = VisionSensor.create([128, 128])
print("VisionSensor created")

pr.step()
rgb = vs.capture_rgb()
print("RGB captured:", rgb.shape, rgb.dtype, np.min(rgb), np.max(rgb))

pr.shutdown()
print("SLURM VisionSensor GLX test OK")
PY
