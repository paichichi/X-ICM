source /data/xzha593/miniconda3/etc/profile.d/conda.sh
conda activate /data/xzha593/envs/xicm

export XICM=/data/xzha593/projects/X-ICM
export COPPELIASIM_ROOT=/data/xzha593/software/CoppeliaSim
export COPPELIASIM_REAL=$(readlink -f "$COPPELIASIM_ROOT")

export LLVM7_LIB=/data/xzha593/software/llvm-private/usr/lib64
export SYSROOT_LIB=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64
export DRI_DIR=$SYSROOT_LIB/dri

export LD_LIBRARY_PATH=$LLVM7_LIB:$SYSROOT_LIB:$COPPELIASIM_REAL:$CONDA_PREFIX/lib

export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_REAL/platforms
export QT_PLUGIN_PATH=$COPPELIASIM_REAL
export QT_QPA_PLATFORM=offscreen

export LIBGL_DRIVERS_PATH=$DRI_DIR
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=swrast
export GALLIUM_DRIVER=llvmpipe

export PYTHONPATH=$XICM:$XICM/YARR:${PYTHONPATH:-}

cd "$XICM"
