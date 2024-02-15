#PATH=~/software/miniconda3/bin:~/anaconda3/bin:$PATH
cd softgym_medor
. ./prepare_1.0.sh
cd ..
export PYTORCH_JIT=0
export PYTHONPATH=${PWD}:${PWD}/softgym_medor:$PYTHONPATH
export PYTHONPATH=${PWD}/garmentnets:$PYTHONPATH
export MUJOCO_gl=egl
export EGL_GPU=$CUDA_VISIBLE_DEVICES
