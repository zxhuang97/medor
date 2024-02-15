# MEDOR


# Installation
## Clone the repo
```
git clone --recursive git@github.com:zxhuang97/medor.git
```
## Configure python environment
`Mamba` is highly recommended for configuring the python environment. It's a drop-in replacement for `conda` but much faster. 
```
mamba env create -f release.yml
```
## Install Softgym
Step 1: Install [docker](https://docs.docker.com/engine/install/ubuntu/) and [nvidia-container](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Then pull the docker image by
```
docker pull xingyu/softgym
```
Step 2: Set the path to conda directory as `CONDA_PATH`, then enter the docker container.
```
export CONDA_PATH=/home/zixuanh/miniforge3
sudo docker run \
--runtime=nvidia  \
-v ${PWD}/softgym_medor:/workspace/softgym   \
-v ${CONDA_PATH}:${CONDA_PATH} \
-v /tmp/.X11-unix:/tmp/.X11-unix  \
--gpus all   \
-e DISPLAY=$DISPLAY  \
-e QT_X11_NO_MITSHM=1   \
-it xingyu/softgym:latest bash
```
Step 3: Inside the docker container, run the following commands to compile the softgym.
```
cd softgym
export CONDA_PATH=/home/zixuanh/miniforge3
export PATH=${CONDA_PATH}/bin:$PATH
. ./prepare_1.0.sh
export PATH=/usr/local/cuda/bin/:$PATH
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
. ./compile_1.0.sh
```
# Mesh Reconstruction
## Dataset
Currently, we only provide dataset for Tshirt. For training and testing, you need to full dataset. 
If you only want to run the demo, you can download the test set alone.
- [Full dataset](https://drive.google.com/file/d/1JrC2vHrdxXvfjgcmn2tz1eT81U2noTlP/view?usp=sharing)
- [Test set](https://drive.google.com/file/d/1klTUl5xaja3izQ5GoLjDPn88dwbkrERo/view?usp=sharing)

## Demo
Download the [pretrained model](https://drive.google.com/file/d/1Tv5lJuuqI1QLQctgxwtDIebaKeoxp59v/view?usp=sharing) and put it under `data/release`.
```angular2html
data
└── release
    └── tshirt_release
dataset
└── Tshirt_dataset_release2
```

`make_opt_gif` will generate the gifs that visualize the optimization process and each gif will take around 3-4 mins.

```
. ./prepare_release.sh
python garmentnets/eval_pipeline.py \
--model_path data/release/tshirt_release/pipeline/ \
--tt_finetune --cloth_type Tshirt --max_test_num 5 \
--exp_name release_demo 
```
The results can be found in `data/test/release_demo`.
# Training
Train the canonicalization Networks
```
 python garmentnets/train_pointnet2.py \
  --exp_name tshirt_canon \
  --log_dir data/release/Tshirt_release  \
  --ds Tshirt_dataset_release2 \
  --cloth_type Tshirt
```
Train the mesh reconstruction pipeline
```
python garmentnets/train_pipeline.py \
--exp_name tshirt_pipeline \
--log_dir data/release/Tshirt_release \
--ds Tshirt_dataset_release2 \
--cloth_type Tshirt \
--canon_checkpoint data/release/Tshirt_release/tshirt_canon
```

# Mesh GNN
TODO
# Planning
TODO
