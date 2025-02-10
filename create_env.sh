## Bash script for setting up the conda environment
ENV_NAME=$1
#env_name=
env_exists=$(micromamba env list | grep -w "${ENV_NAME}")
if [ -z "$env_exists" ]; then
    # The environment does not exist, so create it
    micromamba create -y --name $ENV_NAME 
fi
micromamba install -y -n $ENV_NAME python=3.11 jupyter -c conda-forge
micromamba install -y -n $ENV_NAME numpy anndata pandas seaborn wandb pynrrd scikit-learn -c conda-forge
micromamba install -y -n $ENV_NAME pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
micromamba deactivate
conda deactivate
micromamba activate $ENV_NAME
python -m pip install transformers datasets evaluate
python -m pip install deepspeed
python -m pip install accelerate -U
python -m pip install loompy
python -m pip install isort
python -m pip install jax[cuda12] jax absl-py optax
python -m pip install jiwer pymatreader
python -m pip install hydra-core --upgrade
python -m pip install ccf_streamlines
python -m pip install colour-science