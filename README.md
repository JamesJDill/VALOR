# [VALOR](https://arxiv.org/pdf/1807.10299)

This is a PyTorch implementation of [VALOR](https://arxiv.org/pdf/1807.10299) — **Visual-Action Latent Optimization for Reinforcement Learning** — a method designed to learn rich visual-latent representations from high-dimensional observations to accelerate and stabilize policy learning. This repository provides suppoprt training, evaluation, and visualizations of VALOR on MuJoCo Gym environments.

### Environment Set-up
```bash
git clone https://github.com/JamesJDill/VALOR.git
cd VALOR

conda create -n valor python=3.11 -y
conda activate valor

pip install -r requirements.txt
```

# Demo
Run all the cells in demo.ipynb in order to run a training and see the associated evaluations/visualizations.

This REPO was tested and trained using a GPU. CPU is untested but should work. Make sure to decrease NUM_ENVS during training if on CPU.
