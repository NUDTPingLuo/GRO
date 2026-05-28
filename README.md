# GRO: Rectifying Local Gradient Conflicts via Swarm Dynamics for Robust Federated Learning

This repository provides an official PyTorch implementation of **GRO (Federated Geometric Reflection Optimizer)**, a plug-and-play optimization strategy designed to improve the stability and generalization of **federated learning (FL)** under **heterogeneous (non-IID) data distributions**.

GRO rectifies local gradient conflicts via swarm dynamics, effectively suppressing noisy or conflicting updates while preserving effective descent directions. The proposed strategy can be seamlessly integrated into existing FL baselines without modifying local training procedures.

---

## Features

* **Federated Learning Baselines**: PyTorch implementations of representative FL baselines and their GRO-enhanced variants. The currently supported methods include:
  + [FedAvg](https://arxiv.org/abs/1602.05629) (H. B. McMahan et al., AISTATS 2017)
  + [FedNova](https://arxiv.org/abs/2007.07481) (J. Wang et al., NeurIPS 2020) [:octocat:](https://github.com/JYWa/FedNova)
  + [FedProx](https://arxiv.org/abs/1812.06127) (T. Li et al., MLSys 2020) [:octocat:](https://github.com/litian96/FedProx)
  + [SCAFFOLD](https://arxiv.org/abs/1910.06378) (S. P. Karimireddy et al., ICML 2020) [:octocat:](https://github.com/ki-ljl/Scaffold-Federated-Learning)

* **GRO Aggregation Module**: Implementation of **Federated Geometric Reflection Optimizer (GRO)** as a modular aggregation strategy that can be combined with existing FL baselines to improve robustness under non-IID data distributions.

* **Dataset Preprocessing**: Automated downloading and preprocessing of benchmark datasets, followed by partitioning into multiple clients according to federated learning settings. Non-IID data distributions are simulated via Dirichlet-based label skew. The currently supported datasets include MNIST, Fashion-MNIST, SVHN, CIFAR-10, and CIFAR-100. Other datasets (e.g., medical imaging datasets) need to be downloaded and organized manually.

* **Postprocessing and Visualization**: Tools for visualizing training dynamics and evaluating global model performance, including testing accuracy curves averaged over multiple random seeds.

---

## Installation

### Dependencies

- Python (3.8)
- PyTorch (1.8.1)
- OpenCV (4.5)
- NumPy (1.21.5)

### Install Requirements

Install all required packages by running:

```bash
pip install -r requirements.txt
```
## Running Federated Learning with ECGR

### Test Run

All hyperparameters are specified in a YAML configuration file (e.g., `./config/test_config.yaml`).  
To run federated learning experiments with ECGR or baseline methods, execute:

```bash
python fl_main.py --config "./config/test_config.yaml"
```

## Evaluation Procedures

You can place the results in the `results/test` directory, and then run the following command:

```bash
python postprocessing/eval_main.py -rr ../results/test


