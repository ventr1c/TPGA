# Are You Using Reliable Graph Prompts? Trojan Prompt Attacks on Graph Neural Networks
This is the official implementation of "Are You Using Reliable Graph Prompts? Trojan Prompt Attacks on Graph Neural Networks". (KDD 2025) If you find this repo to be useful, please cite our paper. Thank you.
```
@inproceedings{lin2025areyou,
  title={Are You Using Reliable Graph Prompts? Trojan Prompt Attacks on Graph Neural Networks},
  author={Lin, Minhua and Zhang, Zhiwei and Dai, Enyan and Wu, Zongyu and Wang, Yilong and Zhang, Xiang and Wang, Suhang},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
}
```

## Content
- [Are You Using Reliable Graph Prompts? Trojan Prompt Attacks on Graph Neural Networks](#are-you-using-reliable-graph-prompts-trojan-prompt-attacks-on-graph-neural-networks)
  - [Content](#content)
  - [1. Overview](#1-overviews)
  - [2. Requirements](#2-requirements)
  - [3. TGPA](#3-tgpa)
    - [Reproduce the Results](#reproduce-the-results)
    - [Step 1: Pre-train GNN encoders](#step-1-pre-train-gnn-encoders)
    - [Step 2: Conduct Trojan Graph Prompt Attacks](#step-2-conduct-trojan-graph-prompt-attacks)
  - [4. Compared Methods](#4-compared-methods)
  - [5. Dataset](#5-dataset)

## 1. Overview
* `./config`: Contains configuration files used in the project.
* `./Experiment`: This directory holds preprocessed data and models:
  - `answering_model`: the directory contains the trained downstream task header.
  - `induced_graph`: the directory contains the induced $k$-hop neighborhood graphs.
  - `pre_trained_model`: the directory contains the GNN models pre-trained by using self-supervised learning.
  - `prompt_graph`: the directory contains learned graph prompts.
  - `sample_data`: this directory sampled training/test nodes' indexs and labels.
* `./prompt_graph`: Contains modules related to graph prompts processing with subdirectories:
  - `utils`: Various utility functions and scripts.
  - `pretrain`: Scripts and models used for pretraining tasks.
  - `prompt`: Modules handling different prompt-related tasks.
  - `model`: Implementations of different graph models (e.g., GCN, GraphTransformer).
  - `evaluation`: Evaluation functions for assessing model performance.
  - `data`: Data processing and loading utilities.
  - `tasker`: Defines different tasks related to graphs, such as node and link tasks.
* `./Trojans`: This directory contains the implementations of various backdoor attacks:
* `downstream_task_trojan.py`: the script to run the downstream task on the trojan dataset.
* `downstream_task.py`: the script to run the downstream task on the benign dataset.
* `environment.yml`: the environmental configuration file.
* `pre_train.py`: the script to pre-train the GNN encoders
* `scripts.sh`: some representative scripts to reproduce the results.

## 2. Requirements
The required packages are listed in `environment.yml`. You can run the following code to create the environment from this file:
```
conda env create -f environment.yml
```

## 3. TGPA
To reproduce the performance reported in the paper, you can check the bash file:
```
bash scripts.sh
```
### Step 1: Pre-train GNN encoders
To pre-train GNN encoder, you can run the following code:
```
python pre_train.py
```
The pre-trained models will then be saved in `Experiment/pre_trained_model`.
### Step 2: Conduct Trojan Graph Prompt Attacks
To run the trojan graph prompt attacks, you can run:
```
python downstream_task_trojan.py
```
If you want to reproduce the results in the setting of freezing downstream task header, please set the argument `--if_freeze_dt_classifier` as `True`. 

## 4. Compared Methods
### Compared with Graph Backdoor Attack Methods
#### SBA-P
This is a variant from Zhang, Zaixi, et al. "Backdoor Attacks to Graph Neural Networks" [[paper](https://arxiv.org/abs/2006.11165), [code](https://github.com/zaixizhang/graphbackdoor)].
#### GTA-P
This is a variant from Xi, Zhaohan, et al. "Graph Backdoor" [[paper](https://arxiv.org/abs/2006.11890), [code](https://github.com/HarrialX/GraphBackdoor)].

#### UGBA-P
This is a variant from Dai, Enyan, et al. "Unnoticeable Backdoor Attacks on Graph Neural Networks" [[paper](https://arxiv.org/abs/2303.01263), [code](https://github.com/ventr1c/UGBA)].

#### BL-Rand
This is a variant of TGPA. Instead of using adaptive triggers, we inject a fixed subgraph as a universal trigger. The connections of the subgraph are generated based on the ErdosRenyi (ER) model, and its node features are randomly selected from those in the training graph 

## 5. Dataset
The experiments are conducted on four public real-world datasets, i.e., Cora, Citeseer and Pubmed, which can be automatically downloaded to `./data` through torch-geometric API.

