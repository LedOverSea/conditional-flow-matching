# 项目简介

基于TorchCFM进行的单细胞数据实验

`notes/` 包含所有demo, 除了`notes/3.single-cell_preprocess.ipynb`在本地运行，其他笔记本都在kaggle运行

`runner/configs/`实验配置

`runner/scripts/`实验脚本

`runner/logs/`模型参数及运行结果



## 运行

安装:

```bash
pip install torchcfm
```

同时安装以下依赖:

```bash
# clone project
git clone https://github.com/atong01/conditional-flow-matching.git
cd conditional-flow-matching

# [OPTIONAL] create conda environment
conda create -n torchcfm python=3.10
conda activate torchcfm

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt

# install torchcfm
pip install -e .
```

运行jupyter notebook需要以下依赖

```bash
# install ipykernel
conda install -c anaconda ipykernel

# install conda env in jupyter notebook
python -m ipykernel install --user --name=torchcfm

# launch our notebooks with the torchcfm kernel
```

运行`runner/`中的脚本, 如

```bash
cd .\runner

.\scripts\batch_experiments.bat
```



## 项目结构

```

│
├── examples              <- Jupyter notebooks
|   ├── cifar10           <- Cifar10 experiments
│   ├── notebooks         <- Diverse examples with notebooks
│
│── runner                    <- Everything related to the original version (V0) of the library
│
|── torchcfm                  <- Code base of our Flow Matching methods
|   ├── conditional_flow_matching.py      <- CFM classes
│   ├── models                            <- Model architectures
│   │   ├── models                           <- Models for 2D examples
│   │   ├── Unet                             <- Unet models for image examples
|
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```


