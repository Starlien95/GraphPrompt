We provide the code (in pytorch) and datasets for our paper [**"GraphPrompt: Unifying Pre-Training and Downstream Tasks
for Graph Neural Networks"**](https://arxiv.org/pdf/2302.08043.pdf), 
which is accepted by WWW2023.

## Package Dependencies
* cuda 11.3
* dgl-cu113
* dgllife

# Running experiments
## Graph Classification
Default dataset is PROTEINS. You need to change the corresponding parameters in *pre_train.py* and *prompt_fewshot.py* to train and evaluate on other datasets.

Pretrain:
- python pre_train.py
 
Prompt tune and test:
- python prompt_fewshot.py

## Node Classification

Default dataset is PROTEINS. You need to change the corresponding parameters in *prompt_fewshot.py* to train and evaluate on PROTEINS. Flikcr uses different file to pretrain and tune.

Prompt tune and test:
- python run.py

Flickr prompt tune and test:
- python pre_train_flickr.py
- python prompt_fewshot_flickr.py
