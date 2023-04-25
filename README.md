We provide the code (in pytorch) and datasets for our paper [**"GraphPrompt: Unifying Pre-Training and Downstream Tasks
for Graph Neural Networks"**](https://arxiv.org/pdf/2302.08043.pdf), 
which is accepted by WWW2023.

## Description
The repository is organised as follows:
- **data/**: contains data we use.
- **graphdownstream/**: implements pre-training and downstream tasks at the graph level.
- **nodedownstream/**: implements downstream tasks at the node level.
- **convertor/**: generate raw data.

## Package Dependencies
* cuda 11.3
* dgl0.9.0-cu113
* dgllife

## Running experiments
### Graph Classification
Default dataset is ENZYMES. You need to change the corresponding parameters in *pre_train.py* and *prompt_fewshot.py* to train and evaluate on other datasets.

Pretrain:
- python pre_train.py
 
Prompt tune and test:
- python prompt_fewshot.py

### Node Classification

Default dataset is ENZYMES. You need to change the corresponding parameters in *prompt_fewshot.py* to train and evaluate on other datasets. 

Prompt tune and test:
- python run.py

## Note

The code of this project contains a lot of redundancy and is not very readable due to inheriting code from my previous work *Count-GNN*. I deeply apologize for this. I have rewritten the code for GraphPrompt in my new research work, and once the new paper is accepted, I will upload the refactored code to a new repository. Thank you for your understanding and patience.

## Citation
@inproceedings{liu2023graphprompt,\
  title={GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks},\
  author={Liu, Zemin and Yu, Xingtong and Fang, Yuan and Zhang, Xinming},\
  booktitle={Proceedings of the ACM Web Conference 2023},\
  year={2023}\
}
