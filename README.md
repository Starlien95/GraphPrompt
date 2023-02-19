We provide the code (in pytorch) and datasets for our paper [**"GraphPrompt: Unifying Pre-Training and Downstream Tasks
for Graph Neural Networks"**](https://arxiv.org/pdf/2302.08043.pdf), 
which is accepted by WWW2023.


# Running experiments
## Graph Classification
Default dataset is PROTEINS. You need to change the corresponding parameters in *pre_train.py* and *prompt_fewshot.py* to train and evaluate on other datasets.
### Pre-train
python pre_train.py 

### Prompt tune and test
python prompt_fewshot.py

## Node Classification

Default dataset is PROTEINS. You need to change the corresponding parameters in *prompt_fewshot.py* to train and evaluate on PROTEINS. Flikcr uses different file to pretrain and tune.
## Prompt tune and test
python run.py

## Flickr Pre-train
python pre_train_flickr.py
## Flickr Prompt tune and test
python prompt_fewshot_flickr.py
