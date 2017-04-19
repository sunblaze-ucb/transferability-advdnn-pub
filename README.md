# Delving into Transferable Adversarial Examples and Black-box Attacks

This repo provides the code to replicate the experiments in the paper

> Yanpei Liu, Xinyun Chen, Chang Liu, Dawn Song, <cite> Delving into Transferable Adversarial Examples and Black-box Attacks </cite>, in Proceedings of 5th International Conference on Learning Representations (ICLR 2017)

Paper [[arXiv](https://arxiv.org/abs/1611.02770)] [[OpenReview](https://openreview.net/forum?id=Sys6GJqxl&noteId=Sys6GJqxl)]

# Datasets

## ILSVRC12
You can get the dataset by
```bash
cd scripts
bash retrieve_data.sh
```
Or download validation dataset from official website: [[ImageNet](https://www.image-net.org/challenges/LSVRC/2012/)] to data/test_data folder

# Usage

## Model architectures

The code currently only supports GoogleNet, will add more models in the later updates

## Run experiments

In the following we list some important arguments for our python codes:
* `--input_dir`: Directory of dataset.
* `--output_dir`: Directory of output noise file.
* `--model`: Models to be evaluated, now only supports GoogleNet
* `--num_images`: Max number of images to be evaluated (optional).
* `--file_list`: Evaluate a specific list of file in dataset (optional).
* `--num_iter`: Number of iterations to generate attack (optional).
* `--learning_rate`: Learning rate of each iteration (optional).
* `--use_round`: Round to integer (optional).
* `--weight_loss2`: Weight of distance penalty (optional).
* `--noise_file`: Directory of added noise (optional).
* `--target`: Target list of file in dataset, if not provided then will do nontargeted attack (optional).

# Citation

If you use the code in this repo, please cite the following paper:

```
@inproceedings{liu2017delving,
  author    = {Yanpei Liu and
               Xinyun Chen and
               Chang Liu and
               Dawn Song},
  title     = {Delving into Transferable Adversarial Examples and Black-box Attacks},
  year      = {2016},
  booktitle = {Proceedings of 5th International Conference on Learning Representations},
}
```
