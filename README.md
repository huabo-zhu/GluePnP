# GluePnP
A generalized differentiable blind PnP method.

## Requirements

- `pip install -r requirements.txt`

torch==1.8.0
torchvision==0.9.0
tensorboard
numpy
scipy
opencv-python>=4.5.0
pyro-ppl==1.4.0
PyYAML==5.4.1
matplotlib
termcolor
plyfile
easydict 
progress 
numba

## Datasets

The training data we use is provided by bpnpnet. Details can be found at https://github.com/dylan-campbell/bpnpnet .

The pre-processed (randomised) data is available for download at this link (https://drive.google.com/file/d/1y4cbbcVEJFfB3y171GiFSZi6AtgW2nL_/view?usp=sharing) (2.1GB).

## Training

Firstly, you need to download the dataset.

To train the model, run `model_train.py` with the modelnet40 or megadepth.

Command:

```python
python model_train.py data_dir --dataset dataset_name(megadepth or modelnet40) --poseloss 12(Start using Lp) --gpu 0(Number of GPUs) --log-dir log_path --lr learning_rate
```

For example:

```python
python model_train.py I://bpnpnet --dataset megadepth --poseloss 15 --gpu 0 --log-dir ./output  --lr 1e-4
```

**Parameter Description**:

data_dir, path to datasets directory

--dataset, dataset name

-j or --workers, number of data loading workers

--epochs, number of total epochs to run

--start-epoch, manual epoch number (useful on restarts)

-b or --batch-size, mini-batch size

--lr or --learning-rate, initial learning rate(megadepth=1e-4, modelnet40=5e-5)

--resume, path to latest checkpoint

--seed, seed for initializing training

--gpu, GPU id to use (When it>0 indicates the number of GPUs)

--log-dir, Directory for logging loss and accuracy

--poseloss, specify epoch at which to introduce pose loss

--frac_outliers, Proportion of Outliers(set as 0 during normal training)

--sort, n number of subdata(=0 represents using all points,>0 represents using only the first n points)

## Testing

To test a model, run `eval.py` from a checkpoint, with the desired dataset (modelnet40 or megadepth) and the log and data directories:

```python
python eval.py
```

**Parameter Description**:

data_name, megadepth or modelnet40 

data_path, path to datasets directory

method, =ours, dbpnp, ransac, (r1ppnp and vpnp -> need the output from MATLAB)

eval_data,  file path of MATLAB output (Only required when the methods are r1ppnp and vpnp)



## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
We will add information after the paper is published. For the sake of fairness in the review, we will not disclose any relevant information for the time being.
```

Our code was modified from bpnpnet. Please cite the following paper:

```
@inproceedings{campbell2020solving,
  author = {Campbell, Dylan and Liu, Liu and Gould, Stephen},
  title = {Solving the Blind Perspective-n-Point Problem End-To-End With Robust Differentiable Geometric Optimization},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={preprint},
  location={Glasgow, UK},
  month = {Aug},
  year = {2020},
  organization={Springer},
}
```

## License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)]