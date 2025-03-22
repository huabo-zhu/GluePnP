# GluePnP
A generalized differentiable blind PnP method.

## Requirements

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



## Training

To train a model, run `train.py` with the dmodelnet40 or megadepth:

```bash
python train.py --dataset modelnet40 --poseloss <pose loss start epoch> --gpu <gpu ID> --log-dir <log-dir> <data-folder>
```





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

