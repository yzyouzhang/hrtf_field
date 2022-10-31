# HRTF_field

Official implementation of the paper "HRTF Field: Unifying Measured HRTF Magnitude Representation with Neural Fields".
[[arXiv](https://arxiv.org/abs/2210.15196)]

### Requirements
```
pytorch
python-sofa
librosa
natsort
pandas
plotly
wandb
```

### How to run

An example
```
python3 train.py -o /data/neil/hrtf_field/1028_riec_int_train -n ari hutubs ita cipic 3d3a bili listen crossmod sadie
```

### Citation
```
@article{zhang2022hrtf,
  title={HRTF Field: Unifying Measured HRTF Magnitude Representation with Neural Fields},
  author={Zhang, You and Wang, Yuxiang and Duan, Zhiyao},
  journal={arXiv preprint arXiv:2210.15196},
  year={2022}
}
```
