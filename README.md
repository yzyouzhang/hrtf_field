# [HRTF Field: Unifying Measured HRTF Magnitude Representation with Neural Fields](https://ieeexplore.ieee.org/document/10095801)

[![GitHub](https://img.shields.io/github/stars/yzyouzhang/hrtf_field)](https://github.com/yzyouzhang/hrtf_field) | [![IEEE Xplore](https://img.shields.io/badge/IEEE-10095801-E4A42C.svg)](https://ieeexplore.ieee.org/document/10095801) | [![arXiv](https://img.shields.io/badge/arXiv-2210.15196-b31b1b.svg)](https://arxiv.org/abs/2210.15196) 

Official implementation of the ICASSP 2023 paper "HRTF Field: Unifying Measured HRTF Magnitude Representation with Neural Fields."

[![Video](https://img.youtube.com/vi/HoQg8YzX1jg/hqdefault.jpg)](https://youtu.be/HoQg8YzX1jg)

## Updates
Oct. 2023, Check out our follow-up work in WASPAA 2023, "Mitigating Cross-Database Differences for Learning Unified HRTF Representation." [![GitHub](https://img.shields.io/github/stars/YutongWen/HRTF_field_norm)](https://github.com/YutongWen/HRTF_field_norm) [![IEEE Xplore](https://img.shields.io/badge/IEEE-10248178-E4A42C.svg)](https://ieeexplore.ieee.org/document/10248178) [![arXiv](https://img.shields.io/badge/arXiv-2307.14547-b31b1b.svg)](https://arxiv.org/abs/2307.14547)

## Implementation


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

### Data
Get data from [SOFA repository](https://www.sofaconventions.org/mediawiki/index.php/Files)

Then preprocess them with our `preprocess.py`.

### How to run

An example
```
python3 train.py -o /data/neil/hrtf_field/model -n ari hutubs ita cipic 3d3a bili listen crossmod sadie -t riec
```


## Citation
```
@INPROCEEDINGS{10095801,
  author={Zhang, You and Wang, Yuxiang and Duan, Zhiyao},
  booktitle={Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={{HRTF} Field: Unifying Measured {HRTF} Magnitude Representation with Neural Fields}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095801}}
```
