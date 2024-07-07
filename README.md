# Improving Corruption Robustness: Desensitizing the Neural Network to 3D Point Cloud through Adversarial Training

## Prerequisites

Install necessary packages using:
```bash
bash install.sh
```

Install PyGeM
```bash
cd PyGen
python setup.py install
cd ..
```

## Data
Download [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip), [ModelNet40-C](https://github.com/jiachens/ModelNet40-C), and [PointCloud-C](https://drive.google.com/uc?id=1KE6MmXMtfu_mgxg4qLPdEwVD5As8B0rm) datasets and put them in the Data directory:
```plaintext
DenAT/
└── data/
    ├── ModelNet40Ply2048
    ├── modelnet40_c
    └── pointcloud_c
```
## Generate or download the shapely value corresponding to modelNet
Generate the shapely value corresponding to modelNet：
```bash
bash ./script/gen_shapely.py
```
Or you can also download it directly from this [link](https://drive.google.com/drive/folders/1vzz-3QjIQ8VBdwMa32cQGAi8aLmPp1RH?usp=sharing).
You need to put the file in the specified directory：
```plaintext
DenAT/
└── data/
    └── shaply_ModelNet40Ply2048
        └── PointNet2Encoder_ST
            ├──train
            └──test
```
## Script
Train PointNet++ model using ST method:
```bash
bash ./script/trainST.py
```
Train PointNet++ model using DesenAT method:
```bash
bash ./script/train.py
```
Testing in ModelNet40-C 
```bash
bash ./script/test_modelnetC.py
```
Testing in PointCloud-C
```bash
bash ./script/test_pointcloudC.py
```
## Acknowledgment
This repository is built on reusing codes of [OpenPoints](https://github.com/guochengqian/openpoints) and [PointNeXt](https://github.com/guochengqian/PointNeXt). We integrated [APES](https://github.com/JunweiZheng93/APES) and [PointMetaBase](https://github.com/linhaojia13/PointMetaBase) into the code. We also have integrated methods for handling corrupted point clouds into our code, thanks to the excellent work of [ModelNet-C](https://github.com/jiachens/ModelNet40-C) and [PointCloud-C](https://github.com/ldkong1205/PointCloud-C).
