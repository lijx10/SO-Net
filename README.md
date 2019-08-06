
# SO-Net
**SO-Net: Self-Organizing Network for Point Cloud Analysis.** CVPR 2018, Salt Lake City, USA
Jiaxin Li, Ben M. Chen, Gim Hee Lee, National University of Singapore



## Introduction
SO-Net is a deep network architecture that processes 2D/3D point clouds. It enables various applications including but not limited to classification, shape retrieval, segmentation, reconstruction. The arXiv version of SO-Net can be found [here](https://arxiv.org/abs/1803.04249).
```
@article{li2018sonet,
      title={SO-Net: Self-Organizing Network for Point Cloud Analysis},
      author={Li, Jiaxin and Chen, Ben M and Lee, Gim Hee},
      journal={arXiv preprint arXiv:1803.04249},
      year={2018}
}
```
Inspired by Self-Organizing Network (SOM), SO-Net performs dimensional reduction on point clouds and extracts features based on the SOM nodes, with theoretical guarantee of invariance to point order. SO-Net explicitly models the spatial distribution of points and provides precise control of the receptive field overlap.

This repository releases codes of 4 applications:
* Classification - ModelNet 40/10, MNIST dataset
* Shape Retrieval - SHREC 2016 dataset
* Part Segmentation - ShapeNetPart dataset
* Auto-encoder - ModelNet 40/10, SHREC 2016, ShapeNetPart


## Installation
Requirements:
- Python 3
- [PyTorch 0.4 or higher](http://pytorch.org/)
- [Faiss](https://github.com/facebookresearch/faiss)
- [visdom](https://github.com/facebookresearch/visdom)
- Compile customized cuda code:
```
cd models/index_max_ext
python3 setup.py install
```

Optional dependency:
 - Faiss [GPU support](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) - required by auto-encoder
 
## Dataset
For [ModelNet40/10](https://1drv.ms/u/s!ApbTjxa06z9CgQfKl99yUDHL_wHs) and [ShapeNetPart](https://1drv.ms/u/s!ApbTjxa06z9CgQnl-Qm6KI3Ywbe1), we use the pre-processed dataset provided by [PointNet++](https://github.com/charlesq34/pointnet2) of Charles R. Qi. For SHREC2016, we sampled points uniformly from the original `*.obj` files. Matlab codes that perform sampling is provided in `data/`.

In SO-Net, we can decouple the SOM training as data pre-processing. So we further process the datasets by generating a SOM for each point cloud. The codes for batch-SOM training can be found in `data/`.

In addition, our prepared datasets can be found in [Google Drive](https://drive.google.com/open?id=184MbflF_RbDX9MyML3hid7OxsYJ8oQQ7): MNIST, ModelNet, ShapeNetPart, SHREC2016.
 
## Usage
### Configuration
The 4 applications share the same SO-Net architecture, which is implemented in `models/`. Typically each task has its own folder like `modelnet/`, `part-seg/` that contains its own configuration `options.py`, training script `train.py` and testing script `test.py`.

To run these tasks, you may need to set the dataset type and path in `options.py`, by changing the default value of `--dataset`, `--dataroot`.
### Visualization
We use visdom for visualization. Various loss values and the reconstructed point clouds (in auto-encoder) are plotted in real-time. Please start the visdom server before training, otherwise there will be warnings/errors, though the warnings/errors won't affect the training process.
```
python3 -m visdom.server
```
The visualization results can be viewed in browser with the address of:
```
http://localhost:8097
```
### Application - Classification
Point cloud classification can be done on ModelNet40/10 and SHREC2016 dataset. Besides setting `--dataset` and `--dataroot`, `--classes` should be set to the desired class number, i.e, 55 for SHREC2016, 40 for ModelNet40 and 10 for ModelNet10.
```
cd modelnet/
python3 train.py
```
### Application - Shape Retrieval
The training of shape retrieval is the same as classification, while at testing phase, the score vector (length 55 for SHREC2016) is regarded as the feature vector. We calculate the L2 feature distance between each shape in the test set and all shapes in the same predicted category from the test set (including itself). The corresponding retrieval list is constructed by sorting these shapes according to the feature distances.
```
cd shrec16/
python3 train.py
```
### Application - Part Segmentation
Segmentation is formulated as a per-point classification problem.
```
cd part-seg/
python3 train.py
```
### Application - Auto-encoder
An input point cloud is compressed into a feature vector, based on which a point cloud is reconstructed to minimize the Chamfer loss. Supports ModelNet, ShapeNetPart, SHREC2016.
```
cd autoencoder/
python3 train.py
```

## License
This repository is released under MIT License (see LICENSE file for details).