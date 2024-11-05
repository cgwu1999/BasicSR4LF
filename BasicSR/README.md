
# Requirements
Our environment:
+ Python >= 3.9
+ torch 2.2.2
+ scikit-image 0.22.0
+ NVDIA GPU RTX4090*4 + CUDA 11.8

# Installation
Please refer to the document of [BasicSR-Installation](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md).

# Data preparation
+ Please download and pre-process the datasets following the repo [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR).
+ For some servers, the `*.h5` file is hard to read. We suggest transferring the `*.h5` file into `*.npy` file.

# Training & Evaluation
Training:
```shell
python -m torch.distributed.launch --nproc_per_node=4 --master_port=46667 basicsr/train.py -opt options/train/final.yml --launcher pytorch 
```
Evaluation:
```shell
python basicsr/test.py -opt options/test/test_final.yml
```
## Related Projects
+ [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)
+ [BasicSR](https://github.com/XPixelGroup/BasicSR)
