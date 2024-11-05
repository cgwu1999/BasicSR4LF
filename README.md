# BasicSR4LF

The official repository of ''**Light Field Super-Resolution with Hybrid Attention Network**''.

Our project is primarily based on [BasicSR](https://github.com/XPixelGroup/BasicSR/tree/master/basicsr), inheriting its usability and efficiency.

## Key Modifications
+ `BasicSR/basicsr/models/lfsr_model.py`
+ `BasicSR/basicsr/archs/*`
+ `BasicSR/basicsr/data/LF_dataset.py`
+ `BasicSR/options/*`





# How to use this project

1. Install [BasicSR](https://github.com/XPixelGroup/BasicSR/tree/master/basicsr) pakage.
2. Download and prepare the datasets following the [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR) instructions.
3. Train the model using the following command:
   ```shell
   python BasicSR/basicsr/train.py -opt BasicSR/options/train/final.yml
   ``` 
   or
   ```shell
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 BasicSR/basicsr/train.py -opt BasicSR/options/train/final.yml --launcher pytorch
   ```
4. Test the model using the following command:
   ```shell
   python BasicSR/basicsr/test.py -opt BasicSR/options/test/test_final.yml
   ```


# How to migrate a model from BasicLFSR.

It is only few step to modify the code from BasicLFSR to our project.

1. In the `BasicSR/basicsr/models/` directory, add a new file named after your model, e.g., `lfssr_arch.py`. note: the file ends with `_arch.py`.
2. Import the necessary packages:
    ```python
    from basicsr.utils.registry import ARCH_REGISTRY
    from .lf_utils import LFDataWarp
    import argparse
    ```
3. Add the `@ARCH_REGISTRY.register()` decorator before the class name.
4. Replace the class name in the `get_model` function with your model name.    
5. Create an `args` namespace using `args=argparse.Namespace(**args)`.
6. Add the `@LFDataWarp` decorator to the `forward` function.
7. Add a YAML configuration file in the `/BasicSR/options/` directory. The main differences are as follows:
    ```yml
    network_g:
        type: DistgSSR
        args:
        scale_factor: 2
        angRes_in: 5
    path:
        param_key_g: state_dict
    ```

## Differences from BasicLFSR
|              | Our Implementation | BasicLFSR   |
| ------------ | ------------------ | ----------- |
| Test Patches | Non-overlapping    | Overlapping |
| Crop Border  | 2                  | 0           |

1. We believe that using overlapping patches for testing is not necessary. Therefore, we tested with patches that are as non-overlapping as possible (except at the edge positions).
2. We have inherited the crop settings from BasicSR. We consider a few pixels at the border to be ill-posed, so we crop 2 pixels.


## Test result
We retest the methods from BasicLFSR with our project.
Here is the result for $\times2$ SR.
| Model       | EPFL          | HCInew        | HCIold        | INRIA         | STFgantry     |
| ----------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| RCAN        | 33.738/0.9433 | 34.715/0.9259 | 40.964/0.9726 | 35.936/0.9551 | 36.213/0.9769 |
| resLF       | 34.611/0.9538 | 36.449/0.9514 | 43.237/0.9848 | 36.887/0.9612 | 37.990/0.9848 |
| LFSSR       | 35.137/0.9590 | 36.569/0.9529 | 43.609/0.9861 | 37.535/0.9657 | 37.658/0.9833 |
| LF-ATO      | 35.412/0.9608 | 36.976/0.9564 | 44.006/0.9871 | 37.931/0.9672 | 39.232/0.9884 |
| LF_InterNet | 35.462/0.9613 | 36.914/0.9551 | 44.285/0.9876 | 37.913/0.9673 | 38.041/0.9846 |
| MEG-Net     | 35.721/0.9629 | 37.202/0.9584 | 44.037/0.9872 | 38.199/0.9684 | 38.554/0.9864 |
| LF-IINet    | 35.796/0.9634 | 37.514/0.9607 | 44.597/0.9883 | 38.383/0.9689 | 39.420/0.9891 |
| DPT         | 35.443/0.9609 | 37.060/0.9568 | 44.015/0.9870 | 37.887/0.9671 | 38.979/0.9877 |
| LFT         | 35.841/0.9635 | 37.490/0.9602 | 44.124/0.9874 | 38.380/0.9689 | 39.881/0.9897 |
| DistgSSR    | 35.978/0.9642 | 37.623/0.9612 | 44.594/0.9882 | 38.476/0.9692 | 39.790/0.9893 |
| LFSSR_SAV   | 35.813/0.9631 | 37.224/0.9580 | 44.119/0.9870 | 38.279/0.9681 | 38.454/0.9860 |
| EPIT        | 35.790/0.9629 | 37.974/0.9646 | 44.872/0.9887 | 38.270/0.9684 | 41.705/0.9931 |
| HLFSR-SSR   | 36.349/0.9664 | 38.029/0.9633 | 44.791/0.9885 | 38.872/0.9710 | 40.205/0.9905 |
| LF-DET      | 36.302/0.9662 | 38.034/0.9634 | 44.786/0.9888 | 38.818/0.9708 | 41.261/0.9924 |

Here is the result for $\times4$ SR.
| Model       | EPFL          | HCInew        | HCIold        | INRIA         | STFgantry     |
| ----------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| RCAN        | 28.132/0.8268 | 29.475/0.7976 | 35.155/0.8960 | 30.206/0.8718 | 28.779/0.8803 |
| resLF       | 29.023/0.8542 | 30.555/0.8413 | 36.579/0.9304 | 31.359/0.8948 | 30.076/0.9148 |
| LFSSR       | 29.450/0.8668 | 30.773/0.8481 | 36.784/0.9334 | 31.837/0.9034 | 30.472/0.9217 |
| LF-ATO      | 29.513/0.8670 | 30.678/0.8452 | 36.944/0.9343 | 32.054/0.9069 | 30.501/0.9233 |
| LF_InterNet | 29.750/0.8740 | 30.841/0.8518 | 37.159/0.9389 | 32.062/0.9086 | 30.318/0.9193 |
| MEG-Net     | 29.741/0.8730 | 30.962/0.8542 | 37.221/0.9384 | 32.072/0.9079 | 30.655/0.9255 |
| LF-IINet    | 29.978/0.8772 | 31.187/0.8599 | 37.445/0.9419 | 32.369/0.9114 | 31.085/0.9333 |
| DPT         | 29.500/0.8640 | 30.814/0.8486 | 36.770/0.9312 | 31.658/0.8965 | 30.474/0.9195 |
| LFT         | 30.093/0.8795 | 31.269/0.8599 | 37.648/0.9426 | 32.483/0.9124 | 31.689/0.9388 |
| DistgSSR    | 29.783/0.8729 | 31.222/0.8601 | 37.496/0.9416 | 32.044/0.9055 | 31.395/0.9347 |
| LFSSR_SAV   | 30.248/0.8813 | 31.302/0.8598 | 37.484/0.9394 | 32.547/0.9132 | 31.247/0.9338 |
| EPIT        | 30.051/0.8775 | 31.349/0.8629 | 37.550/0.9423 | 32.458/0.9122 | 32.047/0.9430 |
| HLFSR-SSR   | 30.187/0.8815 | 31.390/0.8642 | 37.697/0.9435 | 32.600/0.9137 | 31.461/0.9379 |
| LF-DET      | 30.286/0.8834 | 31.399/0.8635 | 37.779/0.9442 | 32.639/0.9146 | 32.034/0.9435 |

# References

+ [BasicSR](https://github.com/XPixelGroup/BasicSR/tree/master/basicsr)
+ [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)