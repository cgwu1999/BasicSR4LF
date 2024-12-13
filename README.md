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
3. Transfer the test set from `*.h` to `*.npy` with `H5toNPY.ipynb`. (large `.h5` file is hard to read for some machine.)
4. Train the model using the following command:
   ```shell
   python BasicSR/basicsr/train.py -opt BasicSR/options/train/final.yml
   ``` 
   or
   ```shell
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 BasicSR/basicsr/train.py -opt BasicSR/options/train/final.yml --launcher pytorch
   ```
5. Test the model using the following command:
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
|                 | Our Implementation | BasicLFSR    |
| --------------- | ------------------ | ------------ |
| Test Patches    | Non-overlapping    | Overlapping  |
| Crop Border     | 2                  | 0            |
| SSIM data_range | 1.0                | 2.0     **** |

1. We believe that using overlapping patches for testing is not necessary. Therefore, we tested with patches that are as non-overlapping as possible (except at the edge positions).
2. We have inherited the crop settings from BasicSR. We consider a few pixels at the border to be ill-posed, so we crop 2 pixels.


## Test result
We retest the methods from BasicLFSR with our project.

Here is the result for $\times2$ SR.
| Model            | EPFL          | HCInew        | HCIold        | INRIA         | STFgantry     |
| ---------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| RCAN             | 33.738/0.9433 | 34.715/0.9259 | 40.964/0.9726 | 35.936/0.9551 | 36.213/0.9769 |
| resLF            | 34.611/0.9538 | 36.449/0.9514 | 43.237/0.9848 | 36.887/0.9612 | 37.990/0.9848 |
| LFSSR            | 35.137/0.9590 | 36.569/0.9529 | 43.609/0.9861 | 37.535/0.9657 | 37.658/0.9833 |
| LF-ATO           | 35.412/0.9608 | 36.976/0.9564 | 44.006/0.9871 | 37.931/0.9672 | 39.232/0.9884 |
| LF_InterNet      | 35.462/0.9613 | 36.914/0.9551 | 44.285/0.9876 | 37.913/0.9673 | 38.041/0.9846 |
| MEG-Net          | 35.721/0.9629 | 37.202/0.9584 | 44.037/0.9872 | 38.199/0.9684 | 38.554/0.9864 |
| LF-IINet         | 35.796/0.9634 | 37.514/0.9607 | 44.597/0.9883 | 38.383/0.9689 | 39.420/0.9891 |
| DPT              | 35.443/0.9609 | 37.060/0.9568 | 44.015/0.9870 | 37.887/0.9671 | 38.979/0.9877 |
| LFT              | 35.841/0.9635 | 37.490/0.9602 | 44.124/0.9874 | 38.380/0.9689 | 39.881/0.9897 |
| DistgSSR         | 35.978/0.9642 | 37.623/0.9612 | 44.594/0.9882 | 38.476/0.9692 | 39.790/0.9893 |
| LFSSR_SAV        | 35.813/0.9631 | 37.224/0.9580 | 44.119/0.9870 | 38.279/0.9681 | 38.454/0.9860 |
| EPIT             | 35.790/0.9629 | 37.974/0.9646 | 44.872/0.9887 | 38.270/0.9684 | 41.705/0.9931 |
| HLFSR-SSR        | 36.349/0.9664 | 38.029/0.9633 | 44.791/0.9885 | 38.872/0.9710 | 40.205/0.9905 |
| LF-DET           | 36.302/0.9662 | 38.034/0.9634 | 44.786/0.9888 | 38.818/0.9708 | 41.261/0.9924 |
| **LF-HAN(ours)** | 36.701/0.9683 | 38.341/0.9652 | 45.195/0.9895 | 39.133/0.9716 | 41.822/0.9932 |

Here is the result for $\times4$ SR.
| Model            | EPFL          | HCInew        | HCIold        | INRIA         | STFgantry     |
| ---------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| RCAN             | 28.132/0.8268 | 29.475/0.7976 | 35.155/0.8960 | 30.206/0.8718 | 28.779/0.8803 |
| resLF            | 29.023/0.8542 | 30.555/0.8413 | 36.579/0.9304 | 31.359/0.8948 | 30.076/0.9148 |
| LFSSR            | 29.450/0.8668 | 30.773/0.8481 | 36.784/0.9334 | 31.837/0.9034 | 30.472/0.9217 |
| LF-ATO           | 29.513/0.8670 | 30.678/0.8452 | 36.944/0.9343 | 32.054/0.9069 | 30.501/0.9233 |
| LF_InterNet      | 29.750/0.8740 | 30.841/0.8518 | 37.159/0.9389 | 32.062/0.9086 | 30.318/0.9193 |
| MEG-Net          | 29.741/0.8730 | 30.962/0.8542 | 37.221/0.9384 | 32.072/0.9079 | 30.655/0.9255 |
| LF-IINet         | 29.978/0.8772 | 31.187/0.8599 | 37.445/0.9419 | 32.369/0.9114 | 31.085/0.9333 |
| DPT              | 29.500/0.8640 | 30.814/0.8486 | 36.770/0.9312 | 31.658/0.8965 | 30.474/0.9195 |
| LFT              | 30.093/0.8795 | 31.269/0.8599 | 37.648/0.9426 | 32.483/0.9124 | 31.689/0.9388 |
| DistgSSR         | 29.783/0.8729 | 31.222/0.8601 | 37.496/0.9416 | 32.044/0.9055 | 31.395/0.9347 |
| LFSSR_SAV        | 30.248/0.8813 | 31.302/0.8598 | 37.484/0.9394 | 32.547/0.9132 | 31.247/0.9338 |
| EPIT             | 30.051/0.8775 | 31.349/0.8629 | 37.550/0.9423 | 32.458/0.9122 | 32.047/0.9430 |
| HLFSR-SSR        | 30.187/0.8815 | 31.390/0.8642 | 37.697/0.9435 | 32.600/0.9137 | 31.461/0.9379 |
| LF-DET           | 30.286/0.8834 | 31.399/0.8635 | 37.779/0.9442 | 32.639/0.9146 | 32.034/0.9435 |
| **LF-HAN(ours)** | 30.857/0.8917 | 31.732/0.8707 | 38.104/0.9476 | 33.168/0.9203 | 32.838/0.9507 |



The result tested with BasicLFSR

$\times 2$

|                            Methods                             | Scale | #Params. |       EPFL        |      HCInew       |      HCIold       |       INRIA       |      STFgantry      |
| :------------------------------------------------------------: | :---: | :------: | :---------------: | :---------------: | :---------------: | :---------------: | :-----------------: |
|                          **Bilinear**                          |  x2   |    --    |   28.480/0.9180   |   30.718/0.9192   |   36.243/0.9709   |   30.134/0.9455   |    29.577/0.9310    |
|                          **Bicubic**                           |  x2   |    --    |   29.740/0.9376   |   31.887/0.9356   |   37.686/0.9785   |   31.331/0.9577   |    31.063/0.9498    |
|                            **VDSR**                            |  x2   |  0.665M  |   32.498/0.9598   |   34.371/0.9561   |   40.606/0.9867   |   34.439/0.9741   |    35.541/0.9789    |
|                            **EDSR**                            |  x2   |  38.62M  |   33.089/0.9629   |   34.828/0.9592   |   41.014/0.9874   |   34.985/0.9764   |    36.296/0.9818    |
|         [**RCAN**](https://github.com/yulunzhang/RCAN)         |  x2   |  15.31M  |   33.159/0.9634   |   35.022/0.9603   |   41.125/0.9875   |   35.046/0.9769   |    36.670/0.9831    |
|          [**resLF**](https://github.com/shuozh/resLF)          |  x2   |  7.982M  |   33.617/0.9706   |   36.685/0.9739   |   43.422/0.9932   |   35.395/0.9804   |    38.354/0.9904    |
|  [**LFSSR**](https://github.com/jingjin25/LFSSR-SAS-PyTorch)   |  x2   |  0.888M  |   33.671/0.9744   |   36.802/0.9749   |   43.811/0.9938   |   35.279/0.9832   |    37.944/0.9898    |
|      [**LF-ATO**](https://github.com/jingjin25/LFSSR-ATO)      |  x2   |  1.216M  |   34.272/0.9757   |   37.244/0.9767   |   44.205/0.9942   |   36.170/0.9842   |    39.636/0.9929    |
| [**LF_InterNet**](https://github.com/YingqianWang/LF-InterNet) |  x2   |  5.040M  |   34.112/0.9760   |   37.170/0.9763   |   44.573/0.9946   |   35.829/0.9843   |    38.435/0.9909    |
|    [**LF-DFnet**](https://github.com/YingqianWang/LF-DFnet)    |  x2   |  3.940M  |   34.513/0.9755   |   37.418/0.9773   |   44.198/0.9941   |   36.416/0.9840   |    39.427/0.9926    |
|        [**MEG-Net**](https://github.com/shuozh/MEG-Net)        |  x2   |  1.693M  |   34.312/0.9773   |   37.424/0.9777   |   44.097/0.9942   |   36.103/0.9849   |    38.767/0.9915    |
|    [**LF-IINet**](https://github.com/GaoshengLiu/LF-IINet)     |  x2   |  4.837M  |   34.732/0.9773   |   37.768/0.9790   |   44.852/0.9948   |   36.566/0.9853   |    39.894/0.9936    |
|          [**DPT**](https://github.com/BITszwang/DPT)           |  x2   |  3.731M  |   34.490/0.9758   |   37.355/0.9771   |   44.302/0.9943   |   36.409/0.9843   |    39.429/0.9926    |
|        [**LFT**](https://github.com/ZhengyuLiang24/LFT)        |  x2   |  1.114M  |   34.804/0.9781   |   37.838/0.9791   |   44.522/0.9945   |   36.594/0.9855   |    40.510/0.9941    |
|    [**DistgSSR**](https://github.com/YingqianWang/DistgSSR)    |  x2   |  3.532M  |   34.809/0.9787   |   37.959/0.9796   |   44.943/0.9949   |   36.586/0.9859   |    40.404/0.9942    |
|   [**LFSSR_SAV**](https://github.com/Joechann0831/SAV_conv)    |  x2   |  1.217M  |   34.616/0.9772   |   37.425/0.9776   |   44.216/0.9942   |   36.364/0.9849   |    38.689/0.9914    |
|       [**EPIT**](https://github.com/ZhengyuLiang24/EPIT)       |  x2   |  1.421M  |   34.826/0.9775   |  38.228/*0.9810*  |  *45.075*/0.9949  |   36.672/0.9853   | **42.166**/*0.9957* |
|    [**HLFSR-SSR**](https://github.com/duongvinh/HLFSR-SSR)     |  x2   |  13.72M  | *35.310*/*0.9800* |  *38.317*/0.9807  |  44.978/*0.9950*  | *37.060*/*0.9867* |    40.849/0.9947    |
|         [**LF-DET**](https://github.com/Congrx/LF-DET)         |  x2   |  1.588M  |   35.262/0.9797   |   38.314/0.9807   |  44.986/*0.9950*  |   36.949/0.9864   |    41.762/0.9955    |
|                        **LF-HAN(ours)**                        |  x2   |  4.219M  | **35.548/0.9811** | **38.618/0.9815** | **45.319/0.9954** | **37.146/0.9869** | *42.119*/**0.9958** |

$\times4$
|                            Methods                             | Scale | #Params. |       EPFL        |       HCInew       |      HCIold       |       INRIA       |     STFgantry     |
| :------------------------------------------------------------: | :---: | :------: | :---------------: | :----------------: | :---------------: | :---------------: | :---------------: |
|                          **Bilinear**                          |  x4   |    --    |   24.567/0.8158   |   27.085/0.8397    |   31.688/0.9256   |   26.226/0.8757   |   25.203/0.8261   |
|                          **Bicubic**                           |  x4   |    --    |   25.264/0.8324   |   27.715/0.8517    |   32.576/0.9344   |   26.952/0.8867   |   26.087/0.8452   |
|                            **VDSR**                            |  x4   |  0.665M  |   27.246/0.8777   |   29.308/0.8823    |   34.810/0.9515   |   29.186/0.9204   |   28.506/0.9009   |
|                            **EDSR**                            |  x4   |  38.89M  |   27.833/0.8854   |   29.591/0.8869    |   35.176/0.9536   |   29.656/0.9257   |   28.703/0.9072   |
|         [**RCAN**](https://github.com/yulunzhang/RCAN)         |  x4   |  15.36M  |   27.907/0.8863   |   29.694/0.8886    |   35.359/0.9548   |   29.805/0.9276   |   29.021/0.9131   |
|          [**resLF**](https://github.com/shuozh/resLF)          |  x4   |  8.646M  |   28.260/0.9035   |   30.723/0.9107    |   36.705/0.9682   |   30.338/0.9412   |   30.191/0.9372   |
|  [**LFSSR**](https://github.com/jingjin25/LFSSR-SAS-PyTorch)   |  x4   |  1.774M  |   28.596/0.9118   |   30.928/0.9145    |   36.907/0.9696   |   30.585/0.9467   |   30.570/0.9426   |
|      [**LF-ATO**](https://github.com/jingjin25/LFSSR-ATO)      |  x4   |  1.364M  |   28.514/0.9115   |   30.880/0.9135    |   36.999/0.9699   |   30.711/0.9484   |   30.607/0.9430   |
| [**LF_InterNet**](https://github.com/YingqianWang/LF-InterNet) |  x4   |  5.483M  |   28.812/0.9162   |   30.961/0.9161    |   37.150/0.9716   |   30.777/0.9491   |   30.365/0.9409   |
|    [**LF-DFnet**](https://github.com/YingqianWang/LF-DFnet)    |  x4   |  3.990M  |   28.774/0.9165   |   31.234/0.9196    |   37.321/0.9718   |   30.826/0.9503   |   31.147/0.9494   |
|        [**MEG-Net**](https://github.com/shuozh/MEG-Net)        |  x4   |  1.775M  |   28.749/0.9160   |   31.103/0.9177    |   37.287/0.9716   |   30.674/0.9490   |   30.771/0.9453   |
|    [**LF-IINet**](https://github.com/GaoshengLiu/LF-IINet)     |  x4   |  4.886M  |   29.038/0.9188   |   31.331/0.9208    |   37.620/0.9734   |   31.034/0.9515   |   31.261/0.9502   |
|          [**DPT**](https://github.com/BITszwang/DPT)           |  x4   |  3.778M  |   28.939/0.9170   |   31.196/0.9188    |   37.412/0.9721   |   30.964/0.9503   |   31.150/0.9488   |
|        [**LFT**](https://github.com/ZhengyuLiang24/LFT)        |  x4   |  1.163M  |   29.255/0.9210   |   31.462/0.9218    |   37.630/0.9735   |   31.205/0.9524   |   31.860/0.9548   |
|    [**DistgSSR**](https://github.com/YingqianWang/DistgSSR)    |  x4   |  3.582M  |   28.992/0.9195   |   31.380/0.9217    |   37.563/0.9732   |   30.994/0.9519   |   31.649/0.9535   |
|   [**LFSSR_SAV**](https://github.com/Joechann0831/SAV_conv)    |  x4   |  1.543M  | 29.368*/*0.9223*  |   31.450/0.9217    |   37.497/0.9721   |   31.270/0.9531   |   31.362/0.9505   |
|       [**EPIT**](https://github.com/ZhengyuLiang24/EPIT)       |  x4   |  1.470M  |   29.339/0.9197   |   31.511/0.9231    |   37.677/0.9737   |   31.372/0.9526   |  *32.179*/0.9571  |
|    [**HLFSR-SSR**](https://github.com/duongvinh/HLFSR-SSR)     |  x4   |  13.87M  |   29.196/0.9222   | *31.571**/*0.9238* |  37.776*/*0.9742  |  31.241/*0.9534*  |   31.641/0.9537   |
|         [**LF-DET**](https://github.com/Congrx/LF-DET)         |  x4   |  1.687M  |  *29.473*/0.9230  |   31.558*/0.9235   | *37.843*/*0.9744* | *31.389*/*0.9534* |  32.139/*0.9573*  |
|                        **LF-HAN(ours)**                        |  x4   |  4.260M  | **30.233/0.9296** | **31.902/0.9276**  | **38.135/0.9759** | **32.291/0.9576** | **32.953/0.9621** |



# References

+ [BasicSR](https://github.com/XPixelGroup/BasicSR/tree/master/basicsr)
+ [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)