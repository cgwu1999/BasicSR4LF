Our project is based on BasicSR

`BasicSR/basicsr/models/lfsr_model.py`

`BasicSR/basicsr/archs/*`

`BasicSR/basicsr/data/LF_dataset.py`

`/BasicSR/options/*`

How to add a model from BasicLFSR.
1. add a new file under BasicSR/basicsr/models/ with the name of your model, e.g. lfssr_arch.py
2.  import packages
```python
import from basicsr.utils.registry import ARCH_REGISTRY
from .lf_utils import LFDataWarp
import argparse
```
3. add `@ARCH_REGISTRY.register()` before the classname
4. replace the classname from `get_model` to your model name     
5. `args=argparse.Namespace(**args)` make the `args` namespace
6. add a function warp `@LFDataWarp` for the `forward` function
7. add a yml config file under `/BasicSR/options/` the main difference 
```yml
network_g:
type: DistgSSR
    args:
        scale_factor: 2
        angRes_in: 5
path:
  param_key_g: state_dict
```

The difference with the BasicLFSR.
|             | Our         | BasicLFSR |
| ----------- | ----------- | --------- |
| test patch  | non-overlap | overlap   |
| crop border | 2           | 0         |
