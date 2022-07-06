# junotorch
Custom pytorch modules that I use often.

## Install

```
pip install -U git+https://github.com/juno-hwang/junotorch.git
```

## Usage (DDPM v1)

```python
from junotorch.vision import *
from junotorch.ddpm import *

backbone = EfficientUNet(
    dim = 512,
    image_size = 64,
    n_downsample = 3,
    mid_depth = 4,
    n_resblock = 1,
    T = 250
)

ddpm = DDPM(
    backbone = backbone,
    device = 'cuda',
    batch_size = 64,
    loss_type='l2',
    result_folder = 'result_folder/'
)
!nvidia-smi
ddpm.fit('data_folder/', grad_accum=2)
```