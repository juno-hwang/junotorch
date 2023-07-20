# junotorch
Custom pytorch modules that I use often.

[황준오] [오후 3:41] https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nucleotide&id=MN996528%2C1&rettype=fasta&tool=biopython
[황준오] [오후 3:41] https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
[황준오] [오후 3:41] https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcqi


## Install

```
pip install -U git+https://github.com/juno-hwang/junotorch.git
```

## Usage (DDPM v1)

### DDPM
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

ddpm.fit('data_folder/', grad_accum=2)
```

### DDPMUpsampler
```python
from junotorch.vision import *
from ddpm import *

backbone_to256 = EfficientUNetUpsampler(
    dim = 256,
    image_size = 256,
    small_image_size = 64,
    n_downsample = 4,
    mid_depth = 3,
    n_resblock = 1,
    base_dim = 32,
    T = 200
)

ddpm256 = DDPMUpsampler(
    backbone = backbone_to256,
    device = 'cuda',
    batch_size = 32,
    result_folder = 'result_folder/',
    loss_type = 'l2'
)

ddpm256.fit('data_folder/', grad_accum=1)
```
