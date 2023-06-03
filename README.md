# Try it

```python
import torch
import torch.nn.functional as F
from einops import rearrange
from megabyte import MegabyteConfig, Megabyte

V = 512
P = 4
D_G = 512
D_L = 128
T = 1024
B = 2
K = T//P
PAD_ID = 257

config = MegabyteConfig(
    V=V,
    P=P,
    D_G=D_G,
    D_L=D_L,
    T_MAX=T,
    initializer_range=0.02,
    g_nlayers=4,
    g_nheads=16,
    l_nlayers=2,
    l_nheads=8,
    pad_id=PAD_ID
)
megabyte = Megabyte(config)
input_ids = torch.randint(0, 255, (B, T))
loss = megabyte(input_ids)
loss.backward()

print(loss.norm())
```
