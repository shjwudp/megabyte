# Try it

```python
import torch
import torch.nn.functional as F
from einops import rearrange
from megabyte import MegabyteConfig, Megabyte

V = 256
P = 4
D_G = 512
D_L = 128
T = 1024
B = 2
K = T//P

config = MegabyteConfig(
    V=V,
    P=P,
    D_G=D_G,
    D_L=D_L,
    T_MAX=T,
    pad_id=-100,
    initializer_range=0.02,
    g_nlayers=4,
    g_nheads=16,
    l_nlayers=2,
    l_nheads=8,
)

megabyte = Megabyte(config)
input_ids = torch.randint(0, 255, (B, T))
lm_logits = megabyte(input_ids)
loss = F.cross_entropy(
    rearrange(lm_logits, "B K P V -> (B K) V P", B=B, K=K, P=P, V=V),
    rearrange(input_ids, "... (K P) -> (... K) P", K=K, P=P),
)
loss.backward()

print(lm_logits.shape, loss.norm())
```
