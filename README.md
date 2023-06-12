# Megabyte

This repository implements [MEGABYTE](https://arxiv.org/abs/2305.07185) with pytorch, and tries to explore the best practice of Megabyte architecture. The original architecture described in the paper is implemented in [megabyte.py](./model/megabyte.py), and the best practices are implemented in [megabyte_in_action.py](./model/megabyte_in_action.py).

Megabyte is a new architecture that overcomes the performance defects of bytes end-to-end training and makes tokenization-free autoregressive sequence modeling possible.

## Megabyte in autoregressive training

```python
import torch
import torch.nn.functional as F
from einops import rearrange
from model import MegabyteConfig, Megabyte

V = 512         # vocabulary size, input bytes have 256 characters, and the extra 256 are reserved for special tokens.
P = 4           # patch size
D_G = 512       # global model dimension
D_L = 128       # local model dimension
T = 1024        # sequence length
B = 2           # batch size
K = T//P        # number of patches
PAD_ID = 257    # padding token id
EOS_ID = 258    # end of sequence token id

config = MegabyteConfig(
    V=V,
    P=P,
    D_G=D_G,
    D_L=D_L,
    T_MAX=T,
    initializer_range=0.02, # Parameter initialization value range
    g_nlayers=4,            # number of global model layers
    g_nheads=32,            # number of global model attention heads
    l_nlayers=2,            # number of local model attention layers
    l_nheads=2,             # number of local model attention heads
    pad_id=PAD_ID,
    eos_id=EOS_ID,
)
megabyte = Megabyte(config)
input_ids = torch.randint(0, 255, (B, T))
# Autoregressive learning, megabyte will learn from the inputs input[:, :-1], labels input[:, :], and learn to predict the next token.
loss = megabyte(input_ids, return_loss=True).loss
loss.backward()

print(loss.norm())
```

## Megabyte in generation

```python
...
from model.megabyte_transformers import MegabyteLMHeadModel, MegabyteTokenizer
lm_head_megabyte = MegabyteLMHeadModel.from_native_megabyte(megabyte)
tokenizer = MegabyteTokenizer(
    eos_token_id=lm_head_megabyte.config.eos_token_id,
)

inputs = tokenizer("Today is", return_tensors="pt")
outputs = lm_head_megabyte.generate(
    **inputs,
    max_new_tokens=5,
    return_dict_in_generate=True,
    output_scores=True,
)

texts = tokenizer.decode(outputs.sequences)
print(texts)
```

## Benchmark

You can use the [benchmark.py](https://github.com/shjwudp/megabyte/blob/main/benchmark.py) script for Megabyte's performance measurement. The following table compares the training of Megabyte and GPT2 on wikitext-103-v1 with the same parameter scale.

| model                   | # of parameters | training speed (KB/s) | GPU Memory Allocated % | eval loss | eval loss bpc |
| :---------------------- | :-------------- | :-------------------- | :--------------------- | :-------- | :------------ |
| gpt2                    | 124439808       | 143.68                | 42.97                  | 5.06      | 1.10          |
| megabyte(P=8)           | 132278528       | 189.13                | 17.62                  | 1.13      | 1.13          |
| megabyte_in_action(P=8) | 132,573,696     | 188.35                | 17.68                  | 1.10      | 1.10          |

## Citation

```text
@misc{yu2023megabyte,
      title={MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers}, 
      author={Lili Yu and Dániel Simig and Colin Flaherty and Armen Aghajanyan and Luke Zettlemoyer and Mike Lewis},
      year={2023},
      eprint={2305.07185},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
