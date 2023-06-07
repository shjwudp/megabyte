from model.megabyte import MegabyteConfig as NativeMegabyteConfig, Megabyte

import torch
import copy

from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput


class MegabyteConfig(PretrainedConfig):
    model_type = "megabyte"

    def __init__(
        self,
        V,
        P,
        D_G,
        D_L,
        T_MAX,
        g_nheads,
        g_nlayers,
        l_nheads,
        l_nlayers,
        initializer_range,
        pad_id,
        **kwargs,
    ):
        self.V = V
        self.P = P
        self.D_G = D_G
        self.D_L = D_L
        self.T_MAX = T_MAX
        self.g_nheads = g_nheads
        self.g_nlayers = g_nlayers
        self.l_nheads = l_nheads
        self.l_nlayers = l_nlayers
        self.initializer_range = initializer_range
        self.pad_id = pad_id
        super().__init__(**kwargs)


class MegabyteLMHeadModel(PreTrainedModel, GenerationMixin):
    config_class = MegabyteConfig

    def __init__(self, config):
        super().__init__(config)
        native_config = NativeMegabyteConfig(
            V=config.V,
            P=config.P,
            D_G=config.D_G,
            D_L=config.D_L,
            T_MAX=config.T_MAX,
            g_nheads=config.g_nheads,
            g_nlayers=config.g_nlayers,
            l_nheads=config.l_nheads,
            l_nlayers=config.l_nlayers,
            initializer_range=config.initializer_range,
            pad_id=config.pad_id,
        )
        self.model = Megabyte(native_config)

    # TODO: Rewrite the forward function to be compatible with GenerationMixin.
    def forward(
        self,
        input_ids,
        return_dict = None,
        **deprecated_arguments,
    ):
        loss = self.model(input_ids)
        if not return_dict:
            return loss
        
        return CausalLMOutput(
            loss=loss,
            lm_logits=None,
            hidden_states=None,
            attentions=None,
        )
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs) -> dict:
        return {"input_ids": input_ids}


class MegabyteTokenizer:
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        
    def __call__(self, text, return_tensors="pt"):
        tokens = torch.frombuffer(copy.deepcopy(text.encode("utf-8")), dtype=torch.uint8).to(torch.int64)
        tokens = tokens.reshape(1, tokens.numel())
        return {"input_ids": tokens}

if __name__ == "__main__":
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
    model = MegabyteLMHeadModel(config)
    input_ids = torch.randint(0, 255, (B, T))
    loss = model(input_ids)
    loss.backward()

    print(loss.norm())
