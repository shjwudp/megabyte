from model.megabyte import MegabyteConfig as NativeMegabyteConfig, Megabyte

import torch
import torch.nn.functional as F

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
        eos_token_id,
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
        self.eos_token_id = eos_token_id
        self.bos_token_id = eos_token_id
        self.is_encoder_decoder = False
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
            eos_id=config.eos_token_id,
        )
        self.config = config
        self.model = Megabyte(native_config)

    @classmethod
    def from_native_megabyte(cls, native_model):
        native_config = native_model.config
        config = MegabyteConfig(
            V=native_config.V,
            P=native_config.P,
            D_G=native_config.D_G,
            D_L=native_config.D_L,
            T_MAX=native_config.T_MAX,
            g_nheads=native_config.g_nheads,
            g_nlayers=native_config.g_nlayers,
            l_nheads=native_config.l_nheads,
            l_nlayers=native_config.l_nlayers,
            initializer_range=native_config.initializer_range,
            pad_id=native_config.pad_id,
            eos_token_id=native_config.eos_id,
        )
        return cls(config)

    def forward(
        self,
        input_ids,
        return_dict = None,
        **deprecated_arguments,
    ):
        loss, lm_logits = self.model(input_ids)
        if not return_dict:
            return loss
        
        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs) -> dict:
        _, T = input_ids.shape
        P = self.config.P

        # Add a character at the end as a placeholder, and padding input_ids length to an integer multiple of P.
        input_ids = F.pad(input_ids, ((P-1)-T%P, 1), value=self.config.pad_id)

        return {"input_ids": input_ids}


class MegabyteTokenizer:
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        
    def __call__(self, text, return_tensors="pt"):
        tokens = torch.frombuffer(bytearray(text.encode("utf-8")), dtype=torch.uint8).to(torch.int64)
        tokens = tokens.reshape(1, tokens.numel())
        return {"input_ids": tokens}
    
    def decode(self, ids):
        texts = []
        for id_list in ids.tolist():
            line_ids = filter(lambda x: 0<=x and x<256, id_list)
            text = bytearray(list(line_ids)).decode("utf-8")
            texts.append(text)

        return texts

if __name__ == "__main__":
    V = 512
    P = 4
    D_G = 512
    D_L = 128
    T = 1024
    B = 2
    K = T//P
    PAD_ID = 257
    EOS_ID = 258

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
        pad_id=PAD_ID,
        eos_token_id=EOS_ID,
    )
    tokenizer = MegabyteTokenizer(eos_token_id=EOS_ID)
    model = MegabyteLMHeadModel(config)
    
    inputs = tokenizer("Today is", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)

    texts = tokenizer.decode(outputs.sequences)
    print(texts)
