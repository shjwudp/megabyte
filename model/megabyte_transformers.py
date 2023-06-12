import torch
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput


class MegabyteConfig(PretrainedConfig):
    model_type = "megabyte"

    def __init__(
        self,
        V=512,
        P=8,
        D_G=128,
        D_L=256,
        T_MAX=2048,
        g_nheads=16,
        l_nheads=4,
        g_nlayers=12,
        l_nlayers=6,
        initializer_range=0.02,
        pad_id=257,
        eos_token_id=258,
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
        model = cls(config)
        model.config = config
        model.inner_model = native_model
        return model

    def forward(
        self,
        input_ids,
        return_dict = None,
        **deprecated_arguments,
    ):
        output = self.inner_model(input_ids)
        if not return_dict:
            return output.loss
        
        return CausalLMOutput(
            loss=output.loss,
            logits=output.lm_logits,
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
    def __init__(self, eos_token_id=258):
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
