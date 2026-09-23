"""ModelConfig: the recipe card, read from config.json."""
import json
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int          # width of the residual stream (896 for Qwen2.5-0.5B)
    intermediate_size: int    # width of the MLP's middle layer (4864)
    num_hidden_layers: int    # number of stations (24)
    num_attention_heads: int  # query heads (14)
    num_key_value_heads: int  # key/value heads (2)  -> GQA
    rms_norm_eps: float       # tiny constant inside RMSNorm (1e-6)
    rope_theta: float         # RoPE base frequency (1e6)
    tie_word_embeddings: bool # reuse the embedding table as the output head

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads   # 896 // 14 = 64

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        c = json.load(open(path))
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"],
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=c["rope_theta"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
        )
