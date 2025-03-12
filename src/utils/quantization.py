import torch
from transformers import BitsAndBytesConfig


def bitsandbytes_8bit_config():
    return BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
