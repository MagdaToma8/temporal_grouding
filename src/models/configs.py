from typing import Dict, Any, Optional

from transformers import (
    AutoProcessor,
    AutoModel,
    Blip2ForConditionalGeneration,
    Blip2ForImageTextRetrieval,
    CLIPModel,
    LlavaForConditionalGeneration,
)

from src.models.blip2 import (
    BLIP2Wrapper,
    BLIP2MatchingWrapper,
    BLIP2EmbeddingsWrapper,
    Blip2ForImageTextEmbeddings
)
from src.models.clip import CLIPWrapper
from src.models.llava import LLaVaWrapper
from src.models.jinaclip import JinaVLMWrapper, JinaProcessor

CONFIGS = {
    "blip2": {
        "model_id": "Salesforce/blip2-flan-t5-xl",
        "model_class": Blip2ForConditionalGeneration,
        "processor_class": AutoProcessor,
        "wrapper_class": BLIP2Wrapper,
        "needs_processor": False,
        "model_type": "generation"
    },
    "blip2-embeddings": {
        "model_id": "Salesforce/blip2-itm-vit-g",
        "model_class": Blip2ForImageTextEmbeddings,
        "processor_class": AutoProcessor,
        "wrapper_class": BLIP2EmbeddingsWrapper,
        "needs_processor": False,
        "model_type": "embeddings"
    },
    "blip2-matching": {
        "model_id": "Salesforce/blip2-itm-vit-g",
        "model_class": Blip2ForImageTextRetrieval,
        "processor_class": AutoProcessor,
        "wrapper_class": BLIP2MatchingWrapper,
        "needs_processor": False,
        "model_type": "embeddings"
    },
    "clip": {
        "model_id": "openai/clip-vit-base-patch32",
        "model_class": CLIPModel,
        "processor_class": AutoProcessor,
        "wrapper_class": CLIPWrapper,
    },
    "llava": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "model_class": LlavaForConditionalGeneration,
        "processor_class": AutoProcessor,
        "wrapper_class": LLaVaWrapper,
    },
    "jinaclip": {
        "model_id": "jinaai/jina-clip-v2",
        "model_class": AutoModel,
        "processor_class": JinaProcessor,
        "wrapper_class": JinaVLMWrapper,
    }
}

def get_model_config(
        model_family: str,
        model_id: Optional[str] = None,
) -> Dict[str, Any]:
    config = CONFIGS.get(model_family, {})
    if not config:
        raise ValueError(f"Model config is not parsed. Plase use model_family from {list(CONFIGS.keys())}")
    config["model_id"] = model_id
    return config
