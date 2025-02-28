import pytest

import torch
from pytorch_lightning.utilities.model_summary import ModelSummary

from src.models.attentive_summarizer import (
    AttentiveSummarizer,
    AlignmentAttentiveSummarizer,
    CosineSimilarityLoss
)
from src.models.configs import get_model_config


@pytest.fixture
def query_features_inputs():
    return torch.randn(8, 20, 512)

@pytest.fixture
def query_input():
    return [
        "A photo of a big white cat",
        "A photo of a small black dog",
        "A photo of a yellow bird",
        "A photo of a fish",
        "A photo of a big white cat",
        "A photo of a small black dog",
        "A photo of a yellow bird",
        "A photo of a brown bear",
    ]

@pytest.fixture
def text_features_inputs():
    return torch.randn(8, 25, 512)

@pytest.fixture
def text_inputs():
    return [
        "tail fur white",
        "tail fur black",
        "wing yellow",
        "fish water",
        "tail fur white",
        "tail fur black",
        "wing yellow",
        "brown big",
    ]

@pytest.fixture
def clip_vision_features_inputs():
    return torch.randn(8, 50, 768)

@pytest.fixture
def vision_inputs():
    return torch.rand(8, 3, 224, 224)

@pytest.fixture
def gt_text():
    return [
        "A picture of a big white cat in the snow",
        "A picture of a small black dog on the beach",
        "A picture of a yellow bird in the sky",
        "A picture of a fish in the water",
        "A picture of a big white cat in the snow",
        "A picture of a small black dog on the beach",
        "A picture of a yellow bird in the sky",
        "A picture of a brown bear in the forest",
    ]

@pytest.fixture
def gt_features():
    return torch.randn(8, 512)

@pytest.fixture
def pooler_config():
    return {
        "embed_dim": 768,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "depth": 1,
    }

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _init_clip_model(device):
    clip_config = get_model_config(
        "clip",
        "openai/clip-vit-base-patch32"
    )
    model = clip_config["model_class"].from_pretrained(clip_config["model_id"]).to(device)
    processor = clip_config["processor_class"].from_pretrained(clip_config["model_id"])
    vlm_wrapper = clip_config["wrapper_class"](model=model, processor=processor)
    return vlm_wrapper

def test_cosine_similarity_loss():
    query_features_inputs = torch.randn(8, 512)
    gt_features = torch.randn(8, 512)
    loss = CosineSimilarityLoss()(query_features_inputs, gt_features)
    assert 0 < loss.item() < 2

def test_attentive_summarizer_text(
    query_input,
    text_inputs,
    gt_text,
    pooler_config,
    vision_inputs,
    device
):
    vlm_wrapper = _init_clip_model(device)

    # text and vision encoder dimensions in CLIP
    global_embeddings_vision = True
    text_dim = 512
    vision_dim = 512 if global_embeddings_vision else 768

    summarizer = AttentiveSummarizer(
        pooler_config=pooler_config,
        text_dim=text_dim,
        vision_dim=vision_dim,
        vlm_wrapper=vlm_wrapper,
        global_embeddings_vision=global_embeddings_vision
    ).to(device)

    tokenized_query = vlm_wrapper.process_inputs(
        **{
            "image": torch.rand(len(query_input), 3, 224, 224),
            "prompt": query_input
        }
    ).to(device)

    tokenized_text = vlm_wrapper.process_inputs(
        **{
            "image": torch.rand(len(text_inputs), 3, 224, 224),
            "prompt": text_inputs
        }
    ).to(device)

    processed_vision = vlm_wrapper.process_inputs(
        **{
            "image": vision_inputs,
            "prompt": [""] * len(vision_inputs)
        }
    ).to(device)

    tokenized_gt = vlm_wrapper.process_inputs(
        **{
            "image": torch.rand(len(gt_text), 3, 224, 224),
            "prompt": gt_text
        }
    ).to(device)

    forward_output = summarizer(
        q=tokenized_query,
        text_inputs=tokenized_text,
        vision_inputs=processed_vision
    )

    assert forward_output.shape == (8, text_dim)

    contrastive_summarizer = AlignmentAttentiveSummarizer(
        summarizer=summarizer,
        pooler_config=pooler_config,
        temperature=0.1
    ).to(device)

    contrastive_forward_output = contrastive_summarizer(
        q=tokenized_query,
        gt=tokenized_gt,
        text_inputs=tokenized_text,
        vision_inputs=processed_vision
    )

    assert contrastive_forward_output[0].shape == (8, text_dim)
    assert contrastive_forward_output[1].shape == (8, text_dim)

    contrastive_loss = contrastive_summarizer.training_step(
        batch={
            "q": tokenized_query,
            "gt": tokenized_gt,
            "text_inputs": tokenized_text,
            "vision_inputs": processed_vision
        },
        batch_idx=0
    )

    assert contrastive_loss.item() > 0

    summary = ModelSummary(contrastive_summarizer, max_depth=2)
    print(summary)
