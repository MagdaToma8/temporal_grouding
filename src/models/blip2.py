from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple

from transformers import (
    AddedToken,
    AutoProcessor,
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2ForImageTextRetrieval,
    Blip2PreTrainedModel,
    Blip2QFormerModel,
    Blip2VisionModel
)
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2ImageTextMatchingModelOutput,
    Blip2TextEmbeddings
)
import torch
from torch import nn

from src.models.vlm_wrapper import VLMWrapper


@dataclass
class BLIP2Wrapper(VLMWrapper):
    model: Any = field(
        default_factory=lambda: Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/flan-t5-xl",
            device_map={"": 0},
            torch_dtype=torch.float16
        )
    )
    processor: Any = field(
        default_factory=lambda: AutoProcessor.from_pretrained(
            "Salesforce/flan-t5-xl"
        )
    )

    def process_inputs(self, **kwargs) -> Dict[str, Any]:
        required_keys = {'image'}
        if not required_keys.issubset(kwargs.keys()):
            raise ValueError(f"Missing required arguments: {required_keys - set(kwargs.keys())}")
        print(kwargs)
        prompt = self._generate_qa_prompt(
            kwargs.get(
                'prompt',
                ["What is in the picture?"] * len(kwargs['image'])
            )
        )
        return self.processor(kwargs['image'], text=prompt, return_tensors="pt", padding=True)

    def decode(self, generated_ids: Any, **kwargs) -> str:
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        return decoded[0].strip() if len(decoded) == 1 else decoded

    def generate(self, inputs: Dict[str, Any], **kwargs) -> Any:
        max_len = kwargs.get('max_len', 80)
        return self.model.generate(**inputs, max_new_tokens=max_len)

    def _generate_qa_prompt(self, prompt: Union[str, List[str]]):
        if isinstance(prompt, str):
            return f"Question: {prompt} Answer:"
        else:
            return [f"Question: {p} Answer:" for p in prompt]

    def get_embeddings(self, *args, **kwargs) -> Any:
        raise NotImplementedError("`BLIP2Wrapper` does not support embeddings")


@dataclass
class BLIP2MatchingWrapper(VLMWrapper):
    model: Any = field(
        default_factory=lambda: Blip2ForImageTextRetrieval.from_pretrained(
            "Salesforce/blip2-itm-vit-g",
            torch_dtype=torch.float16
        )
    )
    processor: Any = field(
        default_factory=lambda: AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")
    )

    def process_inputs(self, **kwargs) -> Dict[str, Any]:
        required_keys = {'image', 'prompt'}
        if not required_keys.issubset(kwargs.keys()):
            raise ValueError(f"Missing required arguments: {required_keys - set(kwargs.keys())}")
        return self.processor(images=kwargs['image'], text=kwargs['prompt'], return_tensors="pt").to(self.model.device)

    def get_embeddings(self, inputs: Dict[str, Any], use_image_text_matching: bool = False, **kwargs) -> Any:
        """
        Get the outputs of the retrieval model.

        Args:
            inputs (Dict[str, Any]): The inputs to the model.
            use_image_text_matching (bool): Whether to use the image-text matching model.
                If True, the model will return the image-text matching scores along with the embeddings.
                Note! Attention mask in this case will be all ones.
                That means the model will attend to all tokens (text and image).
                https://github.com/huggingface/transformers/blob/d7188ba600e36d3fd191b12e19f1b3bb81a8404f/src/transformers/models/blip_2/modeling_blip_2.py#L2470
                If False, the model will return the embeddings and cosine similarity scores.
        """
        return self.model(**inputs, use_image_text_matching=use_image_text_matching)

    def generate(self, inputs: Dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("`BLIP2MatchingWrapper` does not support generation")

    def decode(self, *args, **kwargs) -> Any:
        raise NotImplementedError("`BLIP2MatchingWrapper` does not support decoding")

@dataclass
class BLIP2EmbeddingsWrapper(VLMWrapper):
    model: Any = field(
        default_factory=lambda: Blip2ForImageTextEmbeddings.from_pretrained(
            "Salesforce/blip2-itm-vit-g",
            torch_dtype=torch.float16
        )
    )
    processor: Any = field(
        default_factory=lambda: AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")
    )

    # def __post_init__(self):
    #     Suggestion from https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042
    #     Causes significant decline in retrieval metrics.
    #     self.processor.num_query_tokens = self.model.config.num_query_tokens
    #     image_token = AddedToken("<image>", normalized=False, special=True)
    #     self.processor.tokenizer.add_tokens([image_token], special_tokens=True)

    #     self.model.resize_token_embeddings(len(self.processor.tokenizer), pad_to_multiple_of=64) # pad for efficient computation
    #     self.model.config.image_token_index = len(self.processor.tokenizer) - 1

    def process_inputs(self, **kwargs) -> Dict[str, Any]:
        return self.processor(
            images=kwargs.get('image', None),
            text=kwargs.get('prompt', None),
            return_tensors="pt"
        ).to(self.model.device)

    def get_embeddings(self, inputs: Dict[str, Any], **kwargs) -> Any:
        outputs = self.model(**inputs)
        return {
            "image_embeds": outputs.image_embeds,
            "text_embeds": outputs.text_embeds,
            "logits_per_image": outputs.logits_per_image,
            "logits_per_text": outputs.logits_per_text,
            "vision_model_output": outputs.vision_model_output.last_hidden_state,
            "text_model_output": outputs.text_model_output.last_hidden_state,
        }

    def generate(self, *args, **kwargs) -> Any:
        raise NotImplementedError("`BLIP2EmbeddingsWrapper` does not support generation")

    def decode(self, *args, **kwargs) -> Any:
        raise NotImplementedError("`BLIP2EmbeddingsWrapper` does not support decoding")


class Blip2ForImageTextEmbeddings(Blip2PreTrainedModel):
    """
    BLIP2 model for generating image and text embeddings.
    This is an adaptation of the original Blip2ForImageTextRetrieval:
    https://github.com/huggingface/transformers/blob/d7188ba600e36d3fd191b12e19f1b3bb81a8404f/src/transformers/models/blip_2/modeling_blip_2.py#L2363

    Changes:
    - Image-text matching head is not used.
    - Model can be used to generate image and text embeddings or *one of them*.
    """
    main_input_name = "pixel_values"
    _keep_in_fp32_modules = []

    def __init__(self, config: Blip2Config):
        super().__init__(config)

        self.vision_model = Blip2VisionModel(config.vision_config)

        # print(f"Q-former uses {config.num_query_tokens} query tokens")
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))

        self.embeddings = Blip2TextEmbeddings(config.qformer_config)
        self.qformer = Blip2QFormerModel(config.qformer_config)

        # vision projection layer
        self.vision_projection = nn.Linear(config.qformer_config.hidden_size, config.image_text_hidden_size)

        # text projection layer
        self.text_projection = nn.Linear(config.qformer_config.hidden_size, config.image_text_hidden_size)

        # image text matching head
        self.itm_head = nn.Linear(config.qformer_config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ImageTextMatchingModelOutput]:
        r"""
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pixel_values is None and input_ids is None:
            raise ValueError("Either pixel_values or input_ids must be provided")

        image_embeds = None
        text_embeds = None
        vision_outputs = None
        text_outputs = None
        logits_per_image = None
        logits_per_text = None

        if pixel_values is not None:
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeds = vision_outputs[0]
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=return_dict,
            )
            image_embeds = query_outputs[0] if not return_dict else query_outputs.last_hidden_state

        if input_ids is not None:
            query_embeds = self.embeddings(
                input_ids=input_ids,
            )
            text_outputs = self.qformer(
                query_embeds=query_embeds,
                query_length=0,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )
            question_embeds = text_outputs[0] if not return_dict else text_outputs.last_hidden_state

        if pixel_values is not None:
            image_embeds = nn.functional.normalize(self.vision_projection(image_embeds), dim=-1)

        if input_ids is not None:
            text_embeds = nn.functional.normalize(self.text_projection(question_embeds[:, 0, :]), dim=-1)

        if pixel_values is not None and input_ids is not None:
            # cosine similarity as logits
            logits_per_image = torch.matmul(image_embeds, text_embeds.t())
            logits_per_image, _ = logits_per_image.max(dim=1)
            logits_per_text = logits_per_image.t()

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return output

        return Blip2ImageTextMatchingModelOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
