from typing import Dict, Any

from src.models.vlm_wrapper import VLMWrapper


class VLMInference:
    def __init__(self, model_wrapper: VLMWrapper) -> None:
        self.model_wrapper = model_wrapper

    def run_inference(self, input_dict: Dict[str, Any]):
        inputs = self.model_wrapper.process_inputs(**input_dict)

        print("Generating response..")
        inputs.to(self.model_wrapper.model.device)
        generated_ids = self.model_wrapper.generate(inputs, **input_dict)
        return self.model_wrapper.decode(generated_ids)

    def get_embeddings(self, input_dict: Dict[str, Any]):
        inputs = self.model_wrapper.process_inputs(**input_dict)
        return self.model_wrapper.get_embeddings(inputs, **input_dict)
