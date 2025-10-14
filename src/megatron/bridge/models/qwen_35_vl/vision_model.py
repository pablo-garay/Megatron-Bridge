from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig 
from transformers import Qwen3VLMoeVisionModel


class Qwen3VLVisionModel(MegatronModule):

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        
        bf16 = config.bf16

        self.vision_model  = Qwen3VLMoeVisionModel.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        ).to('cuda')

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            

        """
        image_embeds, deepstack_feature_lists = self.vision_model(pixel_values, grid_thw, **kwargs)
        # split_sizes = (grid_thw.prod(-1) // self.vision_model.spatial_merge_size**2).tolist()
        # image_embeds = torch.split(image_embeds, split_sizes)
        # image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds, deepstack_feature_lists