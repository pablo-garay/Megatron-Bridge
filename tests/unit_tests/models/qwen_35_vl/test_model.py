# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from unittest.mock import patch, MagicMock

from megatron.bridge.models.qwen_35_vl.model import Qwen35VLModel
from megatron.bridge.models.qwen_35_vl.transformer_config import Qwen3VLTransformerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

class TestQwen3VLModel:
    """Test cases for Qwen3VL Model initialization."""

    def init_parallel_state(self, tp_size=1, cp_size=1, pp_size=1, ep_size=1, vpp_size=None,etp_size=None):
        """Initialize parallel state for testing."""
        
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())

        
        parallel_state.initialize_model_parallel(
           tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vpp_size,
            context_parallel_size=cp_size,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=etp_size,
        )

        model_parallel_cuda_manual_seed(123)

    def get_vision_transformer_config(self):
        """Create a vision transformer config for testing.
        
        Returns:
            TransformerConfig: Configuration for the vision model.
        """
        return TransformerConfig(
            num_layers=2, # not used, use HF model
            hidden_size=128,   # not used, use HF model
            num_attention_heads=8,  # not used, use HF model
            bf16=False,
            use_cpu_initialization=True, # not used, use HF model

        )

    def get_language_transformer_config(self):
        """Create a language transformer config for testing.
        
        Returns:
            Qwen3VLTransformerConfig: Configuration for the language model.
        """
        return Qwen3VLTransformerConfig(
            # Basic transformer parameters (required)
            num_layers=4,  # Small for testing
            hidden_size=256,  # Small for testing
            num_attention_heads=8,
            vocab_size=151936,  # Qwen3 vocab size
            language_max_sequence_length=2048,
        )

    def get_language_model_layer_spec(self):
        """Create a GPT layer spec for the language model.
        
        Returns:
            ModuleSpec: Layer specification for transformer layers.
        """
        return get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,  # No MoE for basic test
            moe_grouped_gemm=False,
            qk_layernorm=False,
            fp8=False,
        )


    def test_qwen3_vl_model_init(self):
        """Test Qwen3VL model initialization."""
        
        self.init_parallel_state()

        
        vision_transformer_config = self.get_vision_transformer_config()
        language_transformer_config = self.get_language_transformer_config()
        language_model_layer_spec = self.get_language_model_layer_spec()
        
        model = Qwen35VLModel(
            vision_transformer_config=vision_transformer_config,
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=language_model_layer_spec,
            parallel_output=True,
            pre_process=True,
            post_process=True,
            add_encoder=True,
            add_decoder=True,
        )


if __name__ == "__main__":
    test_qwen3_vl_model = TestQwen3VLModel()
    test_qwen3_vl_model.test_qwen3_vl_model_init()

