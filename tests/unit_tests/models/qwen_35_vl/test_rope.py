'''
uv run ./tests/unit_tests/models/qwen_35_vl/test_rope.py
'''

import torch
from megatron.bridge.models.qwen_35_vl.rope import Qwen3VLTextRotaryEmbedding
from transformers import Qwen3VLMoeTextConfig
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextRotaryEmbedding
import logging



class TestQwen3VLTextRotaryEmbedding:
    def test_qwen3_vl_text_rotary_embedding(self):
        hf_config = Qwen3VLMoeTextConfig.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
        hf_rope_embedding = Qwen3VLMoeTextRotaryEmbedding(hf_config)


        mbridge_rope_embedding = Qwen3VLTextRotaryEmbedding(hf_config)

        
        seq_len = 1024
        batch_size = 1
        rand_hidden_states = torch.randn(batch_size, seq_len, hf_config.hidden_size)
        

        position_ids_2d = torch.arange(seq_len).unsqueeze(0)  # shape: (1, 1024)
        position_ids_3d = position_ids_2d[None, ...].expand(3, batch_size, -1)  # shape: (3, 1, 1024)

        mrope_section=[24, 20, 20]
        
        # Get HF outputs: (bs, seq_len, head_dim) for both cos and sin
        hf_cos, hf_sin = hf_rope_embedding(rand_hidden_states, position_ids_3d)

        # Get MBridge output: (seq_len, bs, 1, head_dim) raw emb (before cos/sin)
        mbridge_rope_output = mbridge_rope_embedding(position_ids_3d, mrope_section)

        # Apply cos/sin and attention_scaling to match HF processing
        attention_scaling = hf_rope_embedding.attention_scaling
        mbridge_cos = (mbridge_rope_output.cos() * attention_scaling).squeeze(2)  # (seq_len, bs, head_dim)
        mbridge_sin = (mbridge_rope_output.sin() * attention_scaling).squeeze(2)  # (seq_len, bs, head_dim)
        
        # Transpose MBridge to match HF shape: (seq_len, bs, head_dim) -> (bs, seq_len, head_dim)
        mbridge_cos = mbridge_cos.transpose(0, 1)
        mbridge_sin = mbridge_sin.transpose(0, 1)

        logging.info(f"HF     - cos: {hf_cos.shape}, sin: {hf_sin.shape}")
        logging.info(f"MBridge - cos: {mbridge_cos.shape}, sin: {mbridge_sin.shape}")
        
        torch.testing.assert_close(hf_cos, mbridge_cos)
        torch.testing.assert_close(hf_sin, mbridge_sin)

if __name__ == "__main__":
    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    test_rope = TestQwen3VLTextRotaryEmbedding()
    test_rope.test_qwen3_vl_text_rotary_embedding()