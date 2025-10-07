import torch

from megatron.bridge.training.utils.visual_inputs import Qwen2_5_VLVisualInputs
from megatron.bridge.training.vlm_step import get_batch_from_iterator


class _Iterator:
    def __init__(self, batch):
        self.batch = batch
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        self._done = True
        return self.batch


def _make_batch(device="cpu"):
    # Minimal text tensors
    tokens = torch.tensor([[1, 2, 3]], device=device)
    input_ids = tokens.clone()
    position_ids = torch.tensor([[0, 1, 2]], device=device)
    labels = torch.tensor([[2, 3, 4]], device=device)
    loss_mask = torch.ones_like(labels, dtype=torch.float, device=device)
    attention_mask = torch.ones_like(tokens, dtype=torch.bool, device=device)

    # Visual inputs container
    pixel_values = torch.randn(1, 2, 3, 4, 4, device=device)
    image_grid_thw = torch.tensor([[[1, 2, 2], [1, 2, 2]]], device=device)
    vi = Qwen2_5_VLVisualInputs(pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    batch = {
        "tokens": tokens,
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "visual_inputs": vi,
    }
    return batch


def test_get_batch_from_iterator_moves_visual_inputs_to_cuda(monkeypatch):
    # Avoid requiring distributed/parallel initialization in unit test
    monkeypatch.setattr(
        "megatron.core.parallel_state.is_pipeline_last_stage",
        lambda: False,
        raising=True,
    )

    # Simulate Training on CPU-only env by making .cuda a no-op that returns the same tensor
    class _NoCudaTensor(torch.Tensor):
        def cuda(self, non_blocking=False):  # type: ignore[override]
            return self

    def _as_nocuda(t):
        return t.as_subclass(_NoCudaTensor)

    batch = _make_batch()
    # Replace tensors with _NoCudaTensor so calling .cuda works without a GPU
    for k in ["tokens", "input_ids", "position_ids", "labels", "loss_mask", "attention_mask"]:
        batch[k] = _as_nocuda(batch[k])
    vi = batch["visual_inputs"]
    vi.pixel_values = _as_nocuda(vi.pixel_values)
    vi.image_grid_thw = _as_nocuda(vi.image_grid_thw)

    it = _Iterator(batch)
    out = get_batch_from_iterator(it, use_mtp=False, skip_getting_attention_mask_from_dataset=True)

    assert "visual_inputs" in out
    out_vi = out["visual_inputs"]
    assert isinstance(out_vi, Qwen2_5_VLVisualInputs)
    # Verify fields are preserved
    assert out_vi.pixel_values is not None and out_vi.image_grid_thw is not None
