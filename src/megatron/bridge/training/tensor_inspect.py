"""Unified NVIDIA DLFw Inspect integration helpers.

This module centralizes initialization, logger attachment, step advancement,
and shutdown for NVIDIA DLFw Inspect to keep call sites simple and consistent.
"""

from __future__ import annotations

from typing import Any

from megatron.bridge.utils.common_utils import print_rank_0


def initialize_tensor_inspect_pre_model(cfg: Any, state: Any) -> None:
    """Initialize NVIDIA DLFw Inspect before model construction.

    Safe to call when disabled; errors are caught and logged on rank 0.
    """
    if cfg.tensor_inspect is None or not cfg.tensor_inspect.enabled:
        return

    try:
        import nvdlfw_inspect.api as nvinspect_api  # type: ignore

        log_dir = cfg.tensor_inspect.log_dir or cfg.checkpoint.save or "."
        nvinspect_api.initialize(
            config_file=cfg.tensor_inspect.features or "",
            feature_dirs=cfg.tensor_inspect.feature_dirs,
            log_dir=log_dir,
            statistics_logger=None,
            init_training_step=state.train_state.step,
            default_logging_enabled=True,
        )
        print_rank_0("Initialized NVIDIA DLFw Inspect (pre-model).")
    except Exception as e:  # noqa: BLE001
        print_rank_0(f"Skipping NVIDIA DLFw Inspect pre-init due to error: {e}")


def _maybe_attach_metric_loggers(state: Any) -> None:
    """Attach supported metric loggers (TensorBoard, W&B raw module)."""
    try:
        from nvdlfw_inspect.logging import BaseLogger, MetricLogger, wrap_tensorboard_writer  # type: ignore

        # TensorBoard
        if state.tensorboard_logger is not None:
            tb_logger = wrap_tensorboard_writer(state.tensorboard_logger)
            MetricLogger.add_logger(tb_logger)

        # Raw wandb module (with .log)
        if state.wandb_logger is not None and hasattr(state.wandb_logger, "log"):

            class _WandbModuleLogger(BaseLogger):  # type: ignore
                def __init__(self, wandb_module):
                    super().__init__()
                    self._wandb = wandb_module

                def log_scalar(self, name: str, value: float | int, iteration: int, **kwargs):  # type: ignore[override]
                    self._wandb.log({name: value}, step=iteration)

            MetricLogger.add_logger(_WandbModuleLogger(state.wandb_logger))
    except Exception as e:  # noqa: BLE001
        print_rank_0(f"Skipping NVIDIA DLFw Inspect metric logger attach due to error: {e}")


def finalize_tensor_inspect_post_model(model: Any, state: Any) -> None:
    """Finalize setup after model creation: attach loggers, set names and groups."""
    if state.cfg.tensor_inspect is None or not state.cfg.tensor_inspect.enabled:
        return

    try:
        import nvdlfw_inspect.api as nvinspect_api  # type: ignore
        from megatron.core.parallel_state import get_tensor_and_data_parallel_group

        _maybe_attach_metric_loggers(state)

        nvinspect_api.infer_and_assign_layer_names(model)
        nvinspect_api.set_tensor_reduction_group(get_tensor_and_data_parallel_group())
        print_rank_0("Finalized NVIDIA DLFw Inspect (post-model).")
    except Exception as e:  # noqa: BLE001
        print_rank_0(f"Skipping NVIDIA DLFw Inspect post-init due to error: {e}")


def tensor_inspect_step_if_enabled(cfg: Any) -> None:
    """Advance DLFw Inspect step if enabled; ignore errors."""
    if cfg.tensor_inspect is None or not cfg.tensor_inspect.enabled:
        return
    try:
        import nvdlfw_inspect.api as nvinspect_api  # type: ignore

        nvinspect_api.step()
    except Exception:
        # Best effort only
        pass


def tensor_inspect_end_if_enabled(cfg: Any) -> None:
    """Shutdown DLFw Inspect if enabled; ignore errors."""
    if cfg.tensor_inspect is None or not cfg.tensor_inspect.enabled:
        return
    try:
        import nvdlfw_inspect.api as nvinspect_api  # type: ignore

        nvinspect_api.end_debug()
    except Exception:
        # Best effort only
        pass


