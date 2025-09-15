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


from typing import Any

from megatron.bridge.utils.common_utils import print_rank_0
from megatron.bridge.utils.import_utils import MISSING_NVINSPECT_MSG


try:
    import nvdlfw_inspect.api as nvinspect_api
    from nvdlfw_inspect.logging import (
        BaseLogger,
        MetricLogger,
        wrap_tensorboard_writer,
    )
    HAVE_NVINSPECT = True
except (ImportError, ModuleNotFoundError):
    HAVE_NVINSPECT = False


def initialize_tensor_inspect_pre_model(cfg: Any, state: Any) -> None:
    """Initialize NVIDIA-DL-Framework-Inspect before model construction.

    When enabled and the API is unavailable or fails, raise to stop training.
    """

    if cfg.tensor_inspect is None or not cfg.tensor_inspect.enabled:
        return

    if not HAVE_NVINSPECT:
        print_rank_0(MISSING_NVINSPECT_MSG)
        raise ImportError(MISSING_NVINSPECT_MSG)

    try:
        log_dir = cfg.tensor_inspect.log_dir or cfg.checkpoint.save or "."
        nvinspect_api.initialize(
            config_file=cfg.tensor_inspect.features or "",
            feature_dirs=cfg.tensor_inspect.feature_dirs,
            log_dir=log_dir,
            statistics_logger=None,
            init_training_step=state.train_state.step,
            default_logging_enabled=True,
        )
        print_rank_0("Initialized NVIDIA-DL-Framework-Inspect (pre-model).")
    except Exception as e:
        # Treat initialization failures as fatal when enabled so training exits
        print_rank_0(f"NVIDIA DLFw Inspect pre-init failed: {e}")
        raise


def _maybe_attach_metric_loggers(state: Any) -> None:
    """Attach supported metric loggers (TensorBoard, W&B raw module)."""

    try:
        # TensorBoard
        if state.tensorboard_logger is not None:
            tb_logger = wrap_tensorboard_writer(state.tensorboard_logger)
            MetricLogger.add_logger(tb_logger)

        # Raw wandb module (with .log)
        if state.wandb_logger is not None and hasattr(state.wandb_logger, "log"):
            if BaseLogger is None:
                return

            class _WandbModuleLogger(BaseLogger):  # type: ignore
                def __init__(self, wandb_module):
                    super().__init__()
                    self._wandb = wandb_module

                def log_scalar(self, name: str, value: float | int, iteration: int, **kwargs):  # type: ignore[override]
                    self._wandb.log({name: value}, step=iteration)

            MetricLogger.add_logger(_WandbModuleLogger(state.wandb_logger))
    except Exception as e:  
        print_rank_0(f"Skipping NVIDIA DLFw Inspect metric logger attach due to error: {e}")


def finalize_tensor_inspect_post_model(model: Any, state: Any) -> None:
    """Finalize setup after model creation: attach loggers, set names and groups."""

    if state.cfg.tensor_inspect is None or not state.cfg.tensor_inspect.enabled:
        return

    if not HAVE_NVINSPECT:
        print_rank_0(MISSING_NVINSPECT_MSG)
        raise ImportError(MISSING_NVINSPECT_MSG)

    try:
        from megatron.core.parallel_state import get_tensor_and_data_parallel_group

        _maybe_attach_metric_loggers(state)

        nvinspect_api.infer_and_assign_layer_names(model)
        nvinspect_api.set_tensor_reduction_group(get_tensor_and_data_parallel_group())
        print_rank_0("Finalized NVIDIA DLFw Inspect (post-model).")
    except Exception as e:
        # Treat post-model finalize failures as fatal when enabled so training exits
        print_rank_0(f"NVIDIA DLFw Inspect post-init failed: {e}")
        raise


def tensor_inspect_step_if_enabled(cfg: Any) -> None:
    """Advance DLFw Inspect step if enabled."""

    if cfg.tensor_inspect is None or not cfg.tensor_inspect.enabled:
        return
    if not HAVE_NVINSPECT:
        print_rank_0(MISSING_NVINSPECT_MSG)
        raise ImportError(MISSING_NVINSPECT_MSG)
    try:
        nvinspect_api.step()
    except Exception as e:
        print_rank_0(f"NVIDIA DLFw Inspect step failed: {e}")
        raise


def tensor_inspect_end_if_enabled(cfg: Any) -> None:
    """Shutdown DLFw Inspect if enabled."""

    if cfg.tensor_inspect is None or not cfg.tensor_inspect.enabled:
        return
    if not HAVE_NVINSPECT:
        return
    try:
        nvinspect_api.end_debug()
    except Exception as e:
        print_rank_0(f"NVIDIA DLFw Inspect end failed: {e}")


