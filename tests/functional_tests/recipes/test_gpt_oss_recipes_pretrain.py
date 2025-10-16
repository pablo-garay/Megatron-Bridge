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

"""Functional smoke tests for GPT-OSS recipe configurations."""

import pytest
from typing_extensions import Unpack

from megatron.bridge.recipes.gpt_oss.gpt_oss import GPTOSSCommonKwargs, _gpt_oss_common
from megatron.bridge.training.config import ConfigContainer
from tests.functional_tests.recipes.utils import run_pretrain_config_override_test, run_pretrain_recipe_test


def gpt_oss_toy_pretrain_config(**user_kwargs: Unpack[GPTOSSCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for GPT-OSS toy variant."""
    recommended: GPTOSSCommonKwargs = {
        "hf_path": "openai/gpt-oss-20b",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "sequence_parallelism": False,
        "use_null_tokenizer": True,
        "global_batch_size": 8,
        "num_layers": 2,
    }
    kwargs: GPTOSSCommonKwargs = {**recommended, **user_kwargs}
    return _gpt_oss_common(**kwargs)


GPT_OSS_PRETRAIN_RECIPES = [
    # (config_func, name, parallelism_overrides)
    (gpt_oss_toy_pretrain_config, "gpt_oss_toy", {}),
]


class TestGPTOSSRecipes:
    """Test class for GPT-OSS recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,parallelism_overrides", GPT_OSS_PRETRAIN_RECIPES)
    def test_gpt_oss_pretrain_recipes(self, config_func, recipe_name, parallelism_overrides, tmp_path):
        """Functional test for GPT-OSS recipes with appropriate parallelism configurations."""
        run_pretrain_recipe_test(config_func, recipe_name, tmp_path, **parallelism_overrides)

    @pytest.mark.parametrize("config_func,recipe_name,parallelism_overrides", GPT_OSS_PRETRAIN_RECIPES)
    def test_pretrain_config_override_after_instantiation(self, config_func, recipe_name, parallelism_overrides):
        """Functional test for overriding GPT-OSS recipes from CLI"""
        run_pretrain_config_override_test(config_func)
