# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import json
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import random

import torch
import torch.nn as nn
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from qwen_vl_utils import process_vision_info

import copy


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
            

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
                # model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        #self.ref_model = None
        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                # self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or True:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.cotReward_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
        )
        
        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.len_control = script_args.len_control
        self.beta = args.beta
        self.infoNCELoss_weight = 0.01
        self.tau = 0.5
        self.threshold = 0.5

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, **kwargs):
        # logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        # import pdb
        # pdb.set_trace()
        logits = model(
            input_ids,
            **kwargs
        ).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # 由于视频和图片是混合的，所以要先确定哪些数据是视频，哪些数据是图片
        image_index, video_index = [], []
        batch_size = len(inputs)
        prompts = [x["prompt"] for x in inputs] # B
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs] # B
        prompts_relative_data_text = [self.processing_class.apply_chat_template(x, tokenize=False, add_generation_prompt=False) for x in inputs[0]['relative_data_prompt']]
        
        
        input_copy = copy.deepcopy([x['prompt'] for x in inputs])
        relative_data_copy = copy.deepcopy([x for x in inputs[0]['relative_data_prompt']])
        
        input_copy = self.remove_none_from_data(input_copy) # 这一步可以删掉
        
        # 处理原本的数据
        for i,ipt in enumerate(inputs):
            if ipt['data_type'] == 'image':
                image_index.append(i)
                if "image" in ipt:
                    input_copy[i][0]['content'][0]['image'] = ipt['image']
                else:
                    input_copy[i][0]['content'][0]['image'] = '...' + ipt['path'][1:] # 替换为你的数据地址
            elif ipt['data_type'] == 'video':
                video_index.append(i)
                input_copy[i][0]['content'][0]['video'] = '...' + ipt['path'][1:] # 替换为你的数据地址
        
        # 处理相关的数据
        for i,ipt in enumerate(inputs[0]['relative_data_prompt']):
            if inputs[0]['data_type'] == 'image':
                relative_data_copy[i][0]['content'][0]['image'] = '...' + inputs[0]['relative_data_path'][i][1:] # 替换为你的数据地址
            elif inputs[0]['data_type'] == 'video':
                relative_data_copy[i][0]['content'][0]['video'] = '...' + inputs[0]['relative_data_path'][i][1:] # 替换为你的数据地址

        
        
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True) # 这里返回的形式为batch
        relative_image_inputs, relative_video_inputs, relative_video_kwargs = process_vision_info(relative_data_copy, return_video_kwargs=True)
        
        # 图片和视频即使混合了也没关系，process里面的处理逻辑已经自动考虑了这点
        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        ) # 将采样的图片+文本转换为ids；token预处理
        
        
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        # 对于视频token，以"<|placeholder|>"代替,对应的idx为 tensor(151656)

        
        # fix prompt_inputs["input_ids"] length issue 根据最长长度进行裁剪
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        
        if self.max_prompt_length is not None: # 这一步多余了吧
            prompt_ids = prompt_ids[:, -self.max_prompt_length :] # (1,len)
            prompt_mask = prompt_mask[:, -self.max_prompt_length :] # (1,len)
            
        
        
        # Generate completions 生成8条响应
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config) # 这里的结果是shape(8 * batch, len)
            prompt_length = prompt_ids.size(1) # 提示词的长度 444
            prompt_ids = prompt_completion_ids[:, :prompt_length] # 截取回答之前的提示词（包括问题）的部分 torch.Size([8, 444])
            completion_ids = prompt_completion_ids[:, prompt_length:] # 回答的部分 torch.Size([8, 299])
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)# torch.Size([8, 444])
        
        # ######################### 计算InfoNCELoss ##################################################
        # 这里要计算自身和相关的数据部分的InfoNCELoss,需要自身的数据和相关的其他数据
        # 这里在处理文本输入的时候必须按照对应格式处理!
        QandA_prompts = self.processing_class(
            text=copy.deepcopy(prompts_relative_data_text),
            images=relative_image_inputs,
            videos=relative_video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=True,
        ).to(model.device)
        
        QandA_LastHideenstates = model(
            **QandA_prompts,
            output_hidden_states=True,
        ).hidden_states[-1]  # (Batch, len, 3584)
        
        # 对batch里面的掩码进行操作
        batch_mask = (QandA_prompts.attention_mask==0).sum(dim = 1) # (batch)
        hidden_state_list = []
        
        for hs,j in zip(QandA_LastHideenstates, batch_mask):
            st_pos = j.item()
            hidden_state_list.append(torch.mean(hs[st_pos:], dim=0))
        
        # 计算InfoNCELoss损失
        loss_infoNceLoss_batch = torch.tensor(0.0, dtype = QandA_LastHideenstates.dtype,device=model.device)
        sim_list = []
        sim_scale_list = []
        
        sample_hiddenState = hidden_state_list[0]
        pos_sim = torch.tensor(0.0, dtype = QandA_LastHideenstates.dtype,device=model.device)
        all_sim = torch.tensor(0.0, dtype = QandA_LastHideenstates.dtype,device=model.device)
        max_pos_sim = None
        min_pos_sim = None
        
        # 这里有个问题，就是假如pos_sim = 0会导致loss为inf，存在梯度爆炸的情况
        # 解决方法是:如果没有正样本就找相似性最高的强行当成正样本
        
        threshold = self.threshold
        for i,ht in enumerate(hidden_state_list):
            if i == 0:
                continue
            else:
                sim = nn.functional.cosine_similarity(sample_hiddenState.unsqueeze(0), ht.unsqueeze(0), dim = 1)
                all_sim += torch.sum(torch.exp(sim / self.tau))
                
                sim_list.append(sim[0].item())
                sim_scale_list.append(torch.sum(torch.exp(sim / self.tau)).item())
                
                if min_pos_sim == None:
                    min_pos_sim = torch.sum(torch.exp(sim / self.tau))
                else:
                    if min_pos_sim.item() > torch.sum(torch.exp(sim / self.tau)).item():
                        min_pos_sim = torch.sum(torch.exp(sim / self.tau))
                
                print(sim.item())
                if sim.item() <= threshold: # 越大越相似，越小越不相似. 相似的样本要拉远，所以相似的样本为负样本，不相似的样本要拉近，所以不相似的样本为正样本。 
                    pos_sim += torch.sum(torch.exp(sim / self.tau))
                
                if i == len(hidden_state_list) - 1 and pos_sim.item() == 0.0:
                    pos_sim = min_pos_sim
                    

        loss_infoNceLoss_batch += -1.0 * torch.log(pos_sim / all_sim)
        
        loss_infoNceLoss_batch = self.infoNCELoss_weight * loss_infoNceLoss_batch
        
        #########################################################################################
        
        # Mask everything after the first EOS token 根据eos的位置为每个回答创建掩码得到completion_mask torch.Size([8, 370])
        is_eos = completion_ids == self.processing_class.eos_token_id # 找到eos的位置
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device) 
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)] # 找到eos的位置 tensor([192, 226, 369, 179, 219, 239, 293, 288], device='cuda:1')
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1) # 生成一个表示序列位置的张量，形状为 (batch_size, sequence_length)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int() # 根据eos的位置为每个回答创建掩码
        

        
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        
        if 'pixel_values' in prompt_inputs and prompt_inputs.pixel_values is not None:
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(self.num_generations, 1)
        # import pdb; pdb.set_trace()
        

        if 'pixel_values_videos' in prompt_inputs and prompt_inputs.pixel_values_videos is not None:
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(self.num_generations, 1) # 在行这个维度上重复8次 torch.Size([3840, 1176]) -> torch.Size([3840 * 8, 1176])
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(self.num_generations, 1)
            if 'second_per_grid_ts' in prompt_inputs:
                del prompt_inputs["second_per_grid_ts"]
        
        
        
        try: # 计算每个token在policy model下的logps，后面求KL散度需要
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs) # 求出prompt+answer的token logps
            per_token_logps = per_token_logps[:, prompt_length - 1 :] 
        except Exception as e:
            print(f"Error computing per_token_logps: {e}. Setting output to zero.")
            # return torch.tensor(0.0, device=model.device,requires_grad=True)
            per_token_logps = torch.tensor(0.0, device=prompt_completion_ids.device, requires_grad=True)
            # per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
        
        with torch.inference_mode():
            try:
                if self.ref_model is not None:                    
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs) # 计算每个token在ref model下的logps，后面求KL散度需要
                
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :] # 只截取answer部分的logps torch.Size([8, 315])
            except Exception as e:
                print(f"Error computing ref_per_token_logps: {e}. Setting output to zero.")
                ref_per_token_logps = torch.tensor(0.0, device=prompt_completion_ids.device)
                # with self.accelerator.unwrap_model(model).disable_adapter():
                #     ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
                # ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        # 根据policy model和ref model下的token logps计算KL散度,公式见笔记
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)  # 限制 x 的范围
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1
        
        
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
        
        # Compute the rewards 计算默认情况下的奖励值(同上)
        # 计算奖励值的时候原代码是没考虑到多batch的情况，需要修改
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        
        for bth in range(batch_size):
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[bth].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs[bth: bth + 1]:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                
                reward_prompt = prompts[bth * self.num_generations: (bth + 1) * self.num_generations]
                reward_completions = completions[bth * self.num_generations: (bth + 1) * self.num_generations]
                
                if i == 0: # acc奖励 1,0
                    output_reward_func = reward_func(prompts=reward_prompt, completions=reward_completions, **reward_kwargs)
                elif i == 1: # 格式奖励 0.1
                    output_reward_func = reward_func(prompts=reward_prompt, completions=reward_completions, **reward_kwargs)
                    output_reward_func = [x * 0.1 for x in output_reward_func]
                elif i == 2: # Caption Reward 1.0
                    output_reward_func = reward_func(
                        prompts=reward_prompt, 
                        completions=reward_completions, 
                        **reward_kwargs
                    )
                    output_reward_func = [x * 1.0 for x in output_reward_func]
                rewards_per_func[bth * self.num_generations: (bth + 1) * self.num_generations, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                # rewards_per_func tensor([[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.]])

        
        
        
        rewards = rewards_per_func.sum(dim=1) # 计算每条数据的总奖励

        # Compute grouped-wise rewards 计算reward的平均值和标准差
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # 计算每条数据对应的生成的8条回答的奖励的平均值
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # 计算每条数据对应的奖励的标准差

        # Normalize the rewards to compute the advantages 对reward进行标准化,并计算advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4) # 计算每条数据生成的每条响应对应的优势值
        

        # x - x.detach() allows for preserving gradients from x 按照公式计算损失值
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) # 计算每个token的损失 step-1
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)# 计算每个token的损失 step-2
        # per_token_loss = -per_token_loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() # 计算总损失
        loss = loss + loss_infoNceLoss_batch
        
        
       
        # import pdb
        # pdb.set_trace()

        # Log the metrics 对上述loss计算的中间指标进行Log记录
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length) # 记录生成的平均长度
        
        # 记录infoNCELoss损失函数
        self._metrics["infoNCELoss"].append(loss_infoNceLoss_batch.item())

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs): # 记录每个奖励函数的平均奖励值
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        
        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        
        num_devices = gathered_rewards.size(0) // self.num_generations 
        rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices
        
        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices
        
        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)
        
        # 记录整体奖励
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        # 记录标准差
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        # 记录KL散度
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
