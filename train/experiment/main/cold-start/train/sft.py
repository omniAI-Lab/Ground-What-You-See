# Copyright 2024. All rights reserved.
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
"""
Example usage:
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    train_video_llm.py \
    --dataset_name mfarre/simplevideoshorts \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir video-llm-output \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
"""

import os
os.environ["WANDB_MODE"] = "offline"
import json
import random
import requests
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info

from datasets import Dataset, DatasetDict

import wandb

from typing import List, Dict, Any

def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def download_video(url: str, folder: str = '/tmp/videos/') -> str:
    """Download video if not already present locally."""
    filename = url.split("/")[-1]
    local_path = os.path.join(folder, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except requests.RequestException as e:
        raise Exception(f"Failed to download video: {e}")

def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """准备训练数据"""

    system_message = "You are a helpful assistant"
    
    # 问题模板
    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please carefully analyze the pictures (or videos) and problems according to the following requirements"
        "In <prethink> </prethink> tags, carefully analyze the problem and briefly explain the steps to explain the problem and the key thinking direction of reasoning the problem"
        "In <caption> </caption> tags, Please describe the image carefully, paying special attention to the details related to the problem and the reasoning direction of solving the problem"
        "In <think> </think> tags, outline a step-by-step thought process you would use to solve the problem based on the image"
        "In <answer> </answer> tags, give the final answer in a direct format, and it must match the correct answer exactly."
        "Please sort out the output in the format of '<prethink>...</prethink>\n<caption>...</caption>\n<think>...</think>\n<answer>...</answer>' according to the above requirements"
    )
    
    # 回答模板(根据问题类型而不同)
    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    }


    question = example['problem']

    # 制作对话数据 sys + Question + answer
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": example['data_type'],
                    example['data_type']: '...' + example['path'][1:] # 替换为实际的地址
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question)
                }
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example['process_and_answer']}]
        }
    ]
    

    return {"messages": messages}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    # video_inputs = []
    # image_inputs = []

    for i, example in enumerate(examples):
        try:
            
            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))# 利用processor解析对话格式数据转换为模型可接受的纯文本: 
            image_inputs, video_inputs, video_kwargs = process_vision_info(example["messages"], return_video_kwargs=True) # 读取并处理多模态信息
            
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")
    # 利用processor 将数据转换为模型可接受的输入(input_ids,attention...)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )
    # 为什么要把<pad>和<多模态 visual token> 的ids改为-100 ?
    labels = inputs["input_ids"].clone() # 提取出prompt+答案的input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100 # 将<pad>的ids改为-100

    # 找出多模态visual token 对应的ids
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ]

    for visual_token_id in visual_tokens: # 将多模态visual token对应的ids改为-100
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    return inputs

if __name__ == "__main__":
    # 解析输入的参数
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # 加载数据集
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )


    # Model initialization 初始化模型参数，方便加载模型
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
        # quantization_config=bnb_config,
    )# {'revision': 'main', 'trust_remote_code': False, 'torch_dtype': None, 'device_map': {'': 0}}
    
    
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path: # 加载模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    # 加载processor 处理图像
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )

    # Prepare dataset
    prepared_dataset = [prepare_dataset(example) for example in dataset['train']]

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training")

    # 初始化trainer对象
    trainer = SFTTrainer(
        model=model,
        args=training_args, # 训练参数
        train_dataset=prepared_dataset, # 数据 <class 'list'>
        data_collator=collate_fn, # 将数据集的数据制作为能直接输入到模型中的inputs
        peft_config=get_peft_config(model_config),
        # tokenizer=processor.tokenizer
    )

    # Train model
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
