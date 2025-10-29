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
os.environ["WANDB_MODE"] = "offline"
import re
import random
import json
from openai import OpenAI
from tqdm import tqdm
import time
import math
import multiprocessing
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from func_timeout import func_timeout, FunctionTimedOut


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "caption"],
        # ["accuracy", "format", "caption"]
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )

def generate_task(args):
    def extract_caption(text): # 使用正则表达式从文本中提取 <answer> 标签内的内容，去除空格并返回。用于从生成回答和正确答案中提取实际答案。
        pattern = r'<caption>\s*(.*?)\s*</caption>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _call_llm(question: str, caption: str, question_type: str, timeout: int = 90):
        client = OpenAI(api_key="sk-d92330d670ff40fa9e70e6d8c13e52a6", base_url="https://api.deepseek.com")
        
        """90 秒内拿不到结果就抛 FunctionTimedOut"""
        TYPE_TEMPLATE = {
            "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
            "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
            "free-form": " Please provide your text answer within the <answer> </answer> tags.",
            "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
        }

        prompt = (
            "Please answer the question only according to the text description of the visual information below without any visual information.\n"
            f"Visual description:\n {caption} \n"
            f"Question:\n {question} \n"
            "first thinks about the reasoning process in the mind and then provides the user with the answer. "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
            "i.e., <think> reasoning process here </think><answer> answer here </answer>"
            f"{TYPE_TEMPLATE[question_type]}"
        )

        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        return resp.choices[0].message.content
    
    question, text, question_type = args
    caption = extract_caption(text)
    
    try:
        return func_timeout(90, _call_llm, args=(question, caption, question_type))
    except FunctionTimedOut:
        # 超时返回 None，主进程可据此过滤或重试
        return None
    except Exception as e:
        # 其它异常也返回 None
        print(f"[generate_task] Exception: {e}")
        return None
    

# 对<caption></caption>部分进行计算
def caption_reward(completions, solution, **kwargs):
    
    def extract_answer(text): # 使用正则表达式从文本中提取 <answer> 标签内的内容，去除空格并返回。用于从生成回答和正确答案中提取实际答案。
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def extract_caption(text): # 使用正则表达式从文本中提取 <answer> 标签内的内容，去除空格并返回。用于从生成回答和正确答案中提取实际答案。
        pattern = r'<caption>\s*(.*?)\s*</caption>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str): # 将字符串转换为浮点数，处理逗号和点，返回数值或 None。用于数值问题中统一数字格式。  
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"CaptionReward: Error converting '{num_str}' to float: {e}")
            return None

    
    question_type = kwargs['problem_type'][0]
    
    format_question = kwargs['format_question']
    contents = [completion[0]["content"] for completion in completions]
    # index_list = [_ for _ in range(len(contents))]
    question_type_list = kwargs['problem_type']
    
    # 多线程求出LLM基于caption得到的回答
    with multiprocessing.Pool(processes=len(contents)) as pool:
        args_list = zip(format_question, contents, question_type_list)
        results = pool.map(generate_task, args_list)
    
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(results, solution):
        try:
            output_ans = extract_answer(content) # 提取生成的答案
            gt_ans = extract_answer(sol) # 提取ground_truth
            
            if question_type == "multiple choice": # 多选题:答案相等即奖励 直接比较生成答案和正确答案是否完全一致，一致则奖励1.0，否则0.0
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical": # 首先检查是否都包含小数点或逗号，如果不一致，奖励0.0。然后将两个答案转换为浮点数，比较四舍五入后的结果是否相同，相同则奖励1.0，否则0.0
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Caption reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    return rewards



# 准确率计算
def accuracy_reward(completions, solution, **kwargs):
    
    def extract_answer(text): # 使用正则表达式从文本中提取 <answer> 标签内的内容，去除空格并返回。用于从生成回答和正确答案中提取实际答案。
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str): # 将字符串转换为浮点数，处理逗号和点，返回数值或 None。用于数值问题中统一数字格式。  
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    

    question_type = kwargs['problem_type'][0]
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):
        try:
            output_ans = extract_answer(content) # 提取生成的答案
            gt_ans = extract_answer(sol) # 提取ground_truth
            if question_type == "multiple choice": # 多选题:答案相等即奖励 直接比较生成答案和正确答案是否完全一致，一致则奖励1.0，否则0.0
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical": # 首先检查是否都包含小数点或逗号，如果不一致，奖励0.0。然后将两个答案转换为浮点数，比较四舍五入后的结果是否相同，相同则奖励1.0，否则0.0
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<prethink>.*?</prethink>\s*<caption>.*?</caption>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "caption": caption_reward
}

# reward_funcs_registry = {
#     "accuracy": accuracy_reward,
#     "format": format_reward
# }

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs] # 加载reward函数

    # 以jsonl/json的形式加载数据集，DatasetDict格式
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    # 加载json数据
    with open(script_args.dataset_name, 'r', encoding='utf-8') as f:
        data_file = json.load(f)

    data_image = [item for item in data_file if item['data_type'] == 'image']
    data_video = [item for item in data_file if item['data_type'] == 'video']
    SampleNumber = 7


    # 构造对话形式数据
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

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

    # 回答问题模板
    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    }

    def make_conversation_image(example):
        
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
        
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }
        
    def make_conversation_image_and_video(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        if example['data_type'] == 'image':
            data = random.sample(data_image,SampleNumber)
        elif example['data_type'] == 'video':
            data = random.sample(data_video,SampleNumber)
        
        msg = {
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                            # example['data_type']: os.getcwd() + "/Video-R1-data" + example['path'][1:]
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                        }
                        ]
                }],
            "format_question": question,
            "relative_data_prompt":[],
            "relative_data_path":[]
        }
        
        # 先加入自身
        msg['relative_data_prompt'].append([
            {
                "role": "user",
                "content": [
                        {"type": example['data_type']},
                        {
                            "type": "text",
                            "text": example['problem']
                        }
                ]
            }
        ])
        msg['relative_data_path'].append(example['path'])
        
        # 再加入跟随的相关数据
        for x in data:
            msg['relative_data_prompt'].append([
                {
                    "role": "user",
                    "content": [
                            {"type": x['data_type']},
                            {
                                "type": "text",
                                "text": x['problem']
                            }
                    ]
                }
            ])
            msg['relative_data_path'].append(x['path'])        
        
        return msg

    
    dataset = dataset.map(make_conversation_image_and_video)

    
    trainer_cls = Qwen2VLGRPOTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer （重要）
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,# 定义奖励函数[reward1,reward2]
        args=training_args,# 训练时的基本超参数配置
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split], # 训练集
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None, # 测试集,这里为None
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # print(script_args,training_args,model_args)
    main(script_args, training_args, model_args)
