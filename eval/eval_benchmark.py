import os
import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse
from datasets import load_dataset, load_from_disk


BSZ = 64


parser = argparse.ArgumentParser(description="Evaluation benchmark")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--dataset', type=str, required=True, help="Path to the Dataset")
parser.add_argument('--cot', type=str, required=True, help="COT To inference")
parser.add_argument('--savepath', type=str, required=True, help="Path To Save")
parser.add_argument('--tips', type=str, required=False, default='default', help="tips")
args = parser.parse_args()

# 提取模型地址和文件名
MODEL_PATH = args.model_path
DATASET = args.dataset
COT = args.cot
TIPS = args.tips
PATHSAVE = args.savepath
DATASETNAME = DATASET.split('->')[1]
DATASET_PATH = DATASET.split('->')[0]
DATASET_SPLIT_VAL = DATASET.split('->')[2]

# 加载模型
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len = 8192,
    gpu_memory_utilization=0.9,
    limit_mm_per_prompt={"image": 1, "video": 1},
)

# 推理时的采样超参数设置(默认下面的设置即可)
sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=4096,# 最多输出token数量
    stop_token_ids=[],
)

# 加载tokenizer 和 processor
processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer



for dataset_name in [DATASETNAME]:

    # 测试结果保存地址和eval数据集目录地址
    OUTPUT_PATH = f"{PATHSAVE}/eval_{dataset_name}_{COT}_{MODEL_PATH.split('/')[-2]}_{MODEL_PATH.split('/')[-1]}_{TIPS}_output.json"
    
    if DATASET_PATH.endswith('.jsonl'):
        data = Dataset.from_json(DATASET_PATH)
    elif DATASET_PATH.endswith('.json'):
        data = Dataset.from_json(DATASET_PATH)
    else:
        data = load_dataset(DATASET_PATH)[DATASET_SPLIT_VAL]
    
    # cot-grpo
    QUESTION_TEMPLATE_TA = (
        "{Question}\n"
        "Please begin by carefully and comprehensively scrutinizing the image provided above, ensuring you grasp every visual detail and any implicit information it conveys. "
        "Next, on the basis of this thorough understanding, thoughtfully address the question that follows. "
        "To make your reasoning transparent and reproducible, proceed step by step just as a human would: present your entire chain of analysis, intermediate deductions, and evidential support within the `<think>` and `</think>` tags. "
        "Finally, place only the definitive answer—after all reflection is complete—between the `<answer>` and `</answer>` tags. Remember, "
        "the final output must strictly adhere to the format `<think>...</think><answer>...</answer>`, with no omissions or misplacement. Thank you for your careful attention to these instructions."
    )
    
    # cot-grpocaptionreward
    QUESTION_TEMPLATE_PCDA = (
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

    if COT == 'TA':
        QUESTION_TEMPLATE = QUESTION_TEMPLATE_TA
    elif COT == 'PCDA':
        QUESTION_TEMPLATE = QUESTION_TEMPLATE_PCDA



    messages = []
    
    def make_conversation_MathVista(example):
        question = example['question']
        answer = example['answer']
        ans_type = example['answer_type']
        image = '...' + example['image'] # 填入MathVista地址
        
        if ans_type == 'text':
            problem_type = 'multiple choice'
        else:
            problem_type = 'numerical'
        
        if problem_type == 'multiple choice':
            question += "Options:\n"
            for i,op in enumerate(example['choices']):
                ans = chr(ord('A') + i)
                question += ans + ". " + op + "\n"
                
                if op == answer:
                    answer = ans
        
        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE['numerical']
                        }
                        ]
                }],
            "solution": "<answer>" + answer + "</answer>",
            "problem_type": problem_type,
            "data_type": "image",
            "format_question": question,
            "q_id": example['pid'],
            }
        
        return msg
    
    def make_conversation_ClevrMath(example):
        question = example['problem']
        answer = example['answer']
        question_id = example['question_id']
        
        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE['numerical']
                        }
                        ]
                }],
            "solution": "<answer>" + answer + "</answer>",
            "problem_type": 'numerical',
            "data_type": "image",
            "format_question": question,
            "q_id": question_id,
            }
        
        return msg
    
    def make_conversation_HallusionBench(example):
        question = example['question'] + "Options:\n" + "A. Yes\nB. No\n"
        
        gt_answer = example['gt_answer']
        
        if gt_answer == '1':
            answer = "A"
        elif gt_answer == '0':
            answer = "B"
        
        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE['multiple choice']
                        }
                        ]
                }],
            "solution": "<answer>" + answer + "</answer>",
            "problem_type": 'multiple choice',
            "data_type": "image",
            "format_question": question,
            "q_id": question,
            }
        
        return msg
    
    def make_conversation_ChartQA(example):
        question = example['query']
        answer = example['label'][0]
        
        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE['numerical']
                        }
                        ]
                }],
            "solution": "<answer>" + answer + "</answer>",
            "problem_type": 'numerical',
            "data_type": "image",
            "format_question": question,
            "q_id": question
            }
        
        return msg
    
    def make_conversation_AOKVQA(example):
        question = example['question'] + "Options:\n"

        for c,op in zip(['A','B','C','D'],example["choices"]):
            question += c + ". " + op + "\n"
        
        answer = chr(ord('A') + int(example['correct_choice_idx']))

        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE['multiple choice']
                        }
                        ]
                }],
            "solution": "<answer>" + answer + "</answer>",
            "problem_type": 'multiple choice',
            "data_type": "image",
            "format_question": question,
            "q_id": example['question_id']
            }
        
        return msg
    
    def make_conversation_MMBench(example):
        question = example['question'] + "Options:\n"

        for c in ['A','B','C','D']:
            op = example[c]
            question += c + ". " + op + "\n"
        
        answer = example['answer']

        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE['multiple choice']
                        }
                        ]
                }],
            "solution": "<answer>" + answer + "</answer>",
            "problem_type": 'multiple choice',
            "data_type": "image",
            "format_question": question,
            "q_id": example['index']
            }
        
        return msg
    
    def make_conversation_MMStar(example):
        question = example['question']
        
        answer = example['answer']

        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE['multiple choice']
                        }
                        ]
                }],
            "solution": "<answer>" + answer + "</answer>",
            "problem_type": 'multiple choice',
            "data_type": "image",
            "format_question": question,
            "q_id": example['index']
            }
        
        return msg
    
    def make_conversation_MMathCoT(example):
        question = example['problem']
        msg ={
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
                        ],
                }],
            "format_question":question,
            "q_id": example['problem_id'],
            "image": '...' + example['path'][1:] # 填入MMathCoT地址
            }
        
        return msg
    
    def make_conversation_POPE(example):
        question = example['question'] + "Options:\n" + "A. yes\nB. no\n"
        ans = example['answer']
        
        if ans == 'yes':
            answer = 'A'
        else:
            answer = 'B'
            

        msg = {
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE['multiple choice']
                        }
                        ]
                }],
            "solution": "<answer>" + answer + "</answer>",
            "problem_type": 'multiple choice',
            "data_type": "image",
            "format_question": question,
            "q_id": example['id']
        }
        
        return msg
    
    def make_conversation_image_and_video(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        
        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                        }
                        ],
                }],
            "format_question":question,
            "q_id": example['problem_id'],
            "image": '/opt/data/private/gwj/omni-video-r1/data/eval_data' + example['path'][1:]
            }
        
        return msg
    
    if dataset_name == 'AOKVQA':
        make_conversation_cot_image = make_conversation_AOKVQA
    elif dataset_name == 'MMBench':
        make_conversation_cot_image = make_conversation_MMBench
    elif dataset_name == 'MMSTAR':
        make_conversation_cot_image = make_conversation_MMStar
    elif dataset_name == 'ChartQA':
        make_conversation_cot_image = make_conversation_ChartQA
    elif dataset_name == 'ClevrMath':
        make_conversation_cot_image = make_conversation_ClevrMath
    elif dataset_name == 'MathVista':
        data = data.filter(lambda x: x['answer_type'] != 'list') # 只保留图像数据
        make_conversation_cot_image = make_conversation_MathVista
    elif dataset_name == 'MMathCoT':
        make_conversation_cot_image = make_conversation_MMathCoT
    elif dataset_name == 'HallusionBench':
        make_conversation_cot_image = make_conversation_HallusionBench
    elif dataset_name == 'POPE':
        make_conversation_cot_image = make_conversation_POPE
    else:
        make_conversation_cot_image = make_conversation_image_and_video

    data = data.map(make_conversation_cot_image)
    
    for x in tqdm(data): # 构造输入数据
        if dataset_name == 'ClevrMath':
            image_or_video = x['images'][0]
        elif dataset_name == 'MathVista':
            image_or_video = x['decoded_image']
        elif dataset_name == 'Video_Hullucer':
            image_or_video = x['path']
        else:
            image_or_video = x['image']
        msg = [{
            "role": "user",
            "content": [
                {
                    "type": x['data_type'],
                    x['data_type']: image_or_video
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=x['format_question']) + TYPE_TEMPLATE[x['problem_type']]
                }
            ]
        }]
        messages.append(msg)
        

    final_output = []
    start_idx = 0
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                final_output = existing.get("results", [])
                start_idx = len(final_output)
                print(f"Resuming from sample index {start_idx}")
        except Exception as e:
            print(f"Error reading existing output file: {e}")

    # 下面是一些后面处理数据会用到的工具函数
    def extract_think(output_str): # 提取<think>中间的内容
        pattern = r'<think>\s*(.*?)\s*</think>'
        match = re.search(pattern, output_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def extract_answer(text): # 提取<answer>中间的内容
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def extract_mathvista_answer(dataset,text): # 提取<answer>中间的内容
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            answer = match.group(1).strip()
            precision = dataset['precision']
            answer_type = dataset['answer_type']
            question_type = dataset['question_type']
            
            if question_type == 'multi_choice':
                return answer
            elif answer_type == 'integer':
                try:
                    answer = str(int(float(answer)))
                except Exception:
                    answer = str(answer)
                return answer
            elif answer_type == 'float':
                try:
                    answer = str(round(float(answer), int(precision)))
                except Exception:
                    answer = str(answer)
                return answer
                
            return answer
        return ""

    def normalize_number(num_str): # 将字符串格式的数字转换为float
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            return None
        
    def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):#平均相对准确度
        # 转换为tensor格式
        if not torch.is_tensor(pred):
            pred = torch.tensor(pred, dtype=torch.float32)
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float32)
        
        epsilon = 1e-8
        rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
        
        thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
        
        conditions = rel_error < (1 - thresholds)  
        mra = conditions.float().mean()  
        return mra.item()


    def reward_fn(sample, model_output, question_type):# 记录当前数据模型推理结果的准确奖励得分 用来计算准确率
        try:
            output_ans = extract_answer(model_output)
            if output_ans == '':
                output_ans = model_output
            gt_ans = extract_answer(sample.get("solution", ""))
            if question_type == "multiple choice":
                return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    return 0.0
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                mra = mean_relative_accuracy(out_number, gt_number)
                return mra
            else:
                return 0.0
        except Exception as e:
            return 0.0

    mean_acc = []
    mean_mra = []
    for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing batches"):
        batch_messages = messages[i:i + BSZ] # 提取出一个batch的数据
        # 构造输入prompt 
        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        

        try:# 解析batch数据里面的图像/视频
            image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
            
            image_idx = 0
            video_idx = 0

            llm_inputs = [] # 构造输入到llm之前的input数据（一个batch）

            
            for idx, prompt in enumerate(prompts): # 遍历prompts
                mm_type = batch_messages[idx][0]['content'][0]['type'] # 当前输入多模态信息的类型 video or image
                sample_mm_data = {} # 构造最终的视频输入数据
                sample_video_kw = {}
                if mm_type == 'image':
                    sample_mm_data["image"] = image_inputs[image_idx]
                    image_idx += 1
                elif mm_type == 'video':
                    sample_mm_data["video"] = video_inputs[video_idx] # 得到视频输入数据放入sample_mm_data
                    for key, value in video_kwargs.items(): # 将视频信息放入sample_mm_data
                        sample_video_kw[key] = value[video_idx]
                    video_idx += 1
                        
                
                llm_inputs.append({ # 加入到llm_inputs中
                    "prompt": prompt,
                    "multi_modal_data": sample_mm_data,
                    "mm_processor_kwargs": sample_video_kw,
                })
                

            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            batch_output_text = [out.outputs[0].text for out in outputs] # 记录一个batch推理之后的输出答案
            
        except Exception as e:
            print('error:', data[i]['path'])
            batch_output_text = ['<answer>error</answer>'] * BSZ
            

        for j, model_output in enumerate(batch_output_text):# sample是原始数据集
            sample = data[j + i]
            result = {}

            if dataset_name == 'MathVista':
                final_ans = extract_mathvista_answer(sample,model_output) # <answer>标签中间的答案
            else:
                final_ans = extract_answer(model_output) # <answer>标签中间的答案
            
            if final_ans == "":
                final_ans = model_output
            
            result['question_id'] = sample['q_id']
            result['format_question'] = sample['format_question']
            result["output"] = model_output # 记录输出
            result["prediction"] = final_ans # 记录预测
            result["solution"] = sample["solution"]
            
            q_type = sample.get("problem_type", "")
            result["reward"] = reward_fn(sample, model_output, q_type) # 记录准确率奖励分数
            result['correct'] = True if result["reward"]==1.0 else False
            
            if sample['problem_type'] != 'regression':
                mean_acc.append(result["reward"])
            else:
                mean_mra.append(result["reward"])

            final_output.append(result) # 记录最后输出(原数据+输出结果)
        

        try: # 保存结果
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
            print(f"Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
        except Exception as e:
            print(f"Error writing to output file: {e}")

    final_acc={'mean_acc': 0.0, 'mean_mra': 0.0}
    final_acc['mean_acc'] = torch.tensor(mean_acc).mean().item()
    if mean_mra != []:
        final_acc['mean_mra'] = torch.tensor(mean_mra).mean().item()
    
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f: # 记录正确率
            json.dump({"results": final_output, "final_acc": [final_acc]}, f, indent=2, ensure_ascii=False)
        print(f"Final accuracy saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error writing final accuracy to output file: {e}")
    
    print(f"Results saved to {OUTPUT_PATH}")
