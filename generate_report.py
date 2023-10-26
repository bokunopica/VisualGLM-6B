#!/usr/bin/env python

import random
import gradio as gr
from PIL import Image
import os
import json
import pandas as pd
from model import (
    is_chinese,
    get_infer_setting,
    generate_input,
    chat,
)
from tqdm import trange
import torch
from finetune_visualglm import FineTuneVisualGLMModel
from sat.quantization.kernels import quantize
from sat.model.mixins import CachedAutoregressiveMixin
from transformers import AutoTokenizer

def extract_bool_from_answer(answer) -> bool:
    # TODO extract
    if answer.startswith('是'):
        return 1
    elif answer.startswith('否') or answer.startswith('不'):
        return 0
    else:
        return -1

def get_result(image_path) -> bool:
    temperature = 1e-6 # 尽可能减小随机性 
    max_length = 512
    top_p = 0.4
    top_k = 100

    # few shot prompt
    # history_prompt = [
    #     ["根据X射线图像，心脏大小正常，肺部看起来很清晰。已经排除了肺炎、积液、水肿、气胸、腺病、结节或肿块的存在。该发现表明一切正常。通过该报告，是否能诊断出肺炎？请回答是或否：", "否"],
    #     ["X光片显示出正常大小和轮廓的心胸廓线。没有气胸或大量胸腔积液。然而，右心尖有肿块状不透明，这表明恶性肿瘤恶化或恶性肿瘤合并阻塞性肺炎。可能需要进一步的评估和测试以确认诊断并决定适当的治疗方案。通过该报告，是否能诊断出肺炎？请回答是或否：", "是"],
    #     ["胸部X光显示心脏大小正常，肺部清晰，没有腺体病变、结节或肿块的迹象。另外，没有肺炎、渗出、水肿、气胸或结核的迹象。总的来说，X光片显示胸部正常，没有任何急性心肺功能异常的情况。通过该报告，是否能诊断出肺炎？请回答是或否：", "否"],
    #     ["X光图像显示没有心肺疾病的急性发现。心脏大小在正常范围内，而纵膈似乎在正常范围内。没有胸腔积液、气胸或局灶性空隙不透明，说明有肺炎。通过该报告，是否能诊断出肺炎？请回答是或否：", "是"]
    # ]

    with torch.no_grad():
        history = []
        input_text = "通过这张胸部X光影像可以诊断出什么？"
        input_image = Image.open(image_path) # direction
        answer, _history, _torch_image = chat(
            None,
            model,
            tokenizer,
            input_text,
            history=history,
            image=input_image,
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            english=False,
        )
    return answer # result int, answer str

def main(args):
    # 固定随机种子
    random.seed(110)
    torch.manual_seed(110)
    # model initialize
    global model, tokenizer
    model, model_args = FineTuneVisualGLMModel.from_pretrained(
        args.ckpt_path,
        args=argparse.Namespace(
            fp16=True,
            skip_init=True,
            use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
            device='cuda' if (torch.cuda.is_available() and args.quant is None) else 'cpu',
        )
    )
    model = model.eval()
    if args.quant:
        quantize(model.transformer, args.quant)
    if torch.cuda.is_available():
        model = model.cuda()
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

    ## validation_data initialize
    result_list = []

    base_dir = "/home/qianq/data/OpenI-zh-resize-384/images"
    file_path = "/home/qianq/data/OpenI-zh-resize-384/images/openi-zh-test-prompt.json"
    with open(file_path) as f:
        data = json.load(f)
    for i in trange(len(data)):
        single_data = data[i]
        image_path = f"{base_dir}/{single_data['img'].split('/')[-1]}"
        generate_report = get_result(image_path)
        single_data['generated'] = generate_report
        result_list.append(single_data)
    
    with open("eval_reports.json", 'w') as f:
        f.write(json.dumps(result_list))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--ckpt_path", type=str)
    args = parser.parse_args()
    main(args)
