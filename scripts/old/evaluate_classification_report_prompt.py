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
    temperature = 1e-5
    max_length = 512
    top_p = 0.4
    top_k = 100
    history_prompt = [
        ["根据X射线图像，心脏大小正常，肺部看起来很清晰。已经排除了肺炎、积液、水肿、气胸、腺病、结节或肿块的存在。该发现表明一切正常。通过该报告，是否能诊断出肺炎？请回答是或否：", "否"],
        ["X光片显示出正常大小和轮廓的心胸廓线。没有气胸或大量胸腔积液。然而，右心尖有肿块状不透明，这表明恶性肿瘤恶化或恶性肿瘤合并阻塞性肺炎。可能需要进一步的评估和测试以确认诊断并决定适当的治疗方案。通过该报告，是否能诊断出肺炎？请回答是或否：", "是"],
        ["胸部X光显示心脏大小正常，肺部清晰，没有腺体病变、结节或肿块的迹象。另外，没有肺炎、渗出、水肿、气胸或结核的迹象。总的来说，X光片显示胸部正常，没有任何急性心肺功能异常的情况。通过该报告，是否能诊断出肺炎？请回答是或否：", "否"],
        ["X光图像显示没有心肺疾病的急性发现。心脏大小在正常范围内，而纵膈似乎在正常范围内。没有胸腔积液、气胸或局灶性空隙不透明，说明有肺炎。通过该报告，是否能诊断出肺炎？请回答是或否：", "是"]
    ]


    with torch.no_grad():
        # TODO 多轮对话获得bool结果
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


        input_text = f"{answer} 通过该报告，是否能诊断出肺炎？请回答是或否："
        answer, _history, _torch_image = chat(
            None,
            model,
            tokenizer,
            input_text,
            history=history_prompt,
            image=input_image,
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            english=False,
        )
        result = extract_bool_from_answer(answer)
    return result, answer # result int, answer str

def main(args):
    random.seed(110)
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
    tokenizer = AutoTokenizer.from_pretrained("/home/qianq/model/chatglm-6b", trust_remote_code=True)

    ## validation_data initialize
    base_dir = "/home/qianq/data/mimic-pa-512/mimic-pa-512/valid"
    label_df = pd.read_csv(f'{base_dir}/metadata.csv')
    label_df = label_df[['file_name', 'Pneumonia']]
    file_name_list = os.listdir(base_dir)[:-2]
    # random.shuffle(file_name_list)

    result_list = []
    for i in trange(len(file_name_list)):
    # for i in trange(10):
        file_name = file_name_list[i]
        image_path = f"{base_dir}/{file_name}"

        if not file_name.endswith('.jpg'):
            continue
        label = label_df[label_df['file_name']==file_name]['Pneumonia'].to_list()[0]
        label = 1 if label else 0
        pred_target, answer = get_result(image_path)
        result_list.append([
            file_name,
            label,
            pred_target,
            int(label==pred_target),
            answer,
        ])

    df_result = pd.DataFrame(result_list)
    df_result.columns = ['file_name', 'label', 'pred', 'is_correct', 'gen_answer']
    correct_ratio = sum(df_result['is_correct'])/len(df_result)
    print(correct_ratio)
    df_result.to_csv('eval_classification_results.csv')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--ckpt_path", type=str)
    args = parser.parse_args()

    main(args)
