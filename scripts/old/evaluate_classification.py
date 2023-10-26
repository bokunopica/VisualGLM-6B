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
    # elif answer.startswith('否'):
    #     return 0
    else:
        return 0
    return -1

def get_result(image_path) -> bool:
    temperature = 1e-5
    max_length = 512
    top_p = 0.4
    top_k = 100
    with torch.no_grad():
        # TODO 多轮对话获得bool结果
        history = []
        # input_text = "通过该x光片是否能诊断出肺炎？请回答True or False:"
        input_text = "通过这张胸部X光影像可以诊断出肺炎吗？请回答是或否："
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
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

    ## validation_data initialize
    # base_dir = "/home/qianq/data/mimic-pa-512/mimic-pa-512/valid"
    base_dir = "/home/qianq/data/balance_mimic_pneumonia"
    
    label_df = pd.read_csv(f'{base_dir}/test_metadata.csv')
    label_df = label_df[['dicom_id', 'Pneumonia']]
    # file_name_list = os.listdir(base_dir)[:-2]
    dicom_id_list = label_df['dicom_id'].tolist()
    # random.shuffle(file_name_list)

    result_list = []
    for i in trange(len(dicom_id_list)):
        dicom_id = dicom_id_list[i]
        file_name = f"{dicom_id}.jpg"
        image_path = f"{base_dir}/{file_name}"

        if not file_name.endswith('.jpg'):
            continue
        label = label_df[label_df['dicom_id']==dicom_id]['Pneumonia'].to_list()[0]
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
    df_result.to_csv('eval_classification_results_new.csv')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--ckpt_path", type=str)
    args = parser.parse_args()

    main(args)
