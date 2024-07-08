#!/usr/bin/env python

import random
from PIL import Image
import os
import json
import pickle
from model import (
    is_chinese,
    get_infer_setting,
    generate_input,
    chat,
)
from tqdm import trange
import torch
from finetune_visualglm_image_mixins import FineTuneVisualGLMModel
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

def get_result(image_path, prompt=None, use_covid_tag=False, is_covid=None) -> str:
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
        input_text = prompt if prompt is not None else "通过这张胸部X光影像可以诊断出什么？"
        if use_covid_tag:
            # if is_covid:
            #     disease_prompt = "该患者患有新冠肺炎。"
            # else:
            #     disease_prompt = "该患者未患有新冠肺炎。"
            if is_covid:
                disease_prompt = "该位受检者患有肺炎。"
            else:
                disease_prompt = "该位受检者未患有肺炎。"
            input_text = f"{disease_prompt}{input_text}"
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
            is_covid=is_covid,
        )
    return answer # answer str

def main(args):
    # 固定随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
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
    # result_list = []

    base_dir = "/home/qianq/data/COV-CTR/"
    file_path = "/home/qianq/data/COV-CTR/eval.json"
    with open(file_path) as f:
        data = json.load(f)

    # TODO Bootstrap 验证方法
    if args.bootstrap:
        # 外部文件夹处理
        save_dir = check_save_dir(args.report_save_path, is_bootstrap=True)
        len_single_eval = int(len(data)*0.8)

        with open("/home/qianq/mycodes/VisualGLM-6B/bootstrap_index.list", 'rb') as f:
            shuffle_lists = pickle.load(f)
        
        for idx in range(10):
            f = open(f"{save_dir}/{idx+1}.jsonl", 'w', encoding='utf-8')
            shuffle_list = shuffle_lists[idx]
            shuffle_data = []
            # 验证集80%随机采样,采样10次
            for item in shuffle_list[:len_single_eval]:
                shuffle_data.append(data[item])
            save_path = f"{save_dir}/{idx+1}.jsonl"
            generate_reports(base_dir, save_path, shuffle_data)
            
    else:
        check_save_dir(args.report_save_path, is_bootstrap=False)
        generate_reports(base_dir, args.report_save_path, data)


def check_save_dir(save_path, is_bootstrap=False):
    """
    外部文件夹处理
    is_bootstrap: 
        true:
            COV-CTR-seed1997-{model_name}
        false:
            COV-CTR-seed1997
    
    """
    if is_bootstrap:
        save_dir = save_path.replace('.jsonl', '')
    else:
        save_dir = "/".join(save_dir.split('/')[:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def generate_reports(base_dir, save_path, data):
    """
    批量生成报告并储存
    """
    f = open(save_path, 'w', encoding='utf-8')
    for i in trange(len(data)):
        single_data = data[i]
        image_path = f"{base_dir}/{single_data['img'].split('/')[-1]}"
        generate_report = get_result(
            image_path, 
            prompt=single_data['prompt'], 
            use_covid_tag=args.use_covid_tag,
            is_covid=torch.LongTensor([single_data['is_covid']]).cuda()
        )
        single_data['generated'] = generate_report
        f.write(json.dumps(single_data, ensure_ascii=False))
        f.write('\n')
    f.close()
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--report_save_path", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_covid_tag", action="store_true")
    parser.add_argument("--vqa", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    main(args)
