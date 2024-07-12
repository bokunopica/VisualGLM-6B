import os
import json
import pickle
import pandas as pd
import numpy as np
import scipy.stats as st
import jieba
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score as meteor
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, bert_name):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 10)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


class MyDataset(Dataset):
    def __init__(self, reports, tokenizer):
        # tokenizer分词后可以被自动汇聚
        self.texts = []
        self.labels = []
        for item in reports:
            text = item["generated"]
            self.texts.append(
                tokenizer(
                    text,
                    padding="max_length",  # 填充到最大长度
                    max_length=128,  # 经过数据分析，最大长度为127
                    truncation=True,
                    return_tensors="pt",
                )
            )
            self.labels.append(item["is_covid"])

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def read_reports_visualglm(path):
    with open(path) as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]


def read_reports_show_and_tell(path):
    """show and tell model reports"""
    # test info
    test_json_path = "/home/qianq/data/COV-CTR/eval.json"
    with open(test_json_path) as f:
        data_list = json.loads(f.read())  # csv_lines

    pd_reports = pd.read_csv(path)
    query_dict = {}

    for _, row in pd_reports.iterrows():
        query_dict[row["image_files"]] = row["caption"]

    for i in range(len(data_list)):
        gen = query_dict.get(data_list[i]["img"], None)
        if gen is None:
            raise Exception("no gen text")
        data_list[i]["generated"] = query_dict.get(data_list[i]["img"], None)
    return data_list


def read_reports(path, model_type):
    if model_type == "visualglm":
        read_func = read_reports_visualglm
    elif model_type == "show_and_tell":
        read_func = read_reports_show_and_tell
    elif model_type == "show_attend_and_tell":
        read_func = read_reports_visualglm
    else:
        raise Exception("model_type unknown")
    return read_func(path)


def bleu_score(raw_splits, generate_splits):
    """
    single sentence bleu score metric
    inputs:
        raw_splits: ['this', 'is', 'a', 'duck']
        generate_splits: ['this', 'is', 'a', 'duck']
    return:
        tuple(bleu@1, bleu@2, bleu@3, bleu@4)
    """
    # reference是标准答案 是一个列表，可以有多个参考答案，每个参考答案都是分词后使用split()函数拆分的子列表
    # # 举个reference例子
    # reference = [['this', 'is', 'a', 'duck']]
    raw_splits = " ".join(raw_splits)
    generate_splits = " ".join(generate_splits)
    reference = []  # 给定标准报告
    candidate = []  # 网络生成的报告
    # 计算BLEU
    reference.append(raw_splits.split())
    candidate = generate_splits.split()
    score1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    score2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
    score3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
    score4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
    reference.clear()
    return score1, score2, score3, score4


def meteor_score(raw_splits, generate_splits):
    # raw_splits = jieba.cut(raw)
    # generate_splits = jieba.cut(generate)
    result = meteor([raw_splits], generate_splits)
    return result


def calc_nlg_metrics(reports):
    bleu1_metric = AverageMeter()
    bleu2_metric = AverageMeter()
    bleu3_metric = AverageMeter()
    bleu4_metric = AverageMeter()
    meteor_metric = AverageMeter()
    for report in reports:
        raw = report["label"]
        generated = report["generated"]
        # 分词
        raw_splits = list(jieba.cut(raw))
        generate_splits = list(jieba.cut(generated))
        # raw_splits = [_ for _ in raw_splits]
        # generate_splits = [_ for _ in generate_splits]
        bleu1, bleu2, bleu3, bleu4 = bleu_score(raw_splits, generate_splits)
        bleu1_metric.update(bleu1, n=1)
        bleu2_metric.update(bleu2, n=1)
        bleu3_metric.update(bleu3, n=1)
        bleu4_metric.update(bleu4, n=1)
        meteor_metric.update(meteor_score(raw_splits, generate_splits), n=1)

    print("BLEU@1 :%f" % bleu1_metric.avg)
    print("BLEU@2 :%f" % bleu2_metric.avg)
    print("BLEU@3 :%f" % bleu3_metric.avg)
    print("BLEU@4 :%f" % bleu4_metric.avg)
    print("METEOR :%f" % meteor_metric.avg)

    print("%.2f" % (bleu1_metric.avg * 100), "%")
    print("%.2f" % (bleu2_metric.avg * 100), "%")
    print("%.2f" % (bleu3_metric.avg * 100), "%")
    print("%.2f" % (bleu4_metric.avg * 100), "%")
    print("%.2f" % (meteor_metric.avg * 100), "%")
    return [
        bleu1_metric.avg,
        bleu2_metric.avg,
        bleu3_metric.avg,
        bleu4_metric.avg,
        meteor_metric.avg,
    ]


def calc_clinical_efficacy(reports, model, tokenizer, device):
    batch_size = 32
    # # 验证数据集
    eval_dataset = MyDataset(reports=reports, tokenizer=tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    model.eval()
    total_acc_val = 0
    with torch.no_grad():
        for inputs, labels in eval_loader:
            input_ids = (
                inputs["input_ids"].squeeze(1).to(device)
            )  # torch.Size([32, 35])
            masks = inputs["attention_mask"].to(device)  # torch.Size([32, 1, 35])
            labels = labels.to(device)
            output = model(input_ids, masks)
            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_val += acc
        accuracy = total_acc_val / len(eval_dataset)
        print(f"Covid Classification Accuracy: {100*accuracy: .2f}%")
        return accuracy

    # for report in reports:
    #     # raw = report['label']
    #     generated = report['generated']
    #     clf_label = report['is_covid']
    #     print(clf_label)
    #     model(generated)
    #     break


def confidence_interval(fb_data, distribution, confidence_level):
    """
    distribution: "T" or "N"
    """

    def format_num(num):
        return round(num * 100, 2)

    input_dict = {
        "confidence": confidence_level,
        "loc": np.mean(fb_data),
        "scale": st.sem(fb_data),
    }
    if distribution == "T":
        dist_obj = st.t
        input_dict.update({"df": len(fb_data) - 1})
    else:
        dist_obj = st.norm
    # 置信水平create 95% confidence interval
    # confidence_level = 0.95
    # t.interval() 计算置信区间 （n<30）
    lower, upper = dist_obj.interval(**input_dict)
    lower = format_num(lower)
    upper = format_num(upper)
    # print(f"{distribution}:({lower},{upper})")
    return lower, upper


def get_all_confidence_interval(
    # b1_total,
    # b2_total,
    # b3_total,
    # b4_total,
    # meteor_total,
    # ce_total,
    **kwargs
):
    """
    **kwargs:
        b1_total,
        b2_total,
        b3_total,
        b4_total,
        meteor_total,
        ce_total,
    """
    dist = "T"
    for key, value in kwargs.items():
        name = key.split('_')[0]
        lower, upper = confidence_interval(
            fb_data=value,
            distribution=dist,
            confidence_level=0.95
        )
        print(f"{name}\t: {lower, upper}")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    seed = 1997
    base_path = "/home/qianq/mycodes/VisualGLM-6B/reports"

    bert_ckpt_path = "/home/qianq/mycodes/VisualGLM-6B/checkpoints/bert-clf/last.pt"
    model_type = "visualglm"
    # is_bootstrap = True
    is_bootstrap = True
    # model_type = "show_and_tell"
    # model_type = "show_attend_and_tell"

    ## bert classifier
    bert_name = "bert-base-chinese"
    model_path = f"/home/qianq/model/{bert_name}"
    bert_tokenizer = BertTokenizer.from_pretrained(model_path)
    # # 定义模型
    bert_model = BertClassifier(model_path)
    bert_model.load_state_dict(torch.load(bert_ckpt_path))
    device = "cuda"
    bert_model = bert_model.to(device)

    if is_bootstrap:
        if model_type == "visualglm":
            file_name = "finetune-visualglm-6b-qformer-cls-fusion+llm-lora-6000.jsonl"
            reports_path = f"{base_path}/COV-CTR-seed{seed}-final/{file_name}"
            reports = read_reports(reports_path, model_type=model_type)
            len_single_eval = int(len(reports) * 0.8)
            with open(
                "/home/qianq/mycodes/VisualGLM-6B/bootstrap_index.list", "rb"
            ) as f:
                shuffle_lists = pickle.load(f)
            n = 10
            b1_total = []
            b2_total = []
            b3_total = []
            b4_total = []
            meteor_total = []
            ce_total = []

            for idx in range(n):
                shuffle_list = shuffle_lists[idx]
                shuffle_data = []
                # 验证集80%随机采样,采样10次
                for item in shuffle_list[:len_single_eval]:
                    shuffle_data.append(reports[item])
                ce_total.append(
                    calc_clinical_efficacy(
                        reports=shuffle_data,
                        model=bert_model,
                        tokenizer=bert_tokenizer,
                        device=device,
                    )
                )
                nlgm_list = calc_nlg_metrics(shuffle_data)
                b1_total.append(nlgm_list[0])
                b2_total.append(nlgm_list[1])
                b3_total.append(nlgm_list[2])
                b4_total.append(nlgm_list[3])
                meteor_total.append(nlgm_list[4])

            get_all_confidence_interval(
                b1_total=b1_total,
                b2_total=b2_total,
                b3_total=b3_total,
                b4_total=b4_total,
                meteor_total=meteor_total,
                ce_total=ce_total,
            )
    else:
        if model_type == "visualglm":
            file_name = "finetune-visualglm-6b-qformer-cls-fusion-6000.jsonl"
            reports_path = f"{base_path}/COV-CTR-seed{seed}-final/{file_name}"

        elif model_type == "show_and_tell":
            # show_and_tell
            file_name = "results_190.csv"
            reports_path = f"{base_path}/show_and_tell/{file_name}"
        else:
            # show_attend_and_tell
            file_name = "show_attend_tell.jsonl"
            reports_path = f"{base_path}/show_attend_and_tell/{file_name}"
        reports = read_reports(reports_path, model_type=model_type)
        calc_clinical_efficacy(
            reports=reports, model=bert_model, tokenizer=bert_tokenizer, device=device
        )
        calc_nlg_metrics(reports)
