import os
import json
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
            text = item['generated']
            self.texts.append(
                tokenizer(
                    text,
                    padding="max_length",  # 填充到最大长度
                    max_length=128,  # 经过数据分析，最大长度为127
                    truncation=True,
                    return_tensors="pt",
                )
            )
            self.labels.append(item['is_covid'])
        
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    


class AverageMeter:
    ''' Computes and stores the average and current value '''
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


def read_reports(path):
    with open(path) as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]


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
    raw_splits = ' '.join(raw_splits)
    generate_splits = ' '.join(generate_splits)
    reference = []  # 给定标准报告
    candidate = []  # 网络生成的报告
    # 计算BLEU
    reference.append(raw_splits.split())
    candidate = (generate_splits.split())
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
        raw = report['label']
        generated = report['generated']
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

    print('BLEU@1 :%f' % bleu1_metric.avg)
    print('BLEU@2 :%f' % bleu2_metric.avg)
    print('BLEU@3 :%f' % bleu3_metric.avg)
    print('BLEU@4 :%f' % bleu4_metric.avg)
    print('METEOR :%f' % meteor_metric.avg)

    print('%.2f' % (bleu1_metric.avg*100), '%')
    print('%.2f' % (bleu2_metric.avg*100), '%')
    print('%.2f' % (bleu3_metric.avg*100), '%')
    print('%.2f' % (bleu4_metric.avg*100), '%')
    print('%.2f' % (meteor_metric.avg*100), '%')


def calc_clinical_efficacy(reports, bert_ckpt_path):
    bert_name = "bert-base-chinese"
    model_path = f"/home/qianq/model/{bert_name}"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    batch_size = 32
    # # 定义模型
    model = BertClassifier(model_path)
    model.load_state_dict(
        torch.load(bert_ckpt_path)
    )

    device = "cuda"
    model = model.to(device)

    # # 验证数据集
    eval_dataset = MyDataset(reports=reports, tokenizer=tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    model.eval()
    total_acc_val = 0
    with torch.no_grad():
        for inputs, labels in eval_loader:
            input_ids = inputs['input_ids'].squeeze(1).to(device) # torch.Size([32, 35])
            masks = inputs['attention_mask'].to(device) # torch.Size([32, 1, 35])
            labels = labels.to(device)
            output = model(input_ids, masks)
            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_val += acc
        print(f'Covid Classification Accuracy: {100 * total_acc_val / len(eval_dataset): .2f}%')

    # for report in reports:
    #     # raw = report['label']
    #     generated = report['generated']
    #     clf_label = report['is_covid']
    #     print(clf_label)
    #     model(generated)
    #     break




if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    seed = 1997
    base_path = "/home/qianq/mycodes/VisualGLM-6B/reports/COV-CTR"
    file_name = "finetune-visualglm-6b-qformer-cls-fusion+llm-lora-6000.jsonl"
    bert_ckpt_path = '/home/qianq/mycodes/VisualGLM-6B/checkpoints/bert-clf/last.pt'
    reports_path = f"{base_path}-seed{seed}/{file_name}"
    reports = read_reports(reports_path)
    calc_clinical_efficacy(reports=reports, bert_ckpt_path=bert_ckpt_path)
    calc_nlg_metrics(reports)
