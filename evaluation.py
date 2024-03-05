import json
import jieba
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score as meteor


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

def calc_clinical_efficacy(reports):
    for report in reports:
        raw = report['label']
        generated = report['generated']
        label = report['is_covid']
        print(report)


if __name__ == "__main__":
    seed = 1919810
    base_path = "/home/qianq/mycodes/VisualGLM-6B/reports/COV-CTR-30k"
    file_name = "finetune-visualglm-6b-qformer-label-hint.jsonl"
    reports_path = f"{base_path}-seed{seed}/{file_name}"
    reports = read_reports(reports_path)
    calc_nlg_metrics(reports)
    # calc_clinical_efficacy(reports)
