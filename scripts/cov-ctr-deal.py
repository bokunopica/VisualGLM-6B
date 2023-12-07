import pandas as pd
import random
import json


def transfer_csv_to_json():
    random.seed(1007)
    base_path = "/home/qianq/data/COV-CTR"
    csv_path = f"{base_path}/reports_ZH_EN.csv"
    img_path = base_path + "/"
    df = pd.read_csv(csv_path)
    result_list = []
    prompt_temp = [
        '通过这张胸部X光影像可以诊断出什么？',
        '这张图片的背景里有什么内容？',
        '详细描述一下这张图片',
        '看看这张图片并描述你注意到的内容',
        '请提供图片的详细描述',
        '你能为我描述一下这张图片的内容吗？'
        # 通过这张胸部X光影像可以诊断出肺炎吗？请回答是或否：
    ]
    for idx, row in df.iterrows():
        result_list.append({
            "img": img_path + row['image_id'],
            "prompt": random.choice(prompt_temp),
            "label": row['findings'],
            "is_covid": row['COVID'],
        })
    random.shuffle(result_list)
    train_len = int(len(result_list)*0.8)
    train_path = f"{base_path}/train.json"
    eval_path = f"{base_path}/eval.json"
    with open(train_path, 'w') as f:
        f.write(json.dumps(result_list[:train_len], ensure_ascii=False))
    with open(eval_path, 'w') as f:
        f.write(json.dumps(result_list[train_len:], ensure_ascii=False))




if __name__ == "__main__":
    transfer_csv_to_json()