import json
from tqdm import tqdm
import random


def main(file_path, save_path):
    with open(file_path) as f:
        data = json.load(f)

    data_info = []
    for i in tqdm(range(len(data['annotations']))):
        prompt_temp = [
            '通过这张胸部X光影像可以诊断出什么？',
            '这张图片的背景里有什么内容？',
            '详细描述一下这张图片',
            '看看这张图片并描述你注意到的内容',
            '请提供图片的详细描述',
            '你能为我描述一下这张图片的内容吗？'
            # 通过这张胸部X光影像可以诊断出肺炎吗？请回答是或否：
        ]
        img = data['annotations'][i]['image_id']
        prompt = random.choice(prompt_temp)
        label = data['annotations'][i]['caption']
        json_data = {
            'img': './data/Xray/'+str(img)+'.png',
            'prompt': prompt,
            'label': str(label)
        }
        data_info.append(json_data)

    with open(save_path, 'w+') as f1:
        json.dump(data_info, f1)

if __name__ == "__main__":
    file_path = "/home/qianq/data/OpenI-zh-resize-384/images/openi-zh.json"
    save_path = "/home/qianq/data/OpenI-zh-resize-384/images/openi-zh-prompt.json"
    main(file_path, save_path)
    