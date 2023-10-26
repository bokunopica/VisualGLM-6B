import json
from tqdm import tqdm
import random


def main(file_path, save_train_path, save_test_path):
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
            'label': str(label),
        }
        data_info.append(json_data)

    random.shuffle(data_info)
    len_train = int(len(data_info) * 0.8)

    with open(save_train_path, 'w') as f1:
        json.dump(data_info[:len_train], f1)

    with open(save_test_path, 'w') as f1:
        json.dump(data_info[len_train:], f1)
    

if __name__ == "__main__":
    file_path = "/home/qianq/data/OpenI-zh-resize-384/images/openi-zh.json"
    save_train_path = "/home/qianq/data/OpenI-zh-resize-384/images/openi-zh-train-prompt.json"
    save_test_path = "/home/qianq/data/OpenI-zh-resize-384/images/openi-zh-test-prompt.json"
    main(file_path, save_train_path, save_test_path)
    