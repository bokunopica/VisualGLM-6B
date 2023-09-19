import json
from tqdm import tqdm
import random

def edit_error_jsonl(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    res = []
    for line in lines:
        line = json.loads(line)
        if isinstance(line['label'], list):
            line['label'] = line['label'][0]
        res.append(line)
    with open(file_path.replace('jsonl', 'json'), 'w') as f:
        f.write(json.dumps(res))


def main():
    edit_error_jsonl("/home/qianq/data/balance_mimic_pneumonia/train_metadata_final.jsonl")
    edit_error_jsonl("/home/qianq/data/balance_mimic_pneumonia/test_metadata_final.jsonl")
    
    

if __name__ == "__main__":
    main()
    