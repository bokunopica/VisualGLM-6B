import json


# report_name = "finetune-visualglm-6b-qformer-cls-fusion-6000.jsonl"
report_name = "finetune-visualglm-6b-llm-lora-6000.jsonl"

# specified_img = "2145.png"
specified_img = "120.png"

with open(f"reports/COV-CTR-seed1997/{report_name}") as f:
    lines = f.readlines()

for line in lines:
    if specified_img not in line:
        continue
    print('---------------------')
    line = json.loads(line)
    print(line['img'])
    print(line['label'])
    print(line['generated'])