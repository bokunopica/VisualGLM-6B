import json
with open("/home/qianq/data/OpenI-zh/openi-zh.json") as f:
    buf = f.read()
data_list = json.loads(buf)['annotations']


res_list = []
for data in data_list:
    if "肺炎"in data['caption']:
        print(data['caption'])
        res_list.append(data['caption'])

with open("test.txt", 'w') as f:
    f.write(json.dumps(res_list))