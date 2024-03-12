import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from torch import nn
from transformers import BertModel


class MyDataset(Dataset):
    def __init__(self, path, tokenizer):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # tokenizer分词后可以被自动汇聚
        self.texts = []
        self.labels = []
        for item in data:
            text = item['label']
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
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_model(model, save_path, save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))


if __name__ == "__main__":
    bert_name = "bert-base-chinese"
    model_path = f"/home/qianq/model/{bert_name}"
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 训练超参数
    epoch = 10
    batch_size = 32
    lr = 1e-5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_seed = 20240306
    save_path = './checkpoints/bert-clf'
    setup_seed(random_seed)

    # 定义模型
    model = BertClassifier()
    model.load_state_dict(
        torch.load()
    )


    # # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(device)
    criterion = criterion.to(device)


    
    # # 构建数据集
    train_dataset = MyDataset(path='/home/qianq/data/COV-CTR/train.json', tokenizer=tokenizer)
    eval_dataset = MyDataset(path='/home/qianq/data/COV-CTR/eval.json', tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    # # 训练
    best_dev_acc = 0
    for epoch_num in range(epoch):
        total_acc_train = 0
        total_loss_train = 0
        for inputs, labels in tqdm(train_loader):
            input_ids = inputs['input_ids'].squeeze(1).to(device) # torch.Size([32, 35])
            masks = inputs['attention_mask'].to(device) # torch.Size([32, 1, 35])
            labels = labels.to(device)
            output = model(input_ids, masks)

            batch_loss = criterion(output, labels)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_train += acc
            total_loss_train += batch_loss.item()

        # ----------- 验证模型 -----------
        model.eval()
        total_acc_val = 0
        total_loss_val = 0
        
        with torch.no_grad():
            for inputs, labels in eval_loader:
                input_ids = inputs['input_ids'].squeeze(1).to(device) # torch.Size([32, 35])
                masks = inputs['attention_mask'].to(device) # torch.Size([32, 1, 35])
                labels = labels.to(device)
                output = model(input_ids, masks)

                batch_loss = criterion(output, labels)
                acc = (output.argmax(dim=1) == labels).sum().item()
                total_acc_val += acc
                total_loss_val += batch_loss.item()
            
            print(f'''Epochs: {epoch_num + 1} 
            | Train Loss: {total_loss_train / len(train_dataset): .3f} 
            | Train Accuracy: {total_acc_train / len(train_dataset): .3f} 
            | Val Loss: {total_loss_val / len(eval_dataset): .3f} 
            | Val Accuracy: {total_acc_val / len(eval_dataset): .3f}''')
            
            # 保存最优的模型
            if total_acc_val / len(eval_dataset) > best_dev_acc:
                best_dev_acc = total_acc_val / len(eval_dataset)
                save_model(model=model, save_path=save_path, save_name='best.pt')
        model.train()

    # 保存最后的模型
    save_model(model=model, save_path=save_path,save_name='last.pt')

