import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers import AdamW
from sklearn.metrics import accuracy_score
import os
import pandas as pd


# 数据集类
class CommentDataset(Dataset):
    def __init__(self, comments, scores, tokenizer, max_len):
        self.comments = comments
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = self.comments[item]
        score = self.scores[item]
        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor([score])
        }


# 定义模型
class BertForSentimentAnalysis(nn.Module):
    def __init__(self, n_classes):
        super(BertForSentimentAnalysis, self).__init__()
        self.bert = BertModel.from_pretrained("D:\\pythonproject1\\cold\\bert-chinese")
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        # 取CLS标记的输出，它是第一个标记的输出
        cls_output = bert_output.last_hidden_state[:, 0, :]
        # 应用dropout
        cls_output = self.dropout(cls_output)
        # 应用线性层
        output = self.out(cls_output)
        return output


# 训练函数
def train_model(epoch, model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for idx, batch in enumerate(data_loader):
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        model.zero_grad()
        outputs = model(b_input_ids, b_input_mask)
        loss = nn.BCEWithLogitsLoss()(outputs, b_labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if idx % 1000 == 0 and idx != 0:
            print('Train Epoch: {} ({})'.format(epoch, idx))

    return total_loss / len(data_loader)


# 评估函数
def evaluate_model(model, data_loader, device):
    model.eval()
    total_score = 0
    with torch.no_grad():
        for batch in data_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            outputs = model(b_input_ids, b_input_mask)
            predictions = torch.sigmoid(outputs).squeeze().tolist()
            total_score += accuracy_score([b >= 0.5 for b in b_labels.cpu().numpy()], [p >= 0.5 for p in predictions])
    return total_score / len(data_loader)


def GetData(path,tokenizer,max_len):
    total_text = pd.read_csv(path[0], sep='\t').values.tolist()
    # 创建两个空列表来存储标签和文本
    scores = []
    comments = []

    for item in total_text:
        fields = item[0].split(',')  # 分割每个元素的第一项
        label = int(fields[3])  # 将第四个字段转换为整数并存储到labels列表
        text = fields[4]  # 将第五个字段存储到texts列表
        scores.append(label)
        comments.append(text)
    dataset = CommentDataset(comments, scores, tokenizer, max_len)
    return dataset


# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('D:\\pythonproject1\\cold\\bert-chinese')
    max_len = 128

    # 假设 comments 和 scores 分别是你的评论列表和对应的分数列表
    train_path = [os.path.join('./COLDataset-main/COLDataset', i) for i in ["train.csv"]]
    test_path = [os.path.join('./COLDataset-main/COLDataset', i) for i in ["dev.csv"]]

    train_dataset = GetData(train_path,tokenizer,max_len)
    test_dataset = GetData(test_path, tokenizer, max_len)
    # 创建数据集和数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 初始化模型、优化器和损失函数
    model = BertForSentimentAnalysis(n_classes=1).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 训练和评估
    epochs = 3
    for epoch in range(epochs):
        train_loss = train_model(epoch, model, train_loader, optimizer, device)
        test_score = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Test Score: {test_score}")
        torch.save(model.state_dict(), "model/cold_bert_{}.pkl".format(epoch))


if __name__ == "__main__":
    main()

"""
Train Epoch: 0 (1000)
Train Epoch: 0 (2000)
Train Epoch: 0 (3000)
Train Epoch: 0 (4000)
Train Epoch: 0 (5000)
Train Epoch: 0 (6000)
Epoch 1, Train Loss: 0.2697081325741645, Test Score: 0.9059390547263682
Train Epoch: 1 (1000)
Train Epoch: 1 (2000)
Train Epoch: 1 (3000)
Train Epoch: 1 (4000)
Train Epoch: 1 (5000)
Train Epoch: 1 (6000)
Epoch 2, Train Loss: 0.17585725006011757, Test Score: 0.9193097014925373
Train Epoch: 2 (1000)
Train Epoch: 2 (2000)
Train Epoch: 2 (3000)
Train Epoch: 2 (4000)
Train Epoch: 2 (5000)
Train Epoch: 2 (6000)
Epoch 3, Train Loss: 0.11627531176787191, Test Score: 0.9261504975124378
"""
