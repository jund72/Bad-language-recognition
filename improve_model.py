import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import AdamW
from get_modelbasedoncold import CommentDataset,BertForSentimentAnalysis,train_model
import warnings
import sqlite3

# 连接到 SQLite 数据库
conn = sqlite3.connect('comments.db')
c = conn.cursor()

# 查询 comments 表中的所有数据
c.execute('SELECT * FROM comments')

# 获取查询结果
comments = c.fetchall()
comments_list = []
labels_list = []
for comment in comments:
    comments_list.append(comment[0])
    if comment[1] > comment[2]:
        label = 0
    else:
        label = 1
    labels_list.append(label)

# 关闭数据库连接
conn.close()

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.models.bert.modeling_bert")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('D:\\pythonproject1\\cold\\bert-chinese')
PATH = 'model/cold_bert_2.pkl'
model = BertForSentimentAnalysis(n_classes=1).to(device)
max_len = 128

dataset = CommentDataset(comments_list, labels_list, tokenizer, max_len)
train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
optimizer = AdamW(model.parameters(), lr=2e-5)
model.load_state_dict(torch.load(PATH))
epochs = 3
for epoch in range(epochs):
    train_loss = train_model(epoch, model, train_loader, optimizer, device)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss}")
    torch.save(model.state_dict(), "model/bili_improved_bert_{}.pkl".format(epoch))


