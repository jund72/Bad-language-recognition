import sqlite3
import tkinter as tk
from tkinter import scrolledtext
import random
from get_modelbasedoncold import CommentDataset, BertForSentimentAnalysis
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertTokenizer

# 创建或连接到 SQLite 数据库
conn = sqlite3.connect('comments.db')
c = conn.cursor()

# 创建一个表来存储评论和用户判断
c.execute('''
CREATE TABLE IF NOT EXISTS comments (
    comment TEXT UNIQUE,
    bad_votes INTEGER DEFAULT 0,
    safe_votes INTEGER DEFAULT 0
)
''')
conn.commit()

tokenizer = BertTokenizer.from_pretrained('D:\\pythonproject1\\cold\\bert-chinese')
max_len = 128

# 创建主窗口
root = tk.Tk()
root.title("不良言论识别")

# 显示评论
comment_text = scrolledtext.ScrolledText(root, width=50, height=10)
comment_text.pack(pady=20)


ai_label = tk.Label(root, text="", font=("Arial", 14))
ai_label.pack(pady=20)

percent_label = tk.Label(root, text="", font=("Arial", 14))
percent_label.pack(pady=20)

# 不良言论按钮
bad_button = tk.Button(root, text="不良言论", command=lambda: user_feedback(0))
bad_button.pack(side=tk.LEFT, padx=20, pady=20)

# 安全言论按钮
safe_button = tk.Button(root, text="安全言论", command=lambda: user_feedback(1))
safe_button.pack(side=tk.RIGHT, padx=20, pady=20)


def show_comment():
    # 随机选择一条评论并显示
    comment = random.choice(bili_comments)
    comment_text.delete('1.0', tk.END)  # 清空文本框
    comment_text.insert(tk.END, comment)  # 插入新的评论
    return comment


def show_label(comment):
    dataset = CommentDataset([comment], [0], tokenizer, max_len)
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            outputs = model(b_input_ids, b_input_mask)
            predictions = torch.sigmoid(outputs).squeeze().tolist()
    if np.array(predictions >= 0.5):
        label = "机器认为：不良言论"
    else:
        label = "机器认为：安全言论"
    ai_label.config(text=label)


def user_feedback(judgment):
    current_comment = show_comment()
    c.execute('INSERT OR IGNORE INTO comments (comment) VALUES (?)', (current_comment,))
    conn.commit()
    # 构建更新语句
    if judgment == 0:
        column_name = 'bad_votes'
    else:
        column_name = 'safe_votes'

    # 执行更新语句
    c.execute(f'UPDATE comments SET {column_name} = {column_name} + 1 WHERE comment = ?', (current_comment,))
    conn.commit()

    show_label(current_comment)
    show_vote_percentage(current_comment)


def show_vote_percentage(comment):
    c.execute('SELECT bad_votes, safe_votes FROM comments WHERE comment = ?', (comment,))
    votes = c.fetchone()
    bad_votes, safe_votes = votes
    total_votes = bad_votes + safe_votes
    bad_percentage = (bad_votes / total_votes * 100) if total_votes > 0 else 0
    safe_percentage = (safe_votes / total_votes * 100) if total_votes > 0 else 0
    vote_label = f"不良言论: {bad_percentage:.2f}% 安全言论: {safe_percentage:.2f}%"
    percent_label.config(text=f"上条投票结果: {vote_label}")


df = pd.DataFrame(columns=['Comment', 'UserJudgment'])
path = [os.path.join('./comments', i) for i in ["bili_comments"]]
text = pd.read_csv(path[0], sep='\t').values.tolist()
bili_comments = []
for item in text:
    fields = item[0].split(',')  # 分割每个元素的第一项
    comment = fields[3]  # 将第五个字段存储到texts列表
    bili_comments.append(comment)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建模型实例
model = BertForSentimentAnalysis(n_classes=1).to(device)
model.load_state_dict(torch.load('model/cold_bert_2.pkl'))

# 显示初始评论
comment = show_comment()
show_label(comment)

# 运行主循环
root.mainloop()

# 关闭数据库连接
conn.close()
