import pandas as pd
from Bio import SeqIO

# 1. 提取FASTA序列
sequences = []
with open("1.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        sequences.append(str(record.seq).replace("\n", ""))

# 2. 处理CSV文件
df = pd.read_csv("inference.csv", low_memory=False)

# 删除前四列
df = df.iloc[:, 4:]

# 添加序列到seq_b列
#df["seq_b"] = sequences[:len(df)]  # 确保序列数量匹配

# 对seq_a列去重（保留首次出现）
df = df.drop_duplicates(subset=["seq_a"])

# 保存结果
df.to_csv("inference_processed.csv", index=False)
