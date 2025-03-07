import pandas as pd
import numpy as np
import re

def replace_sequence_with_protein_id(matched_proteins_file, protein_file):
    # 读取 matched_proteins.csv 文件
    matched_proteins_data = pd.read_csv(matched_proteins_file)

    # 读取 sars-cov-2_protein.faa 文件
    with open(protein_file, 'r') as f:
        protein_data = f.read().split('>')[1:]  # 分割蛋白质序列数据

    # 创建字典，存储 protein.faa 中的序列与对应的蛋白质 ID
    protein_dict = {}
    for entry in protein_data:
        lines = entry.strip().split('\n')
        #protein_id = re.match(r"^[^\[]+", lines[0]).group(0).strip()
        protein_id = lines[0].split()[0]  # 获取蛋白质序列的 ID
        sequence = ''.join(lines[1:])  # 获取蛋白质序列
        protein_dict[sequence] = protein_id

    # 用蛋白质 ID 替换 matched_proteins.csv 的第一列序列
    matched_proteins_data.iloc[:, 0] = matched_proteins_data.iloc[:, 0].apply(lambda seq: protein_dict.get(seq, seq))

    # 保存修改后的文件，直接覆盖原文件
    matched_proteins_data.to_csv(matched_proteins_file, index=False)
    print(f"序列已替换并保存到 {matched_proteins_file}")

# 用法示例
matched_proteins_file = 'new_matched_proteins.csv'
protein_file = 'sars-cov-2_protein.faa'

replace_sequence_with_protein_id(matched_proteins_file, protein_file)
