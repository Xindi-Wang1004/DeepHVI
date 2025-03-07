import pandas as pd
import numpy as np
import re

def match_and_save(task2_result_file, protein_file, output_file):
    # 读取 task2_result.csv 文件
    task2_result_data = pd.read_csv(task2_result_file, low_memory=False)

    # 读取 protein.faa 文件
    with open(protein_file, 'r') as f:
        protein_data = f.read().split('>')[1:]  # 分割蛋白质序列数据

    # 创建字典，存储 protein.faa 中的序列ID和序列数据
    protein_dict = {}
    for entry in protein_data:
        lines = entry.strip().split('\n')
        #protein_id = re.match(r"^\s*[^\[]+", lines[0]).group(0).strip()
        protein_id = lines[0].split()[0]  # 获取蛋白质序列的 ID
        sequence = ''.join(lines[1:])  # 获取蛋白质序列
        protein_dict[protein_id] = sequence

    # 遍历 task2_result.csv 中的 2-7 列，进行替换
    for col in range(1, 7):  # 遍历第2到第7列
        task2_result_data.iloc[:, col] = task2_result_data.iloc[:, col].apply(lambda seq: match_sequence_to_protein_id(seq, protein_dict))

    # 将替换后的数据保存到新的 CSV 文件
    task2_result_data.to_csv(output_file, index=False)
    print(f"匹配结果已保存到 {output_file}")

def match_sequence_to_protein_id(seq, protein_dict):
    # 确保 seq 是字符串类型，如果不是则跳过
    if isinstance(seq, str) and seq.strip():  # 如果 seq 是非空的字符串
        for protein_id, protein_sequence in protein_dict.items():
            if seq in protein_sequence:
                return protein_id  # 返回匹配的蛋白质 ID
    return np.nan  # 如果没有匹配到，返回 NaN

# 用法示例
task2_result_file = 'task1_result.csv'
protein_file = 'protein.faa'
output_file = 'new_matched_proteins.csv'

match_and_save(task2_result_file, protein_file, output_file)
