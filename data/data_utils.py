# 加载数据
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from protain_feature import get_feature


def add_featrue_vector():
    df = pd.read_csv('SH_SV_feature.csv')

    features_SH = []
    features_SV = []

    # 提取所有的 S_H 和 S_V 序列
    sequences_SH = df['S_H'].tolist()
    sequences_SV = df['S_V'].tolist()

    # 生成特征
    features_SH = get_feature(sequences_SH, {'aaindex'})
    features_SV = get_feature(sequences_SV, {'aaindex'})

    df['feature_SH'] = features_SH
    df['feature_SV'] = features_SV

    df.to_csv('SH_SV_feature_with_vectors.csv', index=False)


def add_negative_sample():
    negative_sample_rate = 0.5
    df = pd.read_csv('SH_SV_feature_with_vectors.csv')

    # Create a set of positive sample pairs
    pairs = set([(row['S_H'], row['S_V']) for idx, row in df.iterrows()])

    def generate_negative_samples(pairs, data, negative_sample_rate=0.5):
        negative_samples = []
        all_sh_indices = list(range(len(data)))
        all_sv_indices = list(range(len(data)))

        while len(negative_samples) < int(len(data) * negative_sample_rate):
            sh_idx = random.choice(all_sh_indices)
            sv_idx = random.choice(all_sv_indices)
            sh = data.iloc[sh_idx]['S_H']
            sv = data.iloc[sv_idx]['S_V']
            if (sh, sv) not in pairs:
                feature_SH = data.iloc[sh_idx]['feature_SH']
                feature_SV = data.iloc[sv_idx]['feature_SV']

                # Construct a negative sample and label it as 0
                negative_samples.append({
                    'S_H': sh,
                    'S_V': sv,
                    'feature_SH': feature_SH,
                    'feature_SV': feature_SV,
                    'label': 0
                })
                # Remove selected indices to avoid repetition
                all_sh_indices.remove(sh_idx)
                all_sv_indices.remove(sv_idx)

        return pd.DataFrame(negative_samples)

    # Generate negative samples
    negative_samples_df = generate_negative_samples(pairs, data=df, negative_sample_rate=negative_sample_rate)

    # Label positive samples in df, all positive samples have label 1
    df['label'] = 1

    # Concatenate the positive samples df and negative_samples_df to get the complete dataset
    complete_df = pd.concat([df, negative_samples_df], ignore_index=True)
    print(len(complete_df))
    # Shuffle the complete dataframe randomly
    complete_df = complete_df.sample(frac=1).reset_index(drop=True)
    print(len(complete_df))

    # Save the shuffled complete dataset to a local CSV file
    complete_df.to_csv(f'SH_SV_feature_with_vectors_{negative_sample_rate}.csv', index=False)


def generate_partial_csv():
    df = pd.read_csv('SH_SV_feature_with_vectors.csv')

    def create_partial_df(data, sh_type, sv_type):
        if sh_type == 'full':
            sh_data = data['S_H']
        elif sh_type == 'front':
            sh_data = data['S_H'].apply(lambda x: x[:int(len(x) * 0.3)])
        elif sh_type == 'back':
            sh_data = data['S_H'].apply(lambda x: x[-int(len(x) * 0.3):])

        if sv_type == 'full':
            sv_data = data['S_V']
        elif sv_type == 'front':
            sv_data = data['S_V'].apply(lambda x: x[:int(len(x) * 0.3)])
        elif sv_type == 'back':
            sv_data = data['S_V'].apply(lambda x: x[-int(len(x) * 0.3):])

        partial_df = pd.DataFrame({
            'S_H': sh_data,
            'S_V': sv_data,
            'feature_SH': data['feature_SH'],
            'feature_SV': data['feature_SV'],
            'type': f'{sh_type}_{sv_type}'
        })
        return partial_df

    types = [
        ('full', 'front'),
        ('full', 'back'),
        ('front', 'full'),
        ('back', 'full')
    ]

    partial_dfs = []
    for idx, row in df.iterrows():
        sh_type, sv_type = random.choice(types)
        partial_df = create_partial_df(df.iloc[[idx]], sh_type, sv_type)
        tyep = partial_df['type']
        sh_type = tyep.item().split('_')[0]
        sv_type = tyep.item().split('_')[1]
        if sh_type != 'full':
            sequences_SH = partial_df['S_H'].tolist()
            features_SH = get_feature(sequences_SH, {'aaindex'})
            partial_df['feature_SH'] = features_SH
        if sv_type != 'full':
            sequences_SV = partial_df['S_V'].tolist()
            features_SV = get_feature(sequences_SV, {'aaindex'})
            partial_df['feature_SV'] = features_SV

        partial_dfs.append(partial_df)
    complete_partial_df = pd.concat(partial_dfs, ignore_index=True)
    complete_partial_df.to_csv('SH_SV_feature_with_vectors_partial.csv', index=False)


if __name__ == '__main__':
    # add_featrue_vector()
    # add_negative_sample()
    # 生成 部分模态补全的代码：
    generate_partial_csv()
    # 生成 部分模态补全的代码：
    # df = pd.read_csv('SH_SV_feature_with_vectors_0.5.csv')
    # print(df['label'].tail())  # Quick check to see if labels look correct
    # print(df['label'].isnull().sum())
