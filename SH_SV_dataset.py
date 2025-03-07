# SH_SV_dataset.py
import random

import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, n_classes, n_samples):
        self.dataset = dataset
        self.labels = np.array([item['label'] for item in dataset])
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])

        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = len(self.dataset) // n_samples
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = len(self.dataset) // self.batch_size
        indices = []
        for _ in range(self.count):
            batch_indices = []
            for class_ in self.labels_set:
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
                batch_indices.extend(self.label_to_indices[class_][
                                     self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                               class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
            indices.extend(batch_indices)
        return iter(indices)

    def __len__(self):
        return self.count * self.batch_size


class SH_SV_Dataset(Dataset):
    def __init__(self, pd_data):
        self.data = pd_data.copy()  # 使用.copy()确保这里的self.data是一个全新的副本

        # 安全地修改self.data
        if 'additional_vector' in self.data.columns:
            self.data['additional_vector'] = self.data['additional_vector'].apply(eval)

        # 安全地修改self.data
        if 'feature_SH' in self.data.columns:
            self.data['feature_SH'] = self.data['feature_SH'].apply(eval)

        # 安全地修改self.data
        if 'feature_SV' in self.data.columns:
            self.data['feature_SV'] = self.data['feature_SV'].apply(eval)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        S_H = self.data.iloc[idx]['S_H']
        S_V = self.data.iloc[idx]['S_V']
        feature_SH = torch.tensor(self.data.iloc[idx]['feature_SH'], dtype=torch.float)
        feature_SV = torch.tensor(self.data.iloc[idx]['feature_SV'], dtype=torch.float)

        return {
            'S_H': S_H,  # You might need to transform these strings into a suitable format
            'S_V': S_V,  # Ditto
            'feature_SH': feature_SH,
            'feature_SV': feature_SV
        }

    def SH_SV_Dataset_collate_fn(self, batch):
        # Collect each part of the data into separate lists
        S_H_batch = [item['S_H'] for item in batch]
        S_V_batch = [item['S_V'] for item in batch]
        feature_SH_batch = [item['feature_SH'] for item in batch]
        feature_SV_batch = [item['feature_SV'] for item in batch]

        # Transform lists of S_H and S_V here as needed (e.g., to tensors)
        # S_H_batch = torch.tensor(S_H_batch, dtype=torch.float)
        # S_V_batch = torch.tensor(S_V_batch, dtype=torch.float)

        # Stack all additional_vector tensors into one tensor
        feature_SH_batch = torch.stack(feature_SH_batch)
        feature_SV_batch = torch.stack(feature_SV_batch)

        return {
            'S_H': S_H_batch,
            'S_V': S_V_batch,
            'feature_SH': feature_SH_batch,
            'feature_SV': feature_SV_batch
        }


class SH_SV_Relation_Dataset(SH_SV_Dataset):
    def __init__(self, pd_data, negative_sample_rate=0.5):
        super().__init__(pd_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        S_H = row['S_H']
        S_V = row['S_V']
        feature_SH = row['feature_SH']
        feature_SV = row['feature_SV']
        label = row['label']
        sample = {
            'S_H': S_H,
            'S_V': S_V,
            'feature_SH': feature_SH,
            'feature_SV': feature_SV,
            'label': label
        }

        return sample

    def SH_SV_Relation_Dataset_collate_fn(self, batch):
        S_H_batch = [item['S_H'] for item in batch]
        S_V_batch = [item['S_V'] for item in batch]
        feature_SH_batch = torch.stack([torch.tensor(item['feature_SH']) for item in batch])
        feature_SV_batch = torch.stack([torch.tensor(item['feature_SV']) for item in batch])
        labels_batch = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        return {
            'S_H': S_H_batch,
            'S_V': S_V_batch,
            'feature_SH': feature_SH_batch,
            'feature_SV': feature_SV_batch,
            'labels': labels_batch
        }


def get_dataloaders_for_fold(csv_file, fold_index, batch_size=16,
                             num_splits=10):
    data = pd.read_csv(csv_file)
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    for i, (train_val_index, test_index) in enumerate(kf.split(data)):
        if i == fold_index:
            train_val_data = data.iloc[train_val_index]
            test_data = data.iloc[test_index]

            # Split train_val_data further into train and validation sets
            train_size = int(0.9 * len(train_val_data))
            train_data = train_val_data[:train_size]
            valid_data = train_val_data[train_size:]

            train_dataset = SH_SV_Dataset(train_data)
            valid_dataset = SH_SV_Dataset(valid_data)
            test_dataset = SH_SV_Dataset(test_data)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          collate_fn=train_dataset.SH_SV_Dataset_collate_fn, drop_last=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=valid_dataset.SH_SV_Dataset_collate_fn, drop_last=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=test_dataset.SH_SV_Dataset_collate_fn, drop_last=True)

            return train_dataloader, valid_dataloader, test_dataloader


def get_relation_dataloaders_for_fold(csv_file, fold_index, batch_size=16, num_splits=10):
    data = pd.read_csv(csv_file)
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    for i, (train_val_index, test_index) in enumerate(kf.split(data)):
        if i == fold_index:
            train_val_data = data.iloc[train_val_index]
            test_data = data.iloc[test_index]

            # Split train_val_data further into train and validation sets
            train_size = int(0.9 * len(train_val_data))
            train_data = train_val_data[:train_size]
            valid_data = train_val_data[train_size:]

            # 使用 SH_SV_Relation_Dataset 初始化数据集
            train_dataset = SH_SV_Relation_Dataset(train_data)
            valid_dataset = SH_SV_Relation_Dataset(valid_data)
            test_dataset = SH_SV_Relation_Dataset(test_data)

            # train sampler
            train_sampler = BalancedBatchSampler(train_dataset, n_classes=2, n_samples=8)  # 假设你希望每类中有8个样本
            # 创建 DataLoader
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size  # , sampler=train_sampler
                                          , shuffle=True,
                                          collate_fn=train_dataset.SH_SV_Relation_Dataset_collate_fn, drop_last=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=valid_dataset.SH_SV_Relation_Dataset_collate_fn, drop_last=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=test_dataset.SH_SV_Relation_Dataset_collate_fn, drop_last=True)

            return train_dataloader, valid_dataloader, test_dataloader


if __name__ == '__main__':
    # Example usage
    csv_file = 'data/SH_SV_feature_with_vectors_0.5.csv'  # Your CSV file path
    fold_index = 0  # Choose which fold to use (0-9 for ten-fold CV)
    #
    # train_loader, valid_loader, test_loader = get_dataloaders_for_fold(csv_file,
    #                                                                    fold_index=fold_index,
    #                                                                    batch_size=16)
    # for item in test_loader:
    #     print(type(item))

    train_loader, valid_loader, test_loader = get_relation_dataloaders_for_fold(csv_file,
                                                                                fold_index=fold_index,
                                                                                batch_size=16)
    for item in train_loader:
        print(item)
