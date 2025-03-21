import pandas as pd
import torch
import torch.nn.functional as F
import gc
from datetime import datetime
from ESM2SequenceTransformer import transform_sequence_by_esm2
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    """Load data from CSV files"""
    # Load sv sequences from human_virus_test.csv
    sv_data = pd.read_csv('data/inference_processed.csv')
    sv_sequences = sv_data['seq_b'].dropna().unique().tolist()

    # Load sh sequences from forinference.csv
    sh_data = pd.read_csv('data/human_virus_test.csv', low_memory=False)
    sh_sequences = sh_data['seq_a'].dropna().unique().tolist()

    # Load S.csv for task 2
    s_data = pd.read_csv('data/S.csv')
    return sv_sequences, sh_sequences, s_data


def encode_sequences_batch(sequences, batch_size=100):
    """批量编码序列并存储在 CPU 中"""
    encoded_sequences = []
    total_batches = (len(sequences) + batch_size - 1) // batch_size

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        batch_encoded = []

        # 逐个处理批次中的序列
        for seq in batch:
            try:
                # 编码并立即移到 CPU
                with torch.no_grad():  # 添加这行来禁用梯度计算
                    encoded = transform_sequence_by_esm2(seq).cpu()
                batch_encoded.append(encoded)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # 如果出现内存错误，清理内存并重试
                    torch.cuda.empty_cache()
                    gc.collect()
                    with torch.no_grad():  # 添加这行来禁用梯度计算
                        encoded = transform_sequence_by_esm2(seq).cpu()
                    batch_encoded.append(encoded)
                else:
                    raise e

            # 每个序列处理后都清理一次 GPU 内存
            torch.cuda.empty_cache()

        encoded_sequences.extend(batch_encoded)

        # 每个批次处理完后进行更彻底的清理
        gc.collect()
        torch.cuda.empty_cache()

    return encoded_sequences


def compute_cosine_similarity(seq1_tensor, seq2_tensor):
    """Compute cosine similarity between two sequence tensors"""
    return F.cosine_similarity(seq1_tensor.unsqueeze(0), seq2_tensor.unsqueeze(0)).item()


def task1_analysis(sh_sequences, sv_sequences):
    """Perform task 1 analysis: compute similarities and plot histogram"""
    max_similarities = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 用于保存所有结果的列表
    results = []

    # 分批编码 sh 序列
    print("\nEncoding SH sequences...")
    ENCODE_BATCH_SIZE = 50  # 减小编码批次大小以降低内存使用
    sh_encoded = encode_sequences_batch(sh_sequences, ENCODE_BATCH_SIZE)
    sv_encoded = encode_sequences_batch(sv_sequences, ENCODE_BATCH_SIZE)

    # 计算相似度的批处理大小
    COMPUTE_BATCH_SIZE = 1000

    # 计算相似度
    print("\nComputing similarities...")
    total_sh = len(sh_sequences)
    for idx, sh_seq in enumerate(sh_sequences, 1):
        try:
            # 获取 sh 序列的编码
            with torch.no_grad():
                sh_encoded_tensor = transform_sequence_by_esm2(sh_seq).to(device)

            # 确保 sh_encoded_tensor 是二维张量 [1, hidden_size]
            if sh_encoded_tensor.dim() == 3:  # [1, 1, hidden_size]
                sh_encoded_tensor = sh_encoded_tensor.squeeze(0)  # [1, hidden_size]
            elif sh_encoded_tensor.dim() == 1:  # [hidden_size]
                sh_encoded_tensor = sh_encoded_tensor.unsqueeze(0)  # [1, hidden_size]

            max_similarity = float('-inf')
            best_sv_idx = -1  # 记录最佳匹配的SV序列索引

            # 分批处理 sv_encoded
            for i in range(0, len(sv_encoded), COMPUTE_BATCH_SIZE):
                batch = sv_encoded[i:i + COMPUTE_BATCH_SIZE]
                batch_end = min(i + COMPUTE_BATCH_SIZE, len(sv_encoded))

                # 处理 batch_tensors 的维度
                batch_tensors = torch.stack(batch).to(device)  # [batch_size, hidden_size] 或 [batch_size, 1, hidden_size]
                if batch_tensors.dim() == 3:
                    batch_tensors = batch_tensors.squeeze(1)  # [batch_size, hidden_size]

                # 计算这个批次的所有相似度
                with torch.no_grad():
                    # sh_encoded_tensor: [1, hidden_size]
                    # batch_tensors: [batch_size, hidden_size]
                    batch_similarities = F.cosine_similarity(
                        sh_encoded_tensor,  # [1, hidden_size]
                        batch_tensors,  # [batch_size, hidden_size]
                        dim=1  # 在 hidden_size 维度上计算相似度
                    )

                # 找出这个批次中的最大相似度及其索引
                batch_max_val, batch_max_idx = batch_similarities.max(0)
                batch_max_val = batch_max_val.item()
                batch_max_idx = batch_max_idx.item()
                
                # 如果这个批次的最大相似度比全局最大相似度高，则更新
                if batch_max_val > max_similarity:
                    max_similarity = batch_max_val
                    best_sv_idx = i + batch_max_idx

                # 释放 GPU 内存
                del batch_tensors
                torch.cuda.empty_cache()

            # 保存这个 sh 序列的最大相似度
            max_similarities.append(max_similarity)
            
            # 保存结果：sh_seq, 最佳匹配的sv_seq, 相似度分数
            results.append([sh_seq, sv_sequences[best_sv_idx], max_similarity])

        except RuntimeError as e:
            if "out of memory" in str(e):
                # 如果出现内存错误，清理内存并重试
                print(f"\nWARNING: Out of memory error occurred. Cleaning up and retrying...")
                if 'sh_encoded_tensor' in locals():
                    del sh_encoded_tensor
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                print(f"\nError details: {str(e)}")
                if 'sh_encoded_tensor' in locals():
                    print(f"Tensor shapes - sh_encoded_tensor: {sh_encoded_tensor.shape}")
                if 'batch_tensors' in locals():
                    print(f"batch_tensors: {batch_tensors.shape}")
                raise e
        finally:
            # 释放 sh_encoded_tensor 的 GPU 内存
            if 'sh_encoded_tensor' in locals():
                del sh_encoded_tensor
            torch.cuda.empty_cache()

        # 显示进度
        if idx % 10 == 0:
            print(f"\rProcessed {idx}/{total_sh} sequences ({idx / total_sh * 100:.1f}%)", end="")
    
    # 保存结果到CSV文件
    print("\nSaving results to CSV file...")
    import pandas as pd
    results_df = pd.DataFrame(results, columns=['sh_seq', 'sv_seq', 'similarity_score'])
    results_df.to_csv('sh_sv_similarity_results.csv', index=False)
    print("Results saved to sh_sv_similarity_results.csv")

    print("\nPlotting results...")

    # Plot histogram using the same style as plot_similarity_histograms
    plt.figure(figsize=(6, 5))

    # Plot histogram with frequencies
    counts, bins, _ = plt.hist(max_similarities, bins=20, edgecolor='black', range=(0, 1))
    frequencies = counts / len(max_similarities)
    plt.clf()

    # Plot frequency histogram
    plt.bar(bins[:-1], frequencies, width=np.diff(bins), edgecolor='black', align='edge')
    plt.xlabel('Cosine Similarity Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add statistics as text
    stats = f'Mean: {np.mean(max_similarities):.3f}\n'
    stats += f'Std: {np.std(max_similarities):.3f}\n'
    stats += f'Min: {min(max_similarities):.3f}\n'
    stats += f'Max: {max(max_similarities):.3f}'
    plt.text(0.05, 0.95, stats, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('sv_sh_similarity_histogram.png', dpi=300)
    plt.close()

    return max_similarities


def main():
    # Load data
    print("Loading data...")
    sv_sequences, sh_sequences, s_data = load_data()

    # 根据sh 返回sv 最大的相似度的数据
    task1_analysis(sh_sequences, sv_sequences)


if __name__ == "__main__":
    main()
