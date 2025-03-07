import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import gc

from ESM2SequenceTransformer import transform_sequence_by_esm2

def load_data():
    """Load data from CSV files"""
    # Load sv sequences from human_virus_test.csv
    sv_data = pd.read_csv('data/human_virus_test.csv')
    sv_sequences = sv_data['seq_b'].dropna().unique().tolist()
    
    # Load sh sequences from forinference.csv
    sh_data = pd.read_csv('data/inference_processed.csv', low_memory=False)
    sh_sequences = sh_data['seq_a'].dropna().unique().tolist()
    
    # Load S.csv for task 2
    s_data = pd.read_csv('data/S.csv')
    return sv_sequences, sh_sequences, s_data


def encode_sequences_batch(sequences, batch_size=100):
    """批量编码序列并存储在 CPU 中"""
    encoded_sequences = []
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(sequences), batch_size), total=total_batches, desc="Encoding sequences"):
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

def task1_analysis(sv_sequences, sh_sequences):
    """Perform task 1 analysis: compute similarities and plot histogram"""
    max_similarities = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 分批编码 sh 序列
    print("\nEncoding SH sequences...")
    ENCODE_BATCH_SIZE = 50  # 减小编码批次大小以降低内存使用
    sh_encoded = encode_sequences_batch(sh_sequences, ENCODE_BATCH_SIZE)
    
    # 计算相似度的批处理大小
    COMPUTE_BATCH_SIZE = 1000
    
    # 计算相似度
    print("\nComputing similarities...")
    total_sv = len(sv_sequences)
    for idx, sv_seq in enumerate(sv_sequences, 1):
        try:
            # 获取 sv 序列的编码
            with torch.no_grad():
                sv_encoded = transform_sequence_by_esm2(sv_seq).to(device)
            
            # 确保 sv_encoded 是二维张量 [1, hidden_size]
            if sv_encoded.dim() == 3:  # [1, 1, hidden_size]
                sv_encoded = sv_encoded.squeeze(0)  # [1, hidden_size]
            elif sv_encoded.dim() == 1:  # [hidden_size]
                sv_encoded = sv_encoded.unsqueeze(0)  # [1, hidden_size]
            
            max_similarity = float('-inf')
            
            # 分批处理 sh_encoded
            for i in range(0, len(sh_encoded), COMPUTE_BATCH_SIZE):
                batch = sh_encoded[i:i + COMPUTE_BATCH_SIZE]
                
                # 处理 batch_tensors 的维度
                batch_tensors = torch.stack(batch).to(device)  # [batch_size, hidden_size] 或 [batch_size, 1, hidden_size]
                if batch_tensors.dim() == 3:
                    batch_tensors = batch_tensors.squeeze(1)  # [batch_size, hidden_size]
                
                # 计算这个批次的所有相似度
                with torch.no_grad():
                    # sv_encoded: [1, hidden_size]
                    # batch_tensors: [batch_size, hidden_size]
                    batch_similarities = F.cosine_similarity(
                        sv_encoded,  # [1, hidden_size]
                        batch_tensors,  # [batch_size, hidden_size]
                        dim=1  # 在 hidden_size 维度上计算相似度
                    )
                
                # 更新最大相似度
                max_similarity = max(max_similarity, batch_similarities.max().item())
                
                # 释放 GPU 内存
                del batch_tensors
                torch.cuda.empty_cache()
            
            # 保存这个 sv 序列的最大相似度
            max_similarities.append(max_similarity)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # 如果出现内存错误，清理内存并重试
                print(f"\nWARNING: Out of memory error occurred. Cleaning up and retrying...")
                del sv_encoded
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                print(f"\nError details: {str(e)}")
                print(f"Tensor shapes - sv_encoded: {sv_encoded.shape}")
                if 'batch_tensors' in locals():
                    print(f"batch_tensors: {batch_tensors.shape}")
                raise e
        finally:
            # 释放 sv_encoded 的 GPU 内存
            if 'sv_encoded' in locals():
                del sv_encoded
            torch.cuda.empty_cache()
        
        # 显示进度
        if idx % 10 == 0:
            print(f"\rProcessed {idx}/{total_sv} sequences ({idx/total_sv*100:.1f}%)", end="")
    
    print("\nPlotting results...")
    
    # Plot histogram using the same style as plot_similarity_histograms
    plt.figure(figsize=(6, 5))
    
    # Plot histogram with frequencies
    counts, bins, _ = plt.hist(max_similarities, bins=20, edgecolor='black', range=(0, 1))
    frequencies = counts / len(max_similarities)
    plt.clf()
    
    # Plot frequency histogram
    plt.bar(bins[:-1], frequencies, width=np.diff(bins), edgecolor='black', align='edge')
    plt.title('SV-SH Cosine Similarity Distribution')
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

def task2_analysis(s_data):
    """Perform task 2 analysis: find top 5 similar seq_a for each seq_b"""
    results = []
    # 过滤掉 NaN 值
    seq_b_list = s_data['seq_b'].dropna().unique()
    seq_a_list = s_data['seq_a'].dropna().unique()
    
    # seq_a_list = seq_a_list[:3000]  # 限制处理的序列数量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 分批编码 seq_a 序列
    print("\nEncoding seq_a sequences...")
    ENCODE_BATCH_SIZE = 50  # 减小编码批次大小以降低内存使用
    seq_a_encoded_list = encode_sequences_batch(seq_a_list, ENCODE_BATCH_SIZE)
    seq_a_encoded = list(zip(seq_a_list, seq_a_encoded_list))
    
    # 计算相似度的批处理大小
    COMPUTE_BATCH_SIZE = 1000
    
    print("\nFinding top 5 matches...")
    total_seqb = len(seq_b_list)
    for idx, seq_b in enumerate(seq_b_list, 1):
        try:
            # 获取 seq_b 的编码
            with torch.no_grad():  # 添加这行来禁用梯度计算
                seq_b_encoded = transform_sequence_by_esm2(seq_b).to(device)
            
            # 确保 seq_b_encoded 是二维张量 [1, hidden_size]
            if seq_b_encoded.dim() == 3:  # [1, 1, hidden_size]
                seq_b_encoded = seq_b_encoded.squeeze(0)  # [1, hidden_size]
            elif seq_b_encoded.dim() == 1:  # [hidden_size]
                seq_b_encoded = seq_b_encoded.unsqueeze(0)  # [1, hidden_size]
            
            all_similarities = []
            
            # 分批处理 seq_a_encoded
            for i in range(0, len(seq_a_encoded), COMPUTE_BATCH_SIZE):
                batch = seq_a_encoded[i:i + COMPUTE_BATCH_SIZE]
                batch_seqs = [item[0] for item in batch]
                
                # 处理 batch_tensors 的维度
                batch_tensors = torch.stack([item[1] for item in batch]).to(device)  # [batch_size, hidden_size] 或 [batch_size, 1, hidden_size]
                if batch_tensors.dim() == 3:
                    batch_tensors = batch_tensors.squeeze(1)  # [batch_size, hidden_size]
                
                # 计算这个批次的所有相似度
                with torch.no_grad():  # 添加这行来禁用梯度计算
                    # seq_b_encoded: [1, hidden_size]
                    # batch_tensors: [batch_size, hidden_size]
                    batch_similarities = F.cosine_similarity(
                        seq_b_encoded,  # [1, hidden_size]
                        batch_tensors,  # [batch_size, hidden_size]
                        dim=1  # 在 hidden_size 维度上计算相似度
                    )
                    # 转换为 numpy 数组前确保没有梯度
                    batch_similarities = batch_similarities.cpu()
                
                # 保存这个批次的结果
                all_similarities.extend(list(zip(batch_seqs, batch_similarities.detach().cpu().numpy())))
                
                # 释放 GPU 内存
                del batch_tensors
                torch.cuda.empty_cache()
            
            # 获取 top 5 结果
            top_5 = sorted(all_similarities, key=lambda x: x[1], reverse=True)[:5]
            results.append({
                'seq_b': seq_b,
                'top_5_matches': top_5
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # 如果出现内存错误，清理内存并重试
                print(f"\nWARNING: Out of memory error occurred. Cleaning up and retrying...")
                del seq_b_encoded
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                print(f"\nError details: {str(e)}")
                print(f"Tensor shapes - seq_b_encoded: {seq_b_encoded.shape}")
                if 'batch_tensors' in locals():
                    print(f"batch_tensors: {batch_tensors.shape}")
                raise e
        finally:
            # 释放 seq_b_encoded 的 GPU 内存
            if 'seq_b_encoded' in locals():
                del seq_b_encoded
            torch.cuda.empty_cache()
        
        # 显示进度
        if idx % 10 == 0:
            print(f"\rProcessed {idx}/{total_seqb} sequences ({idx/total_seqb*100:.1f}%)", end="")
    
    print("\nSaving results...")
    # Save results
    save_task2_results(results)
    return results

def save_task2_results(results):
    """Save task 2 results to a CSV file"""
    rows = []
    for result in results:
        seq_b = result['seq_b']
        row = {'seq_b': seq_b}
        
        # 获取 top 5 匹配结果
        top_5_matches = result['top_5_matches']
        
        # 确保每个样本都有 5 个结果
        for i in range(5):
            if i < len(top_5_matches):
                seq_a, similarity = top_5_matches[i]
            else:
                # 如果不足 5 个，用空值填充
                seq_a, similarity = '<null>', '<null>'
            
            row[f'top_{i+1}_seq_a'] = seq_a
            row[f'top_{i+1}_similarity'] = similarity
        
        rows.append(row)
    
    # 创建 DataFrame 并按列排序
    df = pd.DataFrame(rows)
    columns = ['seq_b'] + [f'top_{i}_seq_a' for i in range(1, 6)] + [f'top_{i}_similarity' for i in range(1, 6)]
    df = df[columns]
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'task2_results_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # 打印前几行结果用于验证
    print("\nFirst few results:")
    print(df.head().to_string())

def main():
    # Load data
    print("Loading data...")
    sv_sequences, sh_sequences, s_data = load_data()
    # sv_sequences = sv_sequences[:100]
    
    # Task 1
    print("\nPerforming Task 1: SV-SH similarity analysis...")
    task1_analysis(sv_sequences, sh_sequences)
    print("Task 1 completed. Histogram saved as 'sv_sh_similarity_histogram.png'")
    
    # Task 2
    print("\nPerforming Task 2: Finding top 5 similar sequences...")
    task2_analysis(s_data)
    print("Task 2 completed. Results saved to CSV file.")

if __name__ == "__main__":
    main()
