import pandas as pd
from collections import defaultdict
import random


def load_data(file_path):
    """
    读取CSV文件并返回DataFrame
    """
    return pd.read_csv(file_path)


def create_sv_sh_dict(df):
    """
    创建一个字典，存储每个SV值对应的SH值列表
    """
    sv_sh_dict = defaultdict(list)
    for index, row in df.iterrows():
        sh_value = row["S_H"]
        sv_value = row["S_V"]
        sv_sh_dict[sv_value].append(sh_value)
    return sv_sh_dict


def calculate_sv_counts(sv_sh_dict):
    """
    计算每个SV值对应的不同SH值的数量
    """
    return {sv: len(set(sh_list)) for sv, sh_list in sv_sh_dict.items()}


def filter_sv(sv_counts, threshold):
    """
    筛选出符合条件的SV值，例如至少对应threshold个不同SH值的SV
    """
    return {sv: count for sv, count in sv_counts.items() if count >= threshold}


def adjust_sv_sh_counts(df, target_sh_count=15):
    """
    调整每个SV对应的SH值数量，使每个SV对应的SH值数量恰好为target_sh_count。
    为每个SH项添加一个标识字段，原始数据标记为1，随机添加的数据标记为0。
    """
    sv_sh_dict = create_sv_sh_dict(df)

    adjusted_sv_sh_dict = {}
    for sv, sh_list in sv_sh_dict.items():
        unique_sh_list = list(set(sh_list))  # 去重
        original_count = len(unique_sh_list)  # 记录原始数据的数量

        # 标记原始数据
        extended_sh_list = [(sh, 1) for sh in unique_sh_list]

        # 判断是否需要补全或截取
        if original_count < target_sh_count:
            # 需要补全
            while len(extended_sh_list) < target_sh_count:
                sh_to_add = random.choice(unique_sh_list)
                extended_sh_list.append((sh_to_add, 0))  # 添加随机数据并标记为0
        else:
            # 如果数量超过目标，截取并保留标记
            extended_sh_list = extended_sh_list[:target_sh_count]

        adjusted_sv_sh_dict[sv] = extended_sh_list

    # 将调整后的数据转换为DataFrame
    output_data = []
    for sv, sh_list in adjusted_sv_sh_dict.items():
        for sh, label in sh_list:
            output_data.append((sv, sh, label))

    adjusted_df = pd.DataFrame(output_data, columns=['S_V', 'S_H', 'Label'])
    return adjusted_df


def main(target_sh_count=3):
    # 读取CSV文件
    file_path = "./SH_SV.csv"
    df = load_data(file_path)

    # 创建SV-SH字典
    sv_sh_dict = create_sv_sh_dict(df)

    # 计算每个SV对应的不同SH值的数量
    sv_counts = calculate_sv_counts(sv_sh_dict)

    # 分析每个SV对应的SH数量的分布
    sv_counts_df = pd.DataFrame(list(sv_counts.items()), columns=['SV', 'SH_Count'])
    print(sv_counts_df.describe())

    # 调整每个SV对应的SH值数量
    adjusted_df = adjust_sv_sh_counts(df, target_sh_count=target_sh_count)

    # 将调整后的数据保存到新的CSV文件中
    adjusted_df.to_csv(f'adjusted_SH_SV_{target_sh_count}.csv', index=False)
    print("新的数据集已生成并保存到" + f'adjusted_SH_SV_{target_sh_count}.csv')


if __name__ == '__main__':
    main()
