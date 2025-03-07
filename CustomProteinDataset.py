import ast
import hashlib
import random
from collections import defaultdict

import pandas as pd
import torch
from sklearn.model_selection import KFold
from subword_nmt.apply_bpe import BPE
from torch.utils.data import Dataset, DataLoader
import os, sys, json, codecs

from transformers import BertTokenizer

from LucaProt.src.predict_many_samples import transform_sample_2_feature
from LucaProt.src.protein_structure.predict_structure import predict_embedding


def load_and_adjust_data(file_path, target_sh_count=15):
    adjust_data_path = os.path.join("".join(file_path.split('/')[:-1]), f"adjusted_data_{target_sh_count}.csv")
    if os.path.exists(adjust_data_path):
        return pd.read_csv(adjust_data_path)
    df = pd.read_csv(file_path)
    sv_sh_dict = defaultdict(list)

    for index, row in df.iterrows():
        sh_value = row["S_H"]
        sv_value = row["S_V"]
        sv_sh_dict[sv_value].append(sh_value)

    adjusted_sv_sh_dict = {}
    for sv, sh_list in sv_sh_dict.items():
        unique_sh_list = list(set(sh_list))
        extend_sh_list = [(sh, 1) for sh in unique_sh_list]
        if len(extend_sh_list) < target_sh_count:
            while len(extend_sh_list) < target_sh_count:
                extend_sh_list.append((random.choice(unique_sh_list), 0))
        else:
            extend_sh_list = extend_sh_list[:target_sh_count]
        adjusted_sv_sh_dict[sv] = extend_sh_list

    output_data = [(sv, sh_list) for sv, sh_list in adjusted_sv_sh_dict.items()]
    # 增加一个标签字段来表示新增加的值是随机添加的还是真实的
    adjusted_df = pd.DataFrame(output_data, columns=['S_V', 'S_H_List'])
    adjusted_df['S_H_types'] = adjusted_df['S_H_List'].apply(
        lambda x: [label for _, label in x])  # 0 indicate random sample
    adjusted_df['S_H_List'] = adjusted_df['S_H_List'].apply(lambda x: [sh for sh, _ in x])
    adjusted_df.to_csv(adjust_data_path, index=False)
    return adjusted_df


class CustomProteinDataset(Dataset):
    def __init__(self, data=None, args=None, seq_tokenizer=None, subword=None, struct_tokenizer=None):
        self.data = self.pre_process_data(data)
        self.args = args
        self.seq_tokenizer = seq_tokenizer
        self.subword = subword
        self.struct_tokenizer = struct_tokenizer

    def pre_process_data(self, data: pd.DataFrame):
        if 'S_H_List' not in data.columns or 'S_V' not in data.columns:
            raise ValueError("Data must include 'S_H_List' and 'S_V' columns.")

        def generate_hash_id(value):
            return hashlib.sha256(value.encode()).hexdigest()

        # 创建DataFrame的副本
        data_copy = data.copy()
        # 使用.loc显式地进行赋值
        # data_copy.loc[:, 'S_H_List'] = data_copy['S_H_List'].apply(ast.literal_eval)

        # 创建一个新的DataFrame来保存处理后的数据
        processed_data = pd.DataFrame({
            'S_H_ids': data_copy['S_H_List'].apply(lambda sh_list: [generate_hash_id(sh) for sh in sh_list]),
            'S_V_id': data_copy['S_V'].apply(generate_hash_id),
            'S_V': data_copy['S_V'].values,
            'S_H_List': data_copy['S_H_List'].apply(ast.literal_eval),
            'S_H_types': data_copy['S_H_types'].apply(ast.literal_eval)
        })

        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sv_row = [row["S_V_id"], row["S_V"]]
        sh_rows = [(sh_id, sh) for sh_id, sh in zip(row["S_H_ids"], row["S_H_List"])]
        sh_batch_input_list = []
        for sh_id, sh in sh_rows:
            sh_row = [sh_id, sh]
            sh_batch_info, sh_batch_input = transform_sample_2_feature(self.args, sh_row, self.seq_tokenizer,
                                                                       self.subword, self.struct_tokenizer)
            # Remove batch dimension if present
            for key in sh_batch_input:
                sh_batch_input[key] = sh_batch_input[key].squeeze(0)
            sh_batch_input_list.append(sh_batch_input)
        sv_batch_info, sv_batch_input = transform_sample_2_feature(self.args, sv_row, self.seq_tokenizer, self.subword,
                                                                   self.struct_tokenizer)
        # Remove batch dimension if present
        # for key in sv_batch_input:
        #     sv_batch_input[key] = sv_batch_input[key].squeeze(0)

        # Stack the sh_batch_input_list
        stacked_sh_batch_input_list = {key: torch.stack([sh_batch_input[key] for sh_batch_input in sh_batch_input_list])
                                       for key in sh_batch_input_list[0].keys()}

        sh_types_id = torch.tensor([int(x) for x in row["S_H_types"]], device=self.args.device, dtype=torch.int)
        stacked_sh_batch_input_list["sh_types_id"] = sh_types_id

        return {"input": sv_batch_input, "labels": stacked_sh_batch_input_list}

    def collate_fn(self, batch):
        # 将batch中的输入和标签分开
        inputs = [item['input'] for item in batch]
        labels = [item['labels'] for item in batch]

        return {"inputs": inputs[0], "labels": labels[0]}


def get_dataloaders_for_fold(csv_file, fold_index, batch_size=1, num_splits=10, target_sh_count=3):
    args, row, seq_tokenizer, subword, struct_tokenizer = prepare_param()
    data = load_and_adjust_data(csv_file, target_sh_count)
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    for i, (train_val_index, test_index) in enumerate(kf.split(data)):
        if i == fold_index:
            train_val_data = data.iloc[train_val_index]
            test_data = data.iloc[test_index]

            train_size = int(0.9 * len(train_val_data))
            train_data = train_val_data[:train_size]
            valid_data = train_val_data[train_size:]

            train_dataset = CustomProteinDataset(train_data, args=args, seq_tokenizer=seq_tokenizer, subword=subword,
                                                 struct_tokenizer=struct_tokenizer)
            valid_dataset = CustomProteinDataset(valid_data, args=args, seq_tokenizer=seq_tokenizer, subword=subword,
                                                 struct_tokenizer=struct_tokenizer)
            test_dataset = CustomProteinDataset(test_data, args=args, seq_tokenizer=seq_tokenizer, subword=subword,
                                                struct_tokenizer=struct_tokenizer)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          collate_fn=train_dataset.collate_fn)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=valid_dataset.collate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=test_dataset.collate_fn)

            return train_dataloader, valid_dataloader, test_dataloader


def prepare_param():
    from argparse import Namespace
    import torch

    # 创建 args 参数
    args = Namespace(
        fasta_file='LucaProt/data/rdrp/test/test.fasta',
        save_file='LucaProt/result/rdrp/test/test_result.csv',
        truncation_seq_length=4096,
        emb_dir='LucaProt/emb/',
        pdb_dir='protein',
        chain=None,
        dataset_name='rdrp_40_extend',
        dataset_type='protein',
        task_type='binary_class',
        model_type='sefn',
        time_str='20230201140320',
        step='100000',
        threshold=0.5,
        print_per_number=10,
        has_seq_encoder=True,
        has_struct_encoder=False,
        has_embedding_encoder=True,
        subword=True,
        codes_file='LucaProt/subword/rdrp_40_extend/protein/binary_class/protein_codes_rdrp_20000.txt',
        input_mode='single',
        label_filepath='LucaProt/dataset/rdrp_40_extend/protein/binary_class/label.txt',
        output_dir='LucaProt/models/rdrp_40_extend/protein/binary_class/sefn/20230201140320',
        config_path='LucaProt/config/rdrp_40_extend/protein/binary_class/sefn_config.json',
        do_lower_case=False,
        sigmoid=True,
        loss_type='bce',
        output_mode='binary_class',
        device=torch.device('cuda'),
        seq_vocab_path='LucaProt/vocab/rdrp_40_extend/protein/binary_class/subword_vocab_20000.txt',
        seq_pooling_type='value_attention',
        seq_max_length=512,
        struct_vocab_path=None,
        struct_max_length=2048,
        struct_pooling_type=None,
        trunc_type='right',
        no_position_embeddings=False,
        no_token_type_embeddings=True,
        cmap_type=10.0,
        embedding_input_size=2560,
        embedding_pooling_type='value_attention',
        embedding_max_length=2048,
        embedding_type='matrix'
    )

    # 创建其他参数
    row = ['>bunya-arena-like_Rayfin_XTXMS70955_Wenling_red_spikefish_hantavirus_RdRp_6305bp',
           'PIYKMKAKQAWSEISQLYERRHEVVANKIRVHLLAAYEGEVTLRRVLLDNGLDITLFENVRWNSGIHEERGKFFLEKTPDISVVTMVEGRLRLDVVEVSVSTKVMESRAIKDNKYRKGLELIGETFDIDMNYLVVSSFTDGRNLDTEYPGIDWSVSVEEIRLINERVRLIQTEEIEENEKAMLTAMMKGYFSKGKSRRDTRPSRVIAPQRSDQPEFENDEEVLFELKDFAANDLERTSNFQRVSVAGIKETIENASLIASSSFVEKEHSGVYAYMPFGRAKAERPMTMEYQKTCMLEAEREFSYINERDEFLEAMMVTLEKGVAVGAENWDLWFRKTMDGREPMTVTGVEGRLSTRAKELLQHWTGKDRTSQNFEPKLMESLPFWNDSINDNCYIDVLKIKKRLEMDLEEEFSTEKIDFGNSMAEASLSNVFHAEMKRRVIDPVARSRAFQSSIVVRDVCEWLVAQSGNKRSKRWSVFACCDGECVIIKIPGKSTESLGGKINYMAMCTESAYLGPQTNVMRRYVGKGKTWLLLKPMSLDLRRLESMGSTLEKGLLLCGSIATKHSEATGTLPSPYDLREIFSIHYLVSTTPKNRICSIFDYLRYAINSCVADVSGYGELLKDEFSKACGTSLEVFLRREAASLLEDLAENKDDTLVKKMTLGQHSIKTKFGATGKYRSFSSNLYYKSFSSLHMEIYGLFFTCPKGLHGKIEDLKIQEETVEWQAKWDALAKQLKCEMQEGYTIGDREPRQQTFCRDFMYECGRWIDKQVSYKWDDVQASIQRGGLRENYYANPRNRSTKGMTVFSSDYSKMESTTTIQQALRELAGGTTVGTVEEEAMKSVGKTPNVRLVRKYQRTTSDRGIFVADRDTRAKLQVIERIAGAIAKNVESELISVPGDVKMNIIQDMLTKAIRWSAGESVLRTEFGDMKMKRRVMFCSADATKWSPGDNAYKFIPFVEGITSLTDAEKNLLTACLLGISKSNLGLSDGAFEMLSKMDGTKNPKVDEMKNFFGLPYRRTGKVEGNWLQGNLNFISSLVGVAALNKGGGYAKKLWPELDCFVEVLGHSDDSLILIGWVSPASEDIGQYLLWVDKMAELSEEYKCLKRLNHWECIFRVIERTALMASIKLSTKKTFLSKTMSEFVGYNFEAGNPTTPWIKPAMGALGELKVKGYAEDRASVMSSAVKVLDLSGSLQMAQLVAYIGNGRVLRGYGMQKGMVNHPGALLKLRDVDIPSFLGGGPIPSVLSLATGGTNLQDIMTVKAHVTRYRTHRDTYSERVLKVYKTCEKLFKEEENDQHIFGRVKWRIFFPKSDPYELGFLTRESLKAWEAAHPEFFFLNPTDPKDILFQTWREFRKPEMQASLVRQSETVLRMRLMGRVAGDVVWVNGEWASVRTLLFTVSTQCESELITESDLIRWEQIEENLFSKSIVWTDFLNQTFAEVSKGVKRQVAKLPRRLVVREDDIPLINQKRDILAWGVSSQTTKRLIETQCTDPNMIEVDAAKLRAAAADKLQLDITLVDGAKRCDFLTKGSTVSRTVIVGSNVEPTATGIVIGWLRESSFTRVVSASSSGYISKAKTIFSENTEGQHWLNAKKLIITIWKMAKANSAEPGVWLRSLSFQGATLWKWMQTIVKQTPDSSRVDAALAVALNDTLGDDSWLNSVASQKMLSGKRYVKEQHYDPVLKVWQGQLVVEFLYGSEIGELYFDGDSIVKLSTSIRDPIPLTHMMNTVRKELAGSKFNMPVVRSDGTEGIRVIKRKDFYRWDRVREGDWILPFVFIDPTMSVGSAPTRSYQFTIRDGGFSVWAKDRENSRGVKIASAMSFLSDIPLSALEAADELFHQNVAIHELTKRGFLPNLILGTTARITRSEAAVILVRKTLPKRMAVISEVLDLVGGGKVSFHGIEFTKTSISSWETAEDESDDDYVVDLDDMDFDLELEEPESSFGTVEVNAEFYIEEDRAIENEDEIPRGVTIYEDLESTIRHWVEKDVGDVSNVDGIQSFLFMKWLAKSFDFGRQVDLAMYWDLLSVDTILGPMANTIDLIGMDVLKSKLKEAPEMRELEPSELKPYLGFNYSRAMRLMANLFPRKRVDFYD']
    seq_tokenizer = BertTokenizer.from_pretrained(
        'LucaProt/models/rdrp_40_extend/protein/binary_class/sefn/20230201140320/checkpoint-100000/sequence')
    # 创建 subword 参数
    bpe_codes_prot = codecs.open(args.codes_file, encoding='utf-8')
    subword = BPE(bpe_codes_prot, merges=-1, separator='')
    struct_tokenizer = None

    return args, row, seq_tokenizer, subword, struct_tokenizer

    # 调用 transform_sample_2_feature
    # batch_info, batch_input = transform_sample_2_feature(
    #     args=args,
    #     row=row,
    #     seq_tokenizer=seq_tokenizer,
    #     subword=subword,
    #     struct_tokenizer=struct_tokenizer,
    #     pad_on_left=False,
    #     pad_token=0,
    #     pad_token_segment_id=0,
    #     mask_padding_with_zero=True
    # )


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = get_dataloaders_for_fold('data/SH_SV.csv', fold_index=0)
    for batch in train_loader:
        print(type(batch))
        # break
