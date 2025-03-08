import torch

from CustomProteinDataset import get_dataloaders_for_fold
from LucaProt.src.SSFN.model import SequenceAndStructureFusionNetwork
from PipeLine import FusionLayer, Generator, PipeLine


def directed_hausdorff_distance(set1, set2):
    n = set1.size(0)
    m = set2.size(0)
    d_matrix = torch.cdist(set1, set2, p=2)  # 计算两个集合之间的距离矩阵
    distances, _ = torch.min(d_matrix, dim=1)  # 找到set1中每个点到set2的最近距离
    return distances.mean()  # 返回平均距离


def set_loss(set1, set2):
    loss_12 = directed_hausdorff_distance(set1, set2)
    loss_21 = directed_hausdorff_distance(set2, set1)
    return (loss_12 + loss_21) / 2  # 返回平均的哈斯多夫距离


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_model_dir_and_args():
    model_dir = 'LucaProt/models/rdrp_40_extend/protein/binary_class/sefn/20230201140320/checkpoint-100000'
    args = {
        'fasta_file': 'LucaProt/data/rdrp/test/test.fasta',
        'save_file': 'LucaProt/result/rdrp/test/test_result.csv',
        'truncation_seq_length': 4096,
        'emb_dir': 'LucaProt/emb/',
        'pdb_dir': 'protein',
        'chain': None,
        'dataset_name': 'rdrp_40_extend',
        'dataset_type': 'protein',
        'task_type': 'binary_class',
        'model_type': 'sefn',
        'time_str': '20230201140320',
        'step': '100000',
        'threshold': 0.5,
        'print_per_number': 10,
        'has_seq_encoder': True,
        'has_struct_encoder': False,
        'has_embedding_encoder': True,
        'subword': True,
        'codes_file': 'LucaProt/subword/rdrp_40_extend/protein/binary_class/protein_codes_rdrp_20000.txt',
        'input_mode': 'single',
        'label_filepath': 'LucaProt/dataset/rdrp_40_extend/protein/binary_class/label.txt',
        'output_dir': 'LucaProt/models/rdrp_40_extend/protein/binary_class/sefn/20230201140320',
        'config_path': 'LucaProt/config/rdrp_40_extend/protein/binary_class/sefn_config.json',
        'do_lower_case': False,
        'sigmoid': True,
        'loss_type': 'bce',
        'output_mode': 'binary_class',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'seq_vocab_path': 'LucaProt/vocab/rdrp_40_extend/protein/binary_class/subword_vocab_20000.txt',
        'seq_pooling_type': 'value_attention',
        'seq_max_length': 2048,
        'struct_vocab_path': None,
        'struct_max_length': 2048,
        'struct_pooling_type': None,
        'trunc_type': 'right',
        'no_position_embeddings': False,
        'no_token_type_embeddings': True,
        'cmap_type': 10.0,
        'embedding_input_size': 2560,
        'embedding_pooling_type': 'value_attention',
        'embedding_max_length': 2048,
        'embedding_type': 'matrix'
    }
    return model_dir, Namespace(**args)


def load_model(model_class, model_dir, args):
    model = model_class.from_pretrained(model_dir, args=args)
    model.to(args.device)
    return model


def train(model: PipeLine, data_loader, omics: dict):
    model.train()
    for batch in data_loader:
        generated_outputs = model(batch, omics)
        print(generated_outputs)
        print(torch.cuda.memory_allocated())


if __name__ == '__main__':
    model_dir, args = get_model_dir_and_args()
    luca_prot = load_model(SequenceAndStructureFusionNetwork, model_dir, args)
    train_loader, valid_loader, test_loader = get_dataloaders_for_fold('data/SH_SV.csv', fold_index=0)

    omics = {'sv': 0, 'sh1': 1, 'sh2': 2, 'sh3': 3}
    fusion = FusionLayer(modal_num=4, modal_dim=[256] * 4, latent_dim=64, hidden_dim=[256, 128, 128],
                         pretrain_model_path=None,
                         task="omics_classification", omics_data=True, fixed=False,
                         omics=omics
                         )
    generator = Generator()  # Output dim can be the vocabulary size or another suitable parameter

    # Assemble into a pipeline
    pipeline = PipeLine(luca_prot=luca_prot, fusion=fusion, generator=generator)
    model = pipeline.to(args.device)
    train(model, train_loader, omics)
