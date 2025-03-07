import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import BartForConditionalGeneration

from CustomProteinDataset import prepare_param
from LucaProt.src.SSFN.model import SequenceAndStructureFusionNetwork
from TMONet.model.TMO_Net_model import TMO_Net, dfs_freeze, un_dfs_freeze, product_of_experts, reparameterize

args, row, seq_tokenizer, subword, struct_tokenizer = prepare_param()


class FusionLayer(nn.Module):
    def __init__(self, modal_num, modal_dim, latent_dim, hidden_dim, pretrain_model_path, task, omics_data, fixed,
                 omics):
        super(FusionLayer, self).__init__()
        self.k = modal_num
        #   cross encoders
        self.cross_encoders = TMO_Net(modal_num, modal_dim, latent_dim, hidden_dim, omics_data)
        if pretrain_model_path:
            print('load pretrain model')
            model_pretrain_dict = torch.load(pretrain_model_path, map_location='cpu')
            self.cross_encoders.load_state_dict(model_pretrain_dict)

        omics_values = set(omics.values())
        for i in range(self.k):
            if i not in omics_values:
                print('fix cross-modal encoders')
                dfs_freeze(self.cross_encoders.encoders[:][i])

        if fixed:
            dfs_freeze(self.cross_encoders)

    def un_dfs_freeze_encoder(self):
        un_dfs_freeze(self.cross_encoders)

    #   return embedding
    def get_embedding(self, input_x, omics, batch_size=1):
        output, share_representation = self.cross_encoders(input_x, batch_size)
        embedding_tensor = []
        keys = list(omics.keys())
        share_features = [share_representation[omics[key]] for key in keys]
        share_features = sum(share_features) / len(keys)
        for i in range(self.k):
            mu_set = []
            log_var_set = []
            for j in range(len(omics)):
                latent_z, mu, log_var = output[omics[keys[j]]][i]
                mu_set.append(mu)
                log_var_set.append(log_var)
            poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
            poe_latent_z = reparameterize(poe_mu, poe_log_var)

            embedding_tensor.append(poe_mu)
        embedding_tensor = torch.cat(embedding_tensor, dim=1)
        # multi_representation = torch.concat((embedding_tensor, share_features), dim=1)
        return embedding_tensor

    # omics dict, 标注输入的数据
    def forward(self, input_x, omics, batch_size=1):
        multi_representation = self.cross_encoders.get_embedding(input_x, batch_size, omics)

        return multi_representation


class SequenceGenerator(nn.Module):
    def __init__(self, input_dim, omics_types, seq_size, seq_len, vocab_dim):
        super(SequenceGenerator, self).__init__()
        self.input_dim = input_dim
        self.omics_types = omics_types
        self.seq_size = seq_size
        self.seq_len = seq_len
        self.vocab_dim = vocab_dim

        # 嵌入层，将omics数据映射到潜在空间
        self.omics_embeddings = nn.ModuleDict({
            key: nn.Linear(1, input_dim) for key in omics_types
        })

        # GRU生成器
        self.gru = nn.GRU(input_dim, seq_size, batch_first=True)

        # 输出层
        self.output_layer = nn.Linear(seq_size, vocab_dim)

    def forward(self, input_x, omics):
        # 融合潜在向量与omics数据
        omics_input = torch.cat([
            self.omics_embeddings[key](omics[key].view(1, -1).float()) for key in self.omics_types
        ], dim=0).sum(dim=0).unsqueeze(0)  # 求和融合所有omics数据

        # 将融合后的向量与输入向量结合
        combined_input = input_x + omics_input

        # 生成序列
        gru_output, _ = self.gru(combined_input.unsqueeze(0))
        output = self.output_layer(gru_output.reshape(-1, self.seq_size))

        # 重新排列输出以匹配目标输出尺寸
        return output.view(-1, self.seq_len, self.vocab_model)


class Generator(nn.Module):
    def __init__(self, input_dim=256, decode_dim=768, seq_tokenizer=seq_tokenizer, num_decoders=7):
        super(Generator, self).__init__()
        self.tokenizer = seq_tokenizer
        self.input_dim = input_dim
        self.decode_dim = decode_dim
        self.num_decoders = num_decoders

        # Create a list of Transformer decoders
        self.decoders = nn.ModuleList([
            BartForConditionalGeneration.from_pretrained("./bart-base") for _ in range(num_decoders)
        ])

        self.encoder_to_decoder_linear = nn.ModuleList([
            nn.Linear(input_dim, self.decode_dim) for _ in range(num_decoders)
        ])

    def forward(self, input_x, omics, data, labels, seq_types_id=None):
        """
        :param input_x: torch.Size([1, 256])
        :param omics: {'sv': 0, 'sh1': 1, 'sh2': 2, 'sh3': 3}
        :param labels:[labels_size, seq_len]
        :param seq_types_id: [labels_size] indicated whether the labels are padding label
        """
        # Dictionary to store generated sequences for each omics
        generated_outputs = {}

        omics = {key: omics[key] - 1 for key in omics if key != 'sv'}

        x_ids = data['input_ids']
        x_seq_len = x_ids.shape[1]
        x_attention_mask = data['attention_mask']
        labels_ids = labels['input_ids']
        labels_attention_mask = labels['attention_mask']
        seq_size = labels_ids.shape[0]
        input_x = input_x.repeat(seq_size, x_seq_len, 1)

        # 准备 decoder 的输入：对 labels 进行右移
        decoder_input_ids = self._shift_right(labels_ids)
        decoder_attention_mask = torch.cat(
            [torch.full_like(labels_attention_mask[:, :1], 1), labels_attention_mask[:, :-1]], dim=-1)

        for i, (omic_key, decoder_index) in enumerate(omics.items()):
            # 将encoder 的输出维度映射到 decoder 的维度
            encoder_outputs = self.encoder_to_decoder_linear[decoder_index](input_x[i].unsqueeze(0))

            # 使用 encoder 的输出和处理后的 decoder 的输入和注意力掩码来进行解码
            outputs = self.decoders[decoder_index](
                attention_mask=x_attention_mask,  # 注意这里使用的是 labels_attention_mask
                encoder_outputs=(encoder_outputs,),  # 注意 encoder_outputs 要作为元组传入
                decoder_input_ids=decoder_input_ids[i].unsqueeze(0),
                decoder_attention_mask=decoder_attention_mask[i].unsqueeze(0),  # 为解码器提供正确的注意力掩码
                labels=labels_ids[i].unsqueeze(0)  # labels 用于计算损失
            )

            # Store the generated sequences for the current omic type
            generated_outputs[omic_key] = outputs

        return generated_outputs

    def _shift_right(self, input_ids):
        # 0   4 3 eos  :labels
        # bos 0 4 3    :shift
        # [[0.43,0.34,0.0543],[0.43,0.34,0.0543],[0.43,0.34,0.0543],[0.43,0.34,0.0543]] pred : [0,0,0,0] [seq_len * vocab_size]
        # [1,2,5,9] label
        """Shift input ids one token to the right for decoder input."""
        start_token_id = self.decoders[0].config.decoder_start_token_id
        return torch.cat([torch.full_like(input_ids[:, :1], start_token_id), input_ids[:, :-1]], dim=-1)

    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf')."""
        mask = torch.triu(torch.ones(1, sz) * float('-inf'), diagonal=1)
        return mask


class PipeLine(nn.Module):
    def __init__(self, luca_prot: SequenceAndStructureFusionNetwork, fusion: FusionLayer, generator: Generator):
        super(PipeLine, self).__init__()
        self.luca_port = luca_prot
        self.fusion = fusion
        self.generator = generator

    def forward(self, batch, omics):
        data = batch['inputs']
        labels = batch['labels']
        input = {}
        for key in data.keys():
            input[key] = torch.cat([data[key], labels[key]], dim=0)
        pooled_emd = self.luca_port(**input)
        fusion_emd = self.fusion(pooled_emd, omics)
        generated_outputs = self.generator(fusion_emd, omics, data, labels)
        return generated_outputs

    def predict(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    # 生成 pipeline 的 虚拟输入输出
    # Create instances of the required modules
    from train import get_model_dir_and_args, load_model

    model_dir, args = get_model_dir_and_args()
    luca_prot = load_model(SequenceAndStructureFusionNetwork, model_dir, args)
    omics = {'sv': 0, 'sh1': 1, 'sh2': 2, 'sh3': 3}
    fusion = FusionLayer(modal_num=4, modal_dim=[256] * 4, latent_dim=64, hidden_dim=[256, 128, 128],
                         pretrain_model_path=None,
                         task="omics_classification", omics_data=True, fixed=False,
                         omics=omics
                         )
    generator = Generator()  # Output dim can be the vocabulary size or another suitable parameter

    # Assemble into a pipeline
    pipeline = PipeLine(luca_prot=luca_prot, fusion=fusion, generator=generator)
    pipeline = pipeline.to(args.device)
    inputs = {
        'input_ids': torch.randint(0, 1000, (1, 512), device=args.device),  # Random integers assuming token IDs
        'attention_mask': torch.randint(0, 2, (1, 512), device=args.device),  # Random binary mask
        'token_type_ids': torch.randint(0, 2, (1, 512), device=args.device),
        # Random binary values assuming token type IDs
        'embedding_info': torch.randn(1, 512, 2560, device=args.device),  # Random floating-point numbers
        'embedding_attention_mask': torch.randint(0, 2, (1, 512), device=args.device)  # Random binary mask
    }
    labels = {
        'input_ids': torch.randint(0, 1000, (3, 512), device=args.device),  # Random integers assuming token IDs
        'attention_mask': torch.randint(0, 2, (3, 512), device=args.device),  # Random binary mask
        'token_type_ids': torch.randint(0, 2, (3, 512), device=args.device),
        # Random binary values assuming token type IDs
        'embedding_info': torch.randn(3, 512, 2560, device=args.device),  # Random floating-point numbers
        'embedding_attention_mask': torch.randint(0, 2, (3, 512), device=args.device)  # Random binary mask
    }
    batch = {"inputs": inputs, "labels": labels}

    generated_sequences = pipeline(batch, omics)
    print(generated_sequences)
