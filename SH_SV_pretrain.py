# SH_SV_pretrain.py
import hashlib
import logging
import os
import sys
import typing

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from CustomProteinDataset import prepare_param
from LucaProt.src.SSFN.model import SequenceAndStructureFusionNetwork
from LucaProt.src.predict_many_samples import transform_sample_2_feature
from SH_SV_dataset import SH_SV_Dataset
import time
from tqdm import tqdm

from TMONet.model.TMO_Net_model import TMO_Net, dfs_freeze, un_dfs_freeze, product_of_experts, reparameterize
from SH_SV_dataset import get_dataloaders_for_fold
from train import get_model_dir_and_args, load_model

# luca prot parameter init
args, row, seq_tokenizer, subword, struct_tokenizer = prepare_param()

# Get current file name
current_file_name = os.path.basename(__file__)

# Construct log file path
log_file = f'logs/{current_file_name.replace(".py", ".log")}'

# 如果目录不存在，则创建目录
if not os.path.exists(os.path.dirname(log_file)):
    os.makedirs(os.path.dirname(log_file))

# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler(log_file, mode='a')
console_handler = logging.StreamHandler(sys.__stdout__)  # Use original stdout

# Set levels for handlers
file_handler.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)
# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Ensure logs are flushed immediately after writing
file_handler.flush = lambda: file_handler.stream.flush()
console_handler.flush = lambda: console_handler.stream.flush()


# Redirect print function
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n':
            self.level(message.strip())
            # Ensure immediate flush after writing
            logger.handlers[0].flush()  # Flush the file handler
            logger.handlers[1].flush()  # Flush the console handler

    def flush(self):
        pass


sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)


def transform_sequence_and_structure_by_luca_prot(seq: typing.Optional[typing.Union[str, typing.List[str]]],
                                                  luca_prot_model: SequenceAndStructureFusionNetwork) -> torch.Tensor:
    def generate_hash_id(value):
        return hashlib.sha256(value.encode()).hexdigest()

    def process_single_sequence(s):
        prot_id = generate_hash_id(s)
        seq_row = [prot_id, s]
        sh_batch_info, sh_batch_input = transform_sample_2_feature(args, seq_row, seq_tokenizer,
                                                                   subword, struct_tokenizer)
        embedding = luca_prot_model(**sh_batch_input)
        return embedding

    if isinstance(seq, str):
        return process_single_sequence(seq)
    elif isinstance(seq, list):
        embeddings = [process_single_sequence(s) for s in seq]
        return torch.cat(embeddings, dim=0)  # Concatenate along the first dimension


def load_lucaprot() -> SequenceAndStructureFusionNetwork:
    model_dir, args = get_model_dir_and_args()
    luca_prot = load_model(SequenceAndStructureFusionNetwork, model_dir, args)
    return luca_prot


def val_pretrain(test_dataloader, model: TMO_Net, luca_prot_model: SequenceAndStructureFusionNetwork, epoch,
                 omics: typing.Dict, fold):
    model.eval()
    print(f'-----start epoch {epoch} validation-----')
    total_loss = 0
    total_self_elbo = 0
    total_cross_elbo = 0
    total_cross_infer_loss = 0
    total_dsc_loss = 0
    total_cross_infer_dsc_loss = 0
    Loss = []
    all_embeddings = torch.Tensor([]).cuda()
    all_labels = torch.Tensor([]).cuda()

    with torch.no_grad():
        with tqdm(test_dataloader, unit='batch') as tepoch:
            for batch, data in enumerate(tepoch):
                tepoch.set_description(f" Epoch {epoch}: ")

                SH = data['S_H']  # Assuming these need to be preprocessed into tensors
                SV = data['S_V']  # Assuming these need to be preprocessed into tensors
                feature_SH = data['feature_SH'].cuda()
                feature_SV = data['feature_SV'].cuda()

                # Assuming a method to encode SH and SV into tensors
                # SH_encoded = model.encode_SH(SH)
                # SV_encoded = model.encode_SV(SV)
                # Fixed dummy tensors for quick testing
                # SH_encoded = torch.randn(len(SH), 256).cuda()
                # SV_encoded = torch.randn(len(SV), 256).cuda()
                SH_encoded = transform_sequence_and_structure_by_luca_prot(seq=SH, luca_prot_model=luca_prot_model)
                SV_encoded = transform_sequence_and_structure_by_luca_prot(seq=SV, luca_prot_model=luca_prot_model)
                # #################################

                input_x = [SH_encoded, SV_encoded, feature_SH, feature_SV]
                # all_labels = torch.concat((all_labels, additional_vector), dim=0)

                # Discriminator and generation losses computation
                cross_infer_dsc_loss, dsc_loss = model.compute_dsc_loss(input_x, feature_SH.size(0), omics)

                total_cross_infer_dsc_loss += cross_infer_dsc_loss.item()

                total_dsc_loss += dsc_loss.item()
                generate_loss, self_elbo, cross_elbo, cross_infer_loss, _, _, _, _ = model.compute_generate_loss(
                    input_x,
                    feature_SH.size(
                        0), omics)

                total_self_elbo += self_elbo
                total_cross_elbo += cross_elbo
                total_cross_infer_loss += cross_infer_loss

                # Total loss could be a weighted sum or simply the generate_loss
                loss = generate_loss
                total_loss += loss.item()

                embedding = model.get_embedding(input_x, feature_SH.size(0), omics)
                all_embeddings = torch.concat((all_embeddings, embedding), dim=0)

                tepoch.set_postfix(loss=loss.item(), self_elbo=self_elbo, cross_elbo=cross_elbo,
                                   cross_infer_loss=cross_infer_loss, dsc_loss=dsc_loss)

            print('test total loss: ', total_loss / len(test_dataloader))
            Loss.append(total_loss / len(test_dataloader))
            print('test self elbo loss: ', total_self_elbo / len(test_dataloader))
            print('test cross elbo loss: ', total_cross_elbo / len(test_dataloader))
            print('test cross infer loss: ', total_cross_infer_loss / len(test_dataloader))
            print('test dsc loss', total_dsc_loss / len(test_dataloader))

            # torch.save(all_embeddings, f'./model/model_dict/SH_SV_test_embedding_fold{fold}_epoch{epoch}.pt')
            # torch.save(all_labels, f'./model/model_dict/SH_SV_test_labels_fold{fold}_epoch{epoch}.pt')

    return Loss


def train_pretrain(train_dataloader, model: TMO_Net, luca_prot_model: SequenceAndStructureFusionNetwork, epoch,
                   optimizer, dsc_optimizer, omics: typing.Dict, fold):
    model.train()
    print(f'-----start epoch {epoch} training-----')
    total_loss = 0
    total_dsc_loss = 0
    total_ad_loss = 0
    Loss = []
    pancancer_embedding = torch.Tensor([]).cuda()
    # all_label = torch.Tensor([]).cuda()

    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            SH = data['S_H']  # Assuming these need processing to be tensors
            SV = data['S_V']  # Assuming these need processing to be tensors
            feature_SH = data['feature_SH'].cuda()
            feature_SV = data['feature_SV'].cuda()

            # Assuming a method to encode SH and SV into tensors
            # SH_encoded = model.encode_SH(SH)
            # SV_encoded = model.encode_SV(SV)
            # Fixed dummy tensors for quick testing
            # SH_encoded = torch.randn(len(SH), 256).cuda()
            # SV_encoded = torch.randn(len(SV), 256).cuda()
            SH_encoded = transform_sequence_and_structure_by_luca_prot(seq=SH, luca_prot_model=luca_prot_model)
            SV_encoded = transform_sequence_and_structure_by_luca_prot(seq=SV, luca_prot_model=luca_prot_model)
            # #################################

            input_x = [SH_encoded, SV_encoded, feature_SH, feature_SV]
            # all_label = torch.concat((all_label, additional_vector), dim=0)  # Just an example

            un_dfs_freeze(model.discriminator)
            un_dfs_freeze(model.infer_discriminator)
            cross_infer_dsc_loss, dsc_loss = model.compute_dsc_loss(input_x, feature_SH.size(0), omics)
            ad_loss = cross_infer_dsc_loss + dsc_loss
            total_ad_loss += dsc_loss.item()

            dsc_optimizer.zero_grad()
            ad_loss.backward(retain_graph=True)
            dsc_optimizer.step()

            dfs_freeze(model.discriminator)
            dfs_freeze(model.infer_discriminator)
            loss, _, _, _, _, _, _, _ = model.compute_generate_loss(input_x, feature_SH.size(0), omics)

            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())

        print('total loss: ', total_loss / len(train_dataloader))
        Loss.append(total_loss / len(train_dataloader))
        print('ad loss', total_ad_loss / len(train_dataloader))
        print('dsc loss', total_dsc_loss / len(train_dataloader))
        Loss.append(total_dsc_loss / len(train_dataloader))

        # Assuming some method to evaluate pretrain score
        # pretrain_score = evaluate_pretrain_score(pancancer_embedding, all_label)
        # print('pretrain score:', pretrain_score)
        # pretrain_score

        return Loss


def SH_SV_Dataset_pretrain(train_dataloader, test_dataloader, epochs, device, omics: typing.Dict, fold, file_type=""):
    # model init
    model = TMO_Net(4, [256, 256, 1200, 1200], 64, [256, 128, 64],
                    omics_data=['gaussian', 'gaussian', 'gaussian', 'gaussian'])  # 模型初始化，参数需根据实际情况调整
    # torch.cuda.set_device(device_id)
    # feature 1200 -> 256 -> 128 -> 64 (latent_dim)
    # seq -luca_prot-> 256 -> 256 -> 128 -> 64 (latent_dim)
    model = model.to(device)

    # luca prot load
    luca_prot_model = load_lucaprot()
    luca_prot_model.eval()
    # luca prot load

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    dsc_parameters = list(model.discriminator.parameters()) + list(model.infer_discriminator.parameters())
    dsc_optimizer = torch.optim.Adam(dsc_parameters, lr=0.0001)

    Loss_list = []
    test_Loss_list = []
    pretrain_score_list = []
    for epoch in range(epochs):
        start_time = time.time()

        loss = train_pretrain(train_dataloader, model, luca_prot_model, epoch, optimizer, dsc_optimizer,
                              omics, fold)
        Loss_list.append(loss)
        valid_loss = val_pretrain(test_dataloader, model, luca_prot_model, epoch, omics, fold)
        test_Loss_list.append(valid_loss)
        print(f'fold{fold} time used: ', time.time() - start_time)
        model_dict = model.state_dict()
        torch.save(model_dict,
                   f'./model/model_dict/{file_type}_SH_SV_pretrain_model_epoch{epoch}_fold{fold}_train_loss_{loss}_val_loss_{valid_loss}.pt')

    Loss_list = torch.Tensor(Loss_list)
    test_Loss_list = torch.Tensor(test_Loss_list)
    pretrain_score_list = pd.DataFrame(pretrain_score_list, columns=['pretrain_score'])
    pretrain_score_list.to_csv(f'{file_type}_pretrain_score_list_fold{fold}.csv')

    torch.save(test_Loss_list, f'./model/model_dict/{file_type}_SH_SV_pretrain_test_loss_fold{fold}.pt')
    torch.save(Loss_list, f'./model/model_dict/{file_type}_SH_SV_pretrain_train_loss_fold{fold}.pt')


def start_pretrain():
    ### 完整的sh，sv模态补全的预训练 ###
    csv_file = 'data/SH_SV_feature_with_vectors.csv'  # Your CSV file path
    file_type = ''
    ### 完整的sh，sv模态补全的预训练 ###
    ### 部分sh，sv 模态补全的预训练 ###
    #csv_file = 'data/SH_SV_feature_with_vectors_partial.csv'  # Your CSV file path
    #file_type = "[partial]"
    ### 部分sh，sv 模态补全的预训练 ###
    fold_index = 3  # Choose which fold to use (0-9 for ten-fold CV)

    train_loader, valid_loader, test_loader = get_dataloaders_for_fold(csv_file,
                                                                       fold_index=fold_index,
                                                                       batch_size=2)
    epochs = 10
    omics = {'sh': 0, 'sv': 1, 'sh_feature': 2, 'sv_feature': 3}

    SH_SV_Dataset_pretrain(train_dataloader=train_loader, test_dataloader=test_loader, epochs=epochs, device='cuda',
                           omics=omics, fold=fold_index, file_type=file_type)


if __name__ == '__main__':
    start_pretrain()
