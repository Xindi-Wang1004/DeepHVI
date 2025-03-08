import logging
import os
import sys
import time
import typing

import torch.nn.functional as F
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from ESM2SequenceTransformer import transform_sequence_by_esm2
from LucaProt.src.SSFN.model import SequenceAndStructureFusionNetwork
from SH_SV_dataset import get_relation_dataloaders_for_fold
# from SH_SV_pretrain import transform_sequence_and_structure_by_luca_prot, load_lucaprot
from sh_esm2_sv_esm2_SH_SV_pretrain import transform_sequence_and_structure_by_luca_prot, load_lucaprot
from TMONet.model.TMO_Net_model import DownStream_predictor, TMO_Net
from TMONet.util.loss_function import reconstruction_loss

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


def infer_single_sample(model_path, data: typing.Dict, device='cuda', dataloader=None):
    # Load the model
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    # luca prot load
    luca_prot_model = load_lucaprot()
    luca_prot_model.eval()
    # luca prot load

    omics = {'sh': 0, 'sv': 1, 'sh_feature': 2, 'sv_feature': 3}

    SH = data['S_H']  # Assuming these need processing to be tensors
    SV = data['S_V']  # Assuming these need processing to be tensors
    feature_SH = torch.tensor(eval(data['feature_SH']), dtype=torch.float, device=device)
    feature_SV = torch.tensor(eval(data['feature_SV']), dtype=torch.float, device=device)

    # SH_encoded = transform_sequence_and_structure_by_luca_prot(seq=SH, luca_prot_model=luca_prot_model)
    # SV_encoded = transform_sequence_and_structure_by_luca_prot(seq=SV, luca_prot_model=luca_prot_model)
    SH_encoded = transform_sequence_by_esm2(sequences=SH, transformer=None)
    SV_encoded = transform_sequence_by_esm2(sequences=SV, transformer=None)
    input_x = [SH_encoded, SV_encoded, feature_SH, feature_SV]

    with torch.no_grad():
        output = model(input_x, feature_SH.size(0), omics)
        # _, prediction = torch.max(output, 1)
        probabilities = F.softmax(output, dim=1)

        # 提取负类（第一个）和正类（第二个）的概率
        negative_prob = probabilities[0, 0].item()
        positive_prob = probabilities[0, 1].item()

    with open("infer_single_sample.csv",'w') as f:
        f.write("sh,sv,score\n")
        f.write(f"{SH},{SV},{positive_prob}\n")

    return positive_prob


def cross_modal_generation_main():
    generation_modal = 'sv'
    data_key_map = {'sh': 'S_H', 'sv': 'S_V', 'sh_feature': 'feature_SH', 'sv_feature': 'feature_SV'}

    df = pd.read_csv('data/SH_SV_feature_with_vectors_0.5.csv')
    seq_list = df[data_key_map[generation_modal]].tolist()
    seq_list = list(set(seq_list))
    seq_list = seq_list[:100]
    data = {
        'S_H': 'YYYGMDVWGQGTTVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK',
        'S_V': 'MRVKEKYQHLWRWGWRWGTMLLGMLMICSATEKLWVTVYYGVPVWKEATTTLFCASDAKAYDTEVHNVWATHACVPTDPNPQEVVLVNVTENFNMWKNDMVEQMHEDIISLWDQSLKPCVKLTPLCVSLKCTDLKNDTNTNSSSGRMIMEKGEIKNCSFNISTSIRGKVQKEYAFFYKLDIIPIDNDTTSYKLTSCNTSVITQACPKVSFEPIPIHYCAPAGFAILKCNNKTFNGTGPCTNVSTVQCTHGIRPVVSTQLLLNGSLAEEEVVIRSVNFTDNAKTIIVQLNTSVEINCTRPNNNTRKRIRIQRGPGRAFVTIGKIGNMRQAHCNISRAKWNNTLKQIASKLREQFGNNKTIIFKQSSGGDPEIVTHSFNCGGEFFYCNSTQLFNSTWFNSTWSTEGSNNTEGSDTITLPCRIKQIINMWQKVGKAMYAPPISGQIRCSSNITGLLLTRDGGNSNNESEIFRPGGGDMRDNWRSELYKYKVVKIEPLGVAPTKAKRRVVQREKRAVGIGALFLGFLGAAGSTMGAASMTLTVQARQLLSGIVQQQNNLLRAIEAQQHLLQLTVWGIKQLQARILAVERYLKDQQLLGIWGCSGKLICTTAVPWNASWSNKSLEQIWNHTTWMEWDREINNYTSLIHSLIEESQNQQEKNEQELLELDKWASLWNWFNITNWLWYIKLFIMIVGGLVGLRIVFAVLSIVNRVRQGYSPLSFQTHLPTPRGPDRPEGIEEEGGERDRDRSIRLVNGSLALIWDDLRSLCLFSYHRLRDLLLIVTRIVELLGRRGWEALKYWWNLLQYWSQELKNSAVSLLNATAIAVAEGTDRVIEVVQGACRAIRHIPRRIRQGLERILL',
        'feature_SH': "[194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 165.8, 1.53, 3.93, 0.86, 124.0, 2.4, 1.3, 0.0, 3.14, 1.19, 202.65, -0.3, 114.4, 0.92, 0.94, 0.42, -78.0, 5.5, 0.0, -1.0, 3.89, 0.48, 194.91, 0.19, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 146.9, 1.22, 1.14, 0.8, -73.0, 4.4, 0.0, 0.0, 3.17, 0.95, 235.51, 0.14, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 190.8, 1.19, 10.99, 0.59, 189.0, 3.9, 2.5, 0.0, 6.36, 1.25, 204.74, -0.38, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 102.5, 0.94, 2.56, 0.42, 168.0, 1.9, 0.0, 0.0, 1.07, 1.16, 219.79, -0.38, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 114.4, 0.92, 0.94, 0.42, -78.0, 5.5, 0.0, -1.0, 3.89, 0.48, 194.91, 0.19, 194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 190.8, 1.19, 10.99, 0.59, 189.0, 3.9, 2.5, 0.0, 6.36, 1.25, 204.74, -0.38, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 138.8, 1.67, 0.94, 0.53, -106.0, 7.1, 0.0, -1.0, 4.8, 0.61, 223.16, 0.23, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 122.4, 0.64, 2.31, 0.39, -74.0, 4.0, 0.0, 0.0, 3.71, 0.6, 207.9, 0.13, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 157.5, 0.98, 0.47, 0.57, 50.0, 2.1, 0.5, 0.0, 1.88, 0.93, 242.54, -0.04, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 190.8, 1.19, 10.99, 0.59, 189.0, 3.9, 2.5, 0.0, 6.36, 1.25, 204.74, -0.38, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 146.9, 1.22, 1.14, 0.8, -73.0, 4.4, 0.0, 0.0, 3.17, 0.95, 235.51, 0.14, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 146.9, 1.22, 1.14, 0.8, -73.0, 4.4, 0.0, 0.0, 3.17, 0.95, 235.51, 0.14, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 163.0, 1.04, 13.73, 0.84, 151.0, 5.2, 1.8, 0.0, 6.47, 1.81, 233.21, -0.34]",
        'feature_SV': "[165.8, 1.53, 3.93, 0.86, 124.0, 2.4, 1.3, 0.0, 3.14, 1.19, 202.65, -0.3, 190.3, 1.18, 0.27, 0.77, -70.0, 4.9, 0.0, 1.0, 3.96, 0.93, 341.01, 0.07, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 138.8, 1.67, 0.94, 0.53, -106.0, 7.1, 0.0, -1.0, 4.8, 0.61, 223.16, 0.23, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 146.9, 1.22, 1.14, 0.8, -73.0, 4.4, 0.0, 0.0, 3.17, 0.95, 235.51, 0.14, 157.5, 0.98, 0.47, 0.57, 50.0, 2.1, 0.5, 0.0, 1.88, 0.93, 242.54, -0.04, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 190.3, 1.18, 0.27, 0.77, -70.0, 4.9, 0.0, 1.0, 3.96, 0.93, 341.01, 0.07, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 190.3, 1.18, 0.27, 0.77, -70.0, 4.9, 0.0, 1.0, 3.96, 0.93, 341.01, 0.07, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 165.8, 1.53, 3.93, 0.86, 124.0, 2.4, 1.3, 0.0, 3.14, 1.19, 202.65, -0.3, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 165.8, 1.53, 3.93, 0.86, 124.0, 2.4, 1.3, 0.0, 3.14, 1.19, 202.65, -0.3, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 165.8, 1.53, 3.93, 0.86, 124.0, 2.4, 1.3, 0.0, 3.14, 1.19, 202.65, -0.3, 163.0, 1.04, 13.73, 0.84, 151.0, 5.2, 1.8, 0.0, 6.47, 1.81, 233.21, -0.34, 102.5, 0.94, 2.56, 0.42, 168.0, 1.9, 0.0, 0.0, 1.07, 1.16, 219.79, -0.38, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 138.8, 1.67, 0.94, 0.53, -106.0, 7.1, 0.0, -1.0, 4.8, 0.61, 223.16, 0.23, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 138.8, 1.67, 0.94, 0.53, -106.0, 7.1, 0.0, -1.0, 4.8, 0.61, 223.16, 0.23, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 190.8, 1.19, 10.99, 0.59, 189.0, 3.9, 2.5, 0.0, 6.36, 1.25, 204.74, -0.38, 102.5, 0.94, 2.56, 0.42, 168.0, 1.9, 0.0, 0.0, 1.07, 1.16, 219.79, -0.38, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 114.4, 0.92, 0.94, 0.42, -78.0, 5.5, 0.0, -1.0, 3.89, 0.48, 194.91, 0.19, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 114.4, 0.92, 0.94, 0.42, -78.0, 5.5, 0.0, -1.0, 3.89, 0.48, 194.91, 0.19, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 138.8, 1.67, 0.94, 0.53, -106.0, 7.1, 0.0, -1.0, 4.8, 0.61, 223.16, 0.23, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 157.5, 0.98, 0.47, 0.57, 50.0, 2.1, 0.5, 0.0, 1.88, 0.93, 242.54, -0.04, 122.4, 0.64, 2.31, 0.39, -74.0, 4.0, 0.0, 0.0, 3.71, 0.6, 207.9, 0.13, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 157.5, 0.98, 0.47, 0.57, 50.0, 2.1, 0.5, 0.0, 1.88, 0.93, 242.54, -0.04, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 102.5, 0.94, 2.56, 0.42, 168.0, 1.9, 0.0, 0.0, 1.07, 1.16, 219.79, -0.38, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 114.4, 0.92, 0.94, 0.42, -78.0, 5.5, 0.0, -1.0, 3.89, 0.48, 194.91, 0.19, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 122.4, 0.64, 2.31, 0.39, -74.0, 4.0, 0.0, 0.0, 3.71, 0.6, 207.9, 0.13, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 146.9, 1.22, 1.14, 0.8, -73.0, 4.4, 0.0, 0.0, 3.17, 0.95, 235.51, 0.14, 138.8, 1.67, 0.94, 0.53, -106.0, 7.1, 0.0, -1.0, 4.8, 0.61, 223.16, 0.23, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 122.4, 0.64, 2.31, 0.39, -74.0, 4.0, 0.0, 0.0, 3.71, 0.6, 207.9, 0.13, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 138.8, 1.67, 0.94, 0.53, -106.0, 7.1, 0.0, -1.0, 4.8, 0.61, 223.16, 0.23, 122.4, 0.64, 2.31, 0.39, -74.0, 4.0, 0.0, 0.0, 3.71, 0.6, 207.9, 0.13, 190.8, 1.19, 10.99, 0.59, 189.0, 3.9, 2.5, 0.0, 6.36, 1.25, 204.74, -0.38, 122.4, 0.64, 2.31, 0.39, -74.0, 4.0, 0.0, 0.0, 3.71, 0.6, 207.9, 0.13, 165.8, 1.53, 3.93, 0.86, 124.0, 2.4, 1.3, 0.0, 3.14, 1.19, 202.65, -0.3, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 122.4, 0.64, 2.31, 0.39, -74.0, 4.0, 0.0, 0.0, 3.71, 0.6, 207.9, 0.13, 114.4, 0.92, 0.94, 0.42, -78.0, 5.5, 0.0, -1.0, 3.89, 0.48, 194.91, 0.19, 165.8, 1.53, 3.93, 0.86, 124.0, 2.4, 1.3, 0.0, 3.14, 1.19, 202.65, -0.3]"}
    # 模态生成 model_path 参数填写对应的权重，可以为分类模型的权重，模态生成模型的权重，和部分模态补全的权重，根据文件名区分
    # e.g. model_path='model/model_dict/SH_SV_pretrain_model_epoch0_fold0_dim64.pt'
    cross_modal_generation(model_path='model/model_dict/best_cls_model_fold_3.pt',
                           generation_modal=generation_modal,
                           seq_list=seq_list, data=data, top_k=5)


def cross_modal_generation(model_path, generation_modal: str = 'sh', seq_list=[], data: typing.Dict = None,
                           top_k=5, device='cuda',batch_size=10):
    # Load the model
    if "pretrain" in model_path:
        model: TMO_Net = TMO_Net(4, [256, 256, 1200, 1200], 64, [256, 128, 64],
                                 omics_data=['gaussian', 'gaussian', 'gaussian', 'gaussian'])  # 模型初始化，参数需根据实际情况调整
        # torch.cuda.set_device(device_id)
        # feature 1200 -> 256 -> 128 -> 64 (latent_dim)
        # seq -luca_prot-> 256 -> 256 -> 128 -> 64 (latent_dim)
        # 读取model_path 的 pt 文件，并将model 的 dict 加载为读取的参数文件
        model_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_dict)
    elif "cls" in model_path:
        model: DownStream_predictor = torch.load(model_path, map_location=device)
        model: TMO_Net = model.cross_encoders
    else:
        raise ValueError("Unsupported model file format. Please use '.pt' or '.pth' files.")

    model = model.to(device)
    model.eval()

    # Luca Prot Load
    luca_prot_model = load_lucaprot()
    luca_prot_model.eval()

    # Definitions for omics data processing
    omics = {'sh': 0, 'sv': 1, 'sh_feature': 2, 'sv_feature': 3}
    generation_modal_index = omics[generation_modal]
    data_key_map = {'sh': 'S_H', 'sv': 'S_V', 'sh_feature': 'feature_SH', 'sv_feature': 'feature_SV'}
    un_complete_omics = {key: value for key, value in omics.items() if value != generation_modal_index}

    SH = data['S_H']
    SV = data['S_V']
    #feature_SH = torch.tensor(eval(data['feature_SH']), dtype=torch.float).cuda()
    #feature_SV = torch.tensor(eval(data['feature_SV']), dtype=torch.float).cuda()
    feature_SH = torch.tensor(eval(data['feature_SH']), dtype=torch.float).to(device)  # 移动到指定设备
    feature_SV = torch.tensor(eval(data['feature_SV']), dtype=torch.float).to(device)  # 移动到指定设备

    # Encode sequences using Luca Prot
    # SH_encoded = transform_sequence_and_structure_by_luca_prot(seq=SH, luca_prot_model=luca_prot_model)
    # SV_encoded = transform_sequence_and_structure_by_luca_prot(seq=SV, luca_prot_model=luca_prot_model)
    SH_encoded = transform_sequence_by_esm2(sequences=SH, transformer=None)
    SV_encoded = transform_sequence_by_esm2(sequences=SV, transformer=None)

    # Create input for model
    input_x = [SH_encoded, SV_encoded, feature_SH, feature_SV]
    filtered_input_x = [x for index, x in enumerate(input_x) if index != generation_modal_index]
    reconstruct_omics = model.cross_modal_generation(input_x=filtered_input_x,
                                                     omics=un_complete_omics)

    # Encoding sequences from seq_list
    seq_encoded_list = [transform_sequence_by_esm2(sequences=seq, transformer=None) for seq in seq_list]

    # Get reconstruction output
    reconstructed_output = reconstruct_omics[generation_modal_index]

    # Calculate similarity and sort sequences
    similarities = [F.cosine_similarity(reconstructed_output, seq_encoded, dim=1) for
                    seq_encoded in seq_encoded_list]
    similarities = [sim.item() for sim in similarities]
    top5_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    top_similarities = [similarities[idx] for idx in top5_indices]

    # Print top 5 similar sequences
    # Print top similar sequences
    print(f"Top {top_k} similar sequences:")
    for idx, sim in zip(top5_indices, top_similarities):
        print(f"Sequence: {seq_list[idx]}, Similarity: {sim}")

    infer_loss = 0
    for index, (ground_truth, reconstruct) in enumerate(zip(input_x, reconstruct_omics)):
        if index == generation_modal_index:
            loss = reconstruction_loss(reconstruct, ground_truth, 1.0, 'gaussian')
            infer_loss += loss
            print("Infer loss: {}".format(loss))
            # print(f"ground_truth:{ground_truth}, \nreconstruct:{reconstruct}")
            print(f"ground_truth seq : {data[data_key_map[generation_modal]]}")

    return top5_indices, top_similarities  # Optionally return indices if needed outside the function


def infer_SH_SV_Relation_Dataset(model_path, dataloader, output_file, device='cuda'):
    # Load the model
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    # luca prot load
    luca_prot_model = load_lucaprot()
    luca_prot_model.eval()
    # luca prot load

    all_predictions = []
    all_labels = []

    omics = {'sh': 0, 'sv': 1, 'sh_feature': 2, 'sv_feature': 3}

    with torch.no_grad():
        with tqdm(dataloader, unit='batch') as tepoch:
            tepoch.set_description(f" inference: ")
            for data in tepoch:
                SH = data['S_H']  # Assuming these need processing to be tensors
                SV = data['S_V']  # Assuming these need processing to be tensors
                feature_SH = data['feature_SH'].cuda()
                feature_SV = data['feature_SV'].cuda()
                labels = data['labels']
                # Assuming a method to encode SH and SV into tensors
                # SH_encoded = model.encode_SH(SH)
                # SV_encoded = model.encode_SV(SV)
                # Fixed dummy tensors for quick testing

                # SH_encoded = transform_sequence_and_structure_by_luca_prot(seq=SH, luca_prot_model=luca_prot_model)
                # SV_encoded = transform_sequence_and_structure_by_luca_prot(seq=SV, luca_prot_model=luca_prot_model)
                SH_encoded = transform_sequence_by_esm2(sequences=SH, transformer=None)
                SV_encoded = transform_sequence_by_esm2(sequences=SV, transformer=None)
                # #################################

                input_x = [SH_encoded, SV_encoded, feature_SH, feature_SV]

                # Move tensors to the appropriate device
                labels = labels.cuda()

                # Model prediction
                classification_pred = model(input_x, feature_SH.size(0), omics)
                _, predicted = torch.max(classification_pred, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    # Save results to CSV
    results_df = pd.DataFrame({
        'Predictions': all_predictions,
        'Labels': all_labels
    })
    results_df.to_csv(output_file, index=False)

    return all_predictions, all_labels


def test_classification(dataloader, model: DownStream_predictor, luca_prot_model: SequenceAndStructureFusionNetwork,
                        epoch, fold, optimizer, omics, criterion, factor=1e-4):
    total_loss = 0
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # No gradient needed for evaluation
        with tqdm(dataloader, unit='batch') as tepoch:
            tepoch.set_description(f" Epoch【test】 {epoch}: ")
            all_labels = []
            all_predictions = []

            for data in tepoch:
                SH = data['S_H']  # Assuming these need processing to be tensors
                SV = data['S_V']  # Assuming these need processing to be tensors
                feature_SH = data['feature_SH'].cuda()
                feature_SV = data['feature_SV'].cuda()
                labels = data['labels']
                # Assuming a method to encode SH and SV into tensors
                # SH_encoded = model.encode_SH(SH)
                # SV_encoded = model.encode_SV(SV)
                # Fixed dummy tensors for quick testing

                # SH_encoded = transform_sequence_and_structure_by_luca_prot(seq=SH, luca_prot_model=luca_prot_model)
                # SV_encoded = transform_sequence_and_structure_by_luca_prot(seq=SV, luca_prot_model=luca_prot_model)
                SH_encoded = transform_sequence_by_esm2(sequences=SH, transformer=None)
                SV_encoded = transform_sequence_by_esm2(sequences=SV, transformer=None)
                # #################################

                input_x = [SH_encoded, SV_encoded, feature_SH, feature_SV]

                # Move tensors to the appropriate device
                labels = labels.cuda()

                # Model prediction
                classification_pred = model(input_x, feature_SH.size(0), omics)
                _, labels_pred = torch.max(classification_pred, 1)

                # Calculate loss
                # Compute loss
                pred_loss = criterion(classification_pred, labels)  # pred loss
                pretrain_loss, _, _, _, _, _, _, _ = model.cross_encoders.compute_generate_loss(input_x,
                                                                                                feature_SH.size(
                                                                                                    0),
                                                                                                omics)  # pretrain_loss

                loss = pred_loss + factor * pretrain_loss
                total_loss += loss.item()

                # Collect labels and predictions for metrics calculation
                all_labels.extend(labels.tolist())
                all_predictions.extend(labels_pred.tolist())

                tepoch.set_postfix(loss=loss.item(), pred_loss=pred_loss.item(), pretrain_loss=pretrain_loss.item())

            # Calculate accuracy, precision, recall, and F1 score
            acc = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='macro')
            recall = recall_score(all_labels, all_predictions, average='macro')
            f1 = f1_score(all_labels, all_predictions, average='macro')

            print('fold {} test:, Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.
                  format(fold, total_loss / len(dataloader), acc, precision, recall, f1))

            # all_labels = torch.Tensor(all_labels)
            # torch.save(all_labels, f'{epoch}_test_pancancer_fold{fold}_label.pt')

            return total_loss / len(dataloader), acc, precision, recall, f1


def train_classification(dataloader, model: DownStream_predictor, luca_prot_model: SequenceAndStructureFusionNetwork,
                         epoch, fold, optimizer, omics, criterion, factor=1e-4):
    total_loss = 0
    model.train()

    all_labels = []
    all_predictions = []

    with (tqdm(dataloader, unit='batch') as tepoch):
        for data in tepoch:
            tepoch.set_description(f" Epoch {epoch}: ")
            SH = data['S_H']  # Assuming these need processing to be tensors
            SV = data['S_V']  # Assuming these need processing to be tensors
            feature_SH = data['feature_SH'].cuda()
            feature_SV = data['feature_SV'].cuda()
            labels = data['labels']

            # Assuming a method to encode SH and SV into tensors
            # SH_encoded = model.encode_SH(SH)
            # SV_encoded = model.encode_SV(SV)
            # Fixed dummy tensors for quick testing

            # SH_encoded = transform_sequence_and_structure_by_luca_prot(seq=SH, luca_prot_model=luca_prot_model)
            # SV_encoded = transform_sequence_and_structure_by_luca_prot(seq=SV, luca_prot_model=luca_prot_model)
            SH_encoded = transform_sequence_by_esm2(sequences=SH, transformer=None)
            SV_encoded = transform_sequence_by_esm2(sequences=SV, transformer=None)
            # #################################

            input_x = [SH_encoded, SV_encoded, feature_SH, feature_SV]

            labels = labels.cuda()

            # Model prediction
            classification_pred = model(input_x, feature_SH.size(0), omics)
            _, labels_pred = torch.max(classification_pred, 1)

            # Compute loss
            pred_loss = criterion(classification_pred, labels)  # pred loss
            pretrain_loss, _, _, _, _, _, _, _ = model.cross_encoders.compute_generate_loss(input_x,
                                                                                            feature_SH.size(0),
                                                                                            omics)  # pretrain_loss

            loss = pred_loss + factor * pretrain_loss
            total_loss += loss.item()

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Post-processing for metrics calculation
            all_labels.extend(labels.tolist())
            all_predictions.extend(labels_pred.tolist())

            tepoch.set_postfix(loss=loss.item(), pred_loss=pred_loss.item(), pretrain_loss=pretrain_loss.item())

        # Calculate accuracy, precision, recall, and F1 score
        acc = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')

        print('fold {} train:, Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'
              .format(fold, total_loss / len(dataloader), acc, precision, recall, f1))
        # all_labels = torch.Tensor(all_labels)
        # torch.save(all_labels, f'{epoch}_train_pancancer_fold{fold}_label.pt')

    return total_loss / len(dataloader), acc, precision, recall, f1


def SH_SV_Relation_Dataset_classification(train_dataloader, test_dataloader, epochs,
                                          pretrain_model_path, fixed, omics, fold, device='cuda'):
    criterion = torch.nn.CrossEntropyLoss()
    task = {'output_dim': 2}  # 假设是一个二分类问题

    # 定义网格搜索参数
    param_grid = {
        'learning_rate': [0.0001],
        'factor': [1e-4]
    }

    best_params = None
    best_acc = -1
    best_model = None

    # Load LUCA prot model
    luca_prot_model = load_lucaprot()
    luca_prot_model.eval()

    for params in ParameterGrid(param_grid):
        print(f"Training with parameters: {params}")

        model = DownStream_predictor(4, [256, 256, 1200, 1200], 64, [256, 128, 64],
                                     pretrain_model_path, task,
                                     omics_data=['gaussian', 'gaussian', 'gaussian', 'gaussian'],
                                     fixed=fixed, omics=omics)
        model = model.to(device)

        param_groups = [
            {'params': model.cross_encoders.parameters(), 'lr': params['learning_rate'] / 10},
            {'params': model.downstream_predictor.parameters(), 'lr': params['learning_rate']},
        ]

        optimizer = torch.optim.Adam(param_groups)

        for epoch in range(epochs):
            train_loss, train_acc, train_precision, train_recall, train_f1 = train_classification(
                train_dataloader, model, luca_prot_model, epoch, fold, optimizer, omics, criterion, params['factor'])

            loss, acc, precision, recall, f1 = test_classification(
                test_dataloader, model, luca_prot_model, epoch, fold, optimizer, omics, criterion)

            print(f'Epoch {epoch + 1}, Fold {fold}, Acc: {acc:.4f}, Precision: {precision:.4f}, '
                  f'Recall: {recall:.4f}, F1: {f1:.4f}')

            if acc > best_acc:
                best_acc = acc
                best_params = params
                best_model = model.state_dict()

    print(f"Best parameters: {best_params}")  # {'factor': 0.0001, 'learning_rate': 0.0001}

    # Train final model with best parameters
    final_model = DownStream_predictor(4, [256, 256, 1200, 1200], 64, [256, 128, 64],
                                       pretrain_model_path, task,
                                       omics_data=['gaussian', 'gaussian', 'gaussian', 'gaussian'],
                                       fixed=fixed, omics=omics)
    final_model.load_state_dict(best_model)
    final_model = final_model.to(device)

    param_groups = [
        {'params': final_model.cross_encoders.parameters(), 'lr': best_params['learning_rate'] / 10},
        {'params': final_model.downstream_predictor.parameters(), 'lr': best_params['learning_rate']},
    ]

    optimizer = torch.optim.Adam(param_groups)

    for epoch in range(epochs):
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_classification(
            train_dataloader, final_model, luca_prot_model, epoch, fold, optimizer, omics, criterion,
            best_params['factor'])

        loss, acc, precision, recall, f1 = test_classification(
            test_dataloader, final_model, luca_prot_model, epoch, fold, optimizer, omics, criterion,
            best_params['factor'])

        print(f'Final: Epoch {epoch + 1}, Fold {fold}, Acc: {acc:.4f}, Precision: {precision:.4f}, '
              f'Recall: {recall:.4f}, F1: {f1:.4f}')

        if acc > best_acc:
            best_acc = acc
            torch.save(final_model, f'model/model_dict/best_cls_model_fold_{fold}.pt')
            classification_score = [[fold, acc, precision, recall, f1]]
            classification_score = pd.DataFrame(classification_score,
                                                columns=['fold', 'acc', 'precision', 'recall', 'f1'])
            classification_score.to_csv(f'classification_score_fold{fold}.csv')

    # Clean up GPU memory
    del final_model
    del optimizer
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()

    return loss, best_acc, precision, recall, f1


def load_trained_model():
    # 加载完整模型
    model = torch.load('best_model_structure_fold1.pt')
    return model


def inference(model_path='model/model_dict/best_cls_model_fold_3.pt', output_file='inference_results.csv',
              csv_file='data/SH_SV_feature_with_vectors_0.5.csv', fold_index=0):
    train_loader, valid_loader, test_loader = get_relation_dataloaders_for_fold(csv_file,
                                                                                fold_index=fold_index,
                                                                                batch_size=16)

    # Run inference and save results
    all_predictions, all_labels = infer_SH_SV_Relation_Dataset(model_path, train_loader, output_file)

    # Calculate accuracy, precision, recall, and F1 score
    acc = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print('fold {} infer:, Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'
          .format(fold_index, acc, precision, recall, f1))


def start_cls_train():
    csv_file = 'data/SH_SV_feature_with_vectors_0.5.csv'  # Your CSV file path
    fold_index = 3  # Choose which fold to use (0-9 for ten-fold CV)

    train_loader, valid_loader, test_loader = get_relation_dataloaders_for_fold(csv_file,
                                                                                fold_index=fold_index,
                                                                                batch_size=16)
    epochs = 10  # Increased from 1 to allow for more training
    omics = {'sh': 0, 'sv': 1, 'sh_feature': 2, 'sv_feature': 3}

    loss, acc, precision, recall, f1 = SH_SV_Relation_Dataset_classification(
        train_dataloader=train_loader,
        test_dataloader=valid_loader,
        epochs=epochs,
        device='cuda',
        omics=omics,
        fold=fold_index,
        fixed=False,
        pretrain_model_path='model/model_dict/best_cls_model_fold_3.pt'
    )

    print(f"Final results - Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, F1: {f1:.4f}")


if __name__ == '__main__':
    # start_cls_train()

    ####### 测试集推理 #########
    # model_path: 分类模型权重
    # output_file： 推理结果保存的文件夹
    # csv_file 数据集文件
    # fold_index fold 下标
    # begin
    # inference(model_path='model/model_dict/best_cls_model_fold_0.pt', output_file='inference_results.csv',
    #           csv_file='data/SH_SV_feature_with_vectors_0.5.csv', fold_index=0)
    # end
    ####### 测试集推理 #########

    # 跨模态重建 #
    # cross_modal_generation_main()
    # 跨模态重建 #

    ######## 单样本推理 ###############
    # embedding 的数据在 model/model_embedding 文件夹中
    # SH_SV_embedding_output_matrix.pt
    # SH_SV_embedding_tensor_matrix.pt
    # model_path: 分类模型权重
    # data： 单条样本数据
    # begin
    # data = {
    #     'S_H': 'YYYGMDVWGQGTTVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK',
    #     'S_V': 'MRVKEKYQHLWRWGWRWGTMLLGMLMICSATEKLWVTVYYGVPVWKEATTTLFCASDAKAYDTEVHNVWATHACVPTDPNPQEVVLVNVTENFNMWKNDMVEQMHEDIISLWDQSLKPCVKLTPLCVSLKCTDLKNDTNTNSSSGRMIMEKGEIKNCSFNISTSIRGKVQKEYAFFYKLDIIPIDNDTTSYKLTSCNTSVITQACPKVSFEPIPIHYCAPAGFAILKCNNKTFNGTGPCTNVSTVQCTHGIRPVVSTQLLLNGSLAEEEVVIRSVNFTDNAKTIIVQLNTSVEINCTRPNNNTRKRIRIQRGPGRAFVTIGKIGNMRQAHCNISRAKWNNTLKQIASKLREQFGNNKTIIFKQSSGGDPEIVTHSFNCGGEFFYCNSTQLFNSTWFNSTWSTEGSNNTEGSDTITLPCRIKQIINMWQKVGKAMYAPPISGQIRCSSNITGLLLTRDGGNSNNESEIFRPGGGDMRDNWRSELYKYKVVKIEPLGVAPTKAKRRVVQREKRAVGIGALFLGFLGAAGSTMGAASMTLTVQARQLLSGIVQQQNNLLRAIEAQQHLLQLTVWGIKQLQARILAVERYLKDQQLLGIWGCSGKLICTTAVPWNASWSNKSLEQIWNHTTWMEWDREINNYTSLIHSLIEESQNQQEKNEQELLELDKWASLWNWFNITNWLWYIKLFIMIVGGLVGLRIVFAVLSIVNRVRQGYSPLSFQTHLPTPRGPDRPEGIEEEGGERDRDRSIRLVNGSLALIWDDLRSLCLFSYHRLRDLLLIVTRIVELLGRRGWEALKYWWNLLQYWSQELKNSAVSLLNATAIAVAEGTDRVIEVVQGACRAIRHIPRRIRQGLERILL',
    #     'feature_SH': "[194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
    #     'feature_SV': "[165.8, 1.53, 3.93, 0.86, 124.0, 2.4, 1.3, 0.0, 3.14, 1.19, 202.65, -0.3, 190.3, 1.18, 0.27, 0.77, -70.0, 4.9, 0.0, 1.0, 3.96, 0.93, 341.01, 0.07, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 138.8, 1.67, 0.94, 0.53, -106.0, 7.1, 0.0, -1.0, 4.8, 0.61, 223.16, 0.23, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 146.9, 1.22, 1.14, 0.8, -73.0, 4.4, 0.0, 0.0, 3.17, 0.95, 235.51, 0.14, 157.5, 0.98, 0.47, 0.57, 50.0, 2.1, 0.5, 0.0, 1.88, 0.93, 242.54, -0.04, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 190.3, 1.18, 0.27, 0.77, -70.0, 4.9, 0.0, 1.0, 3.96, 0.93, 341.01, 0.07, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 190.3, 1.18, 0.27, 0.77, -70.0, 4.9, 0.0, 1.0, 3.96, 0.93, 341.01, 0.07, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 165.8, 1.53, 3.93, 0.86, 124.0, 2.4, 1.3, 0.0, 3.14, 1.19, 202.65, -0.3, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 165.8, 1.53, 3.93, 0.86, 124.0, 2.4, 1.3, 0.0, 3.14, 1.19, 202.65, -0.3, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 165.8, 1.53, 3.93, 0.86, 124.0, 2.4, 1.3, 0.0, 3.14, 1.19, 202.65, -0.3, 163.0, 1.04, 13.73, 0.84, 151.0, 5.2, 1.8, 0.0, 6.47, 1.81, 233.21, -0.34, 102.5, 0.94, 2.56, 0.42, 168.0, 1.9, 0.0, 0.0, 1.07, 1.16, 219.79, -0.38, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 138.8, 1.67, 0.94, 0.53, -106.0, 7.1, 0.0, -1.0, 4.8, 0.61, 223.16, 0.23, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 63.8, 0.46, 6.17, 0.0, -13.0, 7.1, 0.0, 0.0, 8.51, 0.61, 127.9, 0.09, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 138.8, 1.67, 0.94, 0.53, -106.0, 7.1, 0.0, -1.0, 4.8, 0.61, 223.16, 0.23, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 190.8, 1.19, 10.99, 0.59, 189.0, 3.9, 2.5, 0.0, 6.36, 1.25, 204.74, -0.38, 102.5, 0.94, 2.56, 0.42, 168.0, 1.9, 0.0, 0.0, 1.07, 1.16, 219.79, -0.38, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 94.2, 0.7, 5.58, 0.53, -70.0, 6.6, 0.0, 0.0, 6.26, 0.82, 174.06, 0.12, 114.4, 0.92, 0.94, 0.42, -78.0, 5.5, 0.0, -1.0, 3.89, 0.48, 194.91, 0.19, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 194.6, 0.69, 3.13, 0.72, 53.0, 3.1, 2.3, 0.0, 3.28, 1.53, 229.15, -0.29, 114.4, 0.92, 0.94, 0.42, -78.0, 5.5, 0.0, -1.0, 3.89, 0.48, 194.91, 0.19, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 138.8, 1.67, 0.94, 0.53, -106.0, 7.1, 0.0, -1.0, 4.8, 0.61, 223.16, 0.23, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 157.5, 0.98, 0.47, 0.57, 50.0, 2.1, 0.5, 0.0, 1.88, 0.93, 242.54, -0.04, 122.4, 0.64, 2.31, 0.39, -74.0, 4.0, 0.0, 0.0, 3.71, 0.6, 207.9, 0.13, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 157.5, 0.98, 0.47, 0.57, 50.0, 2.1, 0.5, 0.0, 1.88, 0.93, 242.54, -0.04, 89.3, 1.43, 9.36, 0.96, 16.0, 7.9, 0.5, 0.0, 9.25, 0.92, 154.33, -0.04, 102.5, 0.94, 2.56, 0.42, 168.0, 1.9, 0.0, 0.0, 1.07, 1.16, 219.79, -0.38, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 114.4, 0.92, 0.94, 0.42, -78.0, 5.5, 0.0, -1.0, 3.89, 0.48, 194.91, 0.19, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 122.4, 0.64, 2.31, 0.39, -74.0, 4.0, 0.0, 0.0, 3.71, 0.6, 207.9, 0.13, 121.6, 0.49, 1.96, -2.5, -20.0, 5.3, 0.0, 0.0, 4.36, 0.4, 179.93, 0.19, 146.9, 1.22, 1.14, 0.8, -73.0, 4.4, 0.0, 0.0, 3.17, 0.95, 235.51, 0.14, 138.8, 1.67, 0.94, 0.53, -106.0, 7.1, 0.0, -1.0, 4.8, 0.61, 223.16, 0.23, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 163.1, 1.36, 16.64, 0.92, 145.0, 8.6, 1.8, 0.0, 10.94, 1.3, 232.3, -0.37, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 122.4, 0.64, 2.31, 0.39, -74.0, 4.0, 0.0, 0.0, 3.71, 0.6, 207.9, 0.13, 138.2, 0.98, 12.43, 0.63, 123.0, 6.8, 1.5, 0.0, 7.55, 1.81, 207.6, -0.29, 119.6, 0.78, 4.68, 0.54, -38.0, 5.3, 0.4, 0.0, 5.66, 1.12, 205.8, 0.03, 138.8, 1.67, 0.94, 0.53, -106.0, 7.1, 0.0, -1.0, 4.8, 0.61, 223.16, 0.23, 122.4, 0.64, 2.31, 0.39, -74.0, 4.0, 0.0, 0.0, 3.71, 0.6, 207.9, 0.13, 190.8, 1.19, 10.99, 0.59, 189.0, 3.9, 2.5, 0.0, 6.36, 1.25, 204.74, -0.38, 122.4, 0.64, 2.31, 0.39, -74.0, 4.0, 0.0, 0.0, 3.71, 0.6, 207.9, 0.13, 165.8, 1.53, 3.93, 0.86, 124.0, 2.4, 1.3, 0.0, 3.14, 1.19, 202.65, -0.3, 226.4, 1.01, 2.2, 0.58, 145.0, 1.2, 3.4, 0.0, 2.22, 1.54, 237.01, -0.33, 165.1, 1.27, 0.58, 0.73, -141.0, 6.7, 0.0, 1.0, 3.5, 0.7, 300.46, 0.33, 122.4, 0.64, 2.31, 0.39, -74.0, 4.0, 0.0, 0.0, 3.71, 0.6, 207.9, 0.13, 114.4, 0.92, 0.94, 0.42, -78.0, 5.5, 0.0, -1.0, 3.89, 0.48, 194.91, 0.19, 165.8, 1.53, 3.93, 0.86, 124.0, 2.4, 1.3, 0.0, 3.14, 1.19, 202.65, -0.3]"}
    
    # probability = infer_single_sample(model_path='model/model_dict/best_cls_model_fold_3.pt',
    #                                   data=data, device='cuda')
    # print("相关性的概率为：", probability)
    # # end
    ######## 单样本推理 ###############

    ###
    # 增加所有文件的日志保存功能，将控制台的日志重定向到日志文件中
    # 模态补齐任务的tmo-net 的预训练权重，适配到 模态生成中去
    # 增加 完整的sh 前ntoken 的sv补全，后ntoken 补全，增加完整的sv 前ntoken sh补全，后ntoken sh 补全
    # 将二分类任务的 0，1 修改为，概率值：话术：相关性的概率为：，
    ###
