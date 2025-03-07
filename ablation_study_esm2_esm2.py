import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from SH_SV_dataset import get_relation_dataloaders_for_fold
import os
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Get current file name and setup logging
current_file_name = os.path.basename(__file__)
log_file = f'logs/{current_file_name.replace(".py", ".log")}'

# Create logs directory if not exists
if not os.path.exists(os.path.dirname(log_file)):
    os.makedirs(os.path.dirname(log_file))

# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler(log_file, mode='a')
console_handler = logging.StreamHandler(sys.__stdout__)

# Set levels for handlers
file_handler.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)

# Create formatter and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n':
            self.level(message.strip())
            logger.handlers[0].flush()
            logger.handlers[1].flush()

    def flush(self):
        pass

sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)

class AblationClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 2)

    def forward(self, x):
        x = self.layers(x)
        return self.classifier(x)

class ESM2Encoder:
    def __init__(self, device='cuda', max_length=1024):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, sequence):
        inputs = self.tokenizer(sequence, 
                              return_tensors="pt", 
                              padding=True,
                              truncation=True,
                              max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

def train_epoch(model, dataloader, optimizer, criterion, encoders, device='cuda'):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        sh_seqs = batch['S_H']
        sv_seqs = batch['S_V']
        labels = batch['labels'].to(device)
        
        sh_encoded = encoders[0].encode(sh_seqs)
        sv_encoded = encoders[1].encode(sv_seqs)
        combined = torch.cat([sh_encoded, sv_encoded], dim=1)
        
        optimizer.zero_grad()
        outputs = model(combined)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary')
    rec = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return total_loss / len(dataloader), acc, prec, rec, f1

def evaluate(model, dataloader, criterion, encoders, device='cuda'):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            sh_seqs = batch['S_H']
            sv_seqs = batch['S_V']
            labels = batch['labels'].to(device)
            
            sh_encoded = encoders[0].encode(sh_seqs)
            sv_encoded = encoders[1].encode(sv_seqs)
            combined = torch.cat([sh_encoded, sv_encoded], dim=1)
            
            outputs = model(combined)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary')
    rec = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return total_loss / len(dataloader), acc, prec, rec, f1

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create necessary directories
    os.makedirs('model', exist_ok=True)
    
    # Load data
    train_loader, valid_loader, _ = get_relation_dataloaders_for_fold(
        'data/SH_SV_feature_with_vectors_0.5.csv',
        fold_index=3,
        batch_size=16
    )
    
    # Initialize encoders
    sh_encoder = ESM2Encoder(device)
    sv_encoder = ESM2Encoder(device)
    
    # Calculate input dimensions
    input_dim = 320 + 320  # ESM2 dimensions
    
    # Initialize model and training components
    model = AblationClassifier(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_acc = 0
    best_metrics = None
    
    print(f"Starting ESM2-ESM2 experiment...")
    
    for epoch in range(EPOCH):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, [sh_encoder, sv_encoder], device)
        
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, valid_loader, criterion, [sh_encoder, sv_encoder], device)
        
        print(f"Epoch {epoch+1}:")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {
                'acc': val_acc,
                'precision': val_prec,
                'recall': val_rec,
                'f1': val_f1
            }
            torch.save(model.state_dict(), f'model/ablation_ESM2_ESM2_best.pt')
    
    print("\nBest validation metrics:")
    print(f"Accuracy: {best_metrics['acc']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    
    # Save results to CSV
    import pandas as pd
    results_df = pd.DataFrame([best_metrics])
    results_df.to_csv('ESM2_ESM2_results.csv')
    print("\nResults saved to ESM2_ESM2_results.csv")

EPOCH = 30
if __name__ == '__main__':
    main()
