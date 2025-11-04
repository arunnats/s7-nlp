#!/usr/bin/env python3
"""
Train BERT entailment models for LegalSumm
Fine-tunes bert-base-multilingual-cased for each strategy
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# Configuration based on paper
CONFIG = {
    'model_name': 'bert-base-multilingual-cased',
    'max_chunk_length': 350,
    'max_summary_length': 150,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'num_epochs': 3,  # Paper: 6K steps ≈ 3 epochs with typical dataset
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
}

class EntailmentDataset(Dataset):
    """Dataset for chunk-summary entailment pairs"""
    
    def __init__(self, chunks, summaries, labels, tokenizer, max_chunk_len, max_summary_len):
        self.encodings = []
        
        for chunk, summary in tqdm(zip(chunks, summaries), total=len(chunks), desc="Tokenizing"):
            # Tokenize chunk and summary separately, then combine
            chunk_tokens = tokenizer(
                chunk,
                max_length=max_chunk_len,
                truncation=True,
                add_special_tokens=False
            )
            summary_tokens = tokenizer(
                summary,
                max_length=max_summary_len,
                truncation=True,
                add_special_tokens=False
            )
            
            # Combine: [CLS] chunk [SEP] summary [SEP]
            input_ids = (
                [tokenizer.cls_token_id] +
                chunk_tokens['input_ids'] +
                [tokenizer.sep_token_id] +
                summary_tokens['input_ids'] +
                [tokenizer.sep_token_id]
            )
            
            # Token type IDs: 0 for chunk, 1 for summary
            token_type_ids = (
                [0] * (len(chunk_tokens['input_ids']) + 2) +
                [1] * (len(summary_tokens['input_ids']) + 1)
            )
            
            attention_mask = [1] * len(input_ids)
            
            self.encodings.append({
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask
            })
        
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def collate_fn(batch):
    """Pad sequences to same length in batch with BERT max length enforcement"""
    max_len = min(max(len(item['input_ids']) for item in batch), 512)  # BERT max is 512
    
    input_ids = []
    token_type_ids = []
    attention_mask = []
    labels = []
    
    for item in batch:
        input_len = min(len(item['input_ids']), 512)  # Truncate if > 512
        padding_len = max_len - input_len
        
        input_ids.append(
            torch.cat([item['input_ids'][:512], torch.zeros(padding_len, dtype=torch.long)])
        )
        token_type_ids.append(
            torch.cat([item['token_type_ids'][:512], torch.zeros(padding_len, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([item['attention_mask'][:512], torch.zeros(padding_len, dtype=torch.long)])
        )
        labels.append(item['labels'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'token_type_ids': torch.stack(token_type_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels)
    }

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100 * correct / total:.2f}%"
        })
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    
    return total_loss / len(dataloader), correct / total

def train_strategy(strategy_num):
    """Train entailment model for one strategy"""
    print(f"\n{'='*60}")
    print(f"Training Entailment Model for Strategy {strategy_num}")
    print('='*60)
    
    # Load data
    data_file = f"entailment_data/entailment_train_strategy_{strategy_num}.csv"
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} examples")
    
    # Split into train/val
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    model = BertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = EntailmentDataset(
        train_df['chunk'].tolist(),
        train_df['summary'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        CONFIG['max_chunk_length'],
        CONFIG['max_summary_length']
    )
    
    val_dataset = EntailmentDataset(
        val_df['chunk'].tolist(),
        val_df['summary'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        CONFIG['max_chunk_length'],
        CONFIG['max_summary_length']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        collate_fn=collate_fn
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    total_steps = len(train_loader) * CONFIG['num_epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_acc = 0
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir = f"models/entailment/strategy_{strategy_num}"
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
    
    print(f"\nCompleted Strategy {strategy_num} - Best Val Acc: {best_val_acc:.4f}")

def main():
    print("="*60)
    print("Training BERT Entailment Models for LegalSumm")
    print("="*60)
    
    # Create output directory
    os.makedirs("models/entailment", exist_ok=True)
    
    # Train all 8 strategies
    for strategy in range(1, 9):
        train_strategy(strategy)
    
    print("\n" + "="*60)
    print("All entailment models trained successfully!")
    print("Models saved in: models/entailment/")
    print("="*60)

if __name__ == "__main__":
    main()
