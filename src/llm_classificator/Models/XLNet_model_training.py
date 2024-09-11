from transformers import XLNetForSequenceClassification, AdamW
from typing import Any, Dict
from datetime import datetime
from tqdm import tqdm
import torch

def train_epoch(model, dataloader, optimizer, device):
    model = model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

def xlNet_model_training(data: Dict[str, Any], epochs: int) -> XLNetForSequenceClassification:
    
    train_dataloader = data['train_dataloader']
    eval_dataloader = data['eval_dataloader']
    
    model_name = 'xlnet-base-cased'
    model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    print('Execution init datetime: ' + str(datetime.now()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}')
        
    return model