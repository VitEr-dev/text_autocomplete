import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence  # Добавьте эту строку
from sklearn.metrics import accuracy_score
import numpy as np

def custom_collate_fn(batch, pad_token_id=0):
    X_batch, y_batch = zip(*batch)
    
    X_tensors = []
    for x in X_batch:
        if torch.is_tensor(x):
            X_tensors.append(x.detach().clone().to(torch.long))
        else:
            X_tensors.append(torch.tensor(x, dtype=torch.long))
    
    if torch.is_tensor(y_batch[0]):
        y_tensor = torch.stack([y.detach().clone().to(torch.long) for y in y_batch])
    else:
        y_tensor = torch.tensor(y_batch, dtype=torch.long)
    
    X_padded = pad_sequence(X_tensors, batch_first=True, padding_value=pad_token_id)
    return X_padded, y_tensor

def train_with_gradient_clipping(model, train_loader, val_loader, num_epochs=10, 
                                 learning_rate=0.001, max_grad_norm=1.0, weight_decay=1e-5,
                                 model_save_path='./models/trained_model.pth'):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    train_losses = []
    val_losses = []
    learning_rates = []
    
    best_val_loss = float('inf')
    patience = 3
    no_improvement_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total_train = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if len(X_batch) == 0:
                continue
                
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            attention_mask = model.get_attention_mask(X_batch)
            
            optimizer.zero_grad()
            outputs, _ = model(X_batch, attention_mask)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            total_train += X_batch.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Валидация
        model.eval()
        val_loss = 0
        total_val = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                if len(X_batch) == 0:
                    continue
                    
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                attention_mask = model.get_attention_mask(X_batch)
                
                outputs, _ = model(X_batch, attention_mask)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                total_val += X_batch.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(y_batch.cpu())
        
        avg_train_loss = train_loss / total_train if total_train > 0 else 0
        avg_val_loss = val_loss / total_val if total_val > 0 else 0
        
        if all_preds:
            all_preds = torch.cat(all_preds).numpy()
            all_targets = torch.cat(all_targets).numpy()
            val_accuracy = accuracy_score(all_targets, all_preds)
        else:
            val_accuracy = 0.0
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        learning_rates.append(current_lr)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy:.4f}, LR: {current_lr:.6f}')
        print('-' * 50)
    
    return model, train_losses, val_losses, learning_rates