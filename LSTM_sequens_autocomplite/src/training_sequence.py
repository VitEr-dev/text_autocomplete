import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def sequence_collate_fn(batch, pad_token_id=0):
    """Collate function для последовательностей"""
    X_batch, y_batch = zip(*batch)
    
    # Паддинг для входных последовательностей
    X_tensors = [torch.tensor(x, dtype=torch.long) for x in X_batch]
    X_padded = pad_sequence(X_tensors, batch_first=True, padding_value=pad_token_id)
    
    # Паддинг для целевых последовательностей
    y_tensors = [torch.tensor(y, dtype=torch.long) for y in y_batch]
    y_padded = pad_sequence(y_tensors, batch_first=True, padding_value=pad_token_id)
    
    return X_padded, y_padded

def train_sequence_model(model, train_loader, val_loader, num_epochs=10, 
                        learning_rate=0.001, max_grad_norm=1.0, weight_decay=1e-5,
                        model_save_path='./models/trained_model.pth'):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total_tokens = 0
        correct_tokens = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Создаем объединенную последовательность: вход + цель
            combined_sequence = torch.cat([X_batch, y_batch], dim=1)
            attention_mask = (combined_sequence != 0).long()
            
            # Прямой проход через модель
            outputs, _ = model(combined_sequence, attention_mask)
            
            # outputs: [batch_size, seq_len, vocab_size]
            # Нам нужны только выходы для позиций целевых токенов
            target_outputs = outputs[:, X_batch.size(1):X_batch.size(1) + y_batch.size(1), :]
            
            # Вычисляем loss только для целевых позиций
            loss = criterion(
                target_outputs.contiguous().view(-1, target_outputs.size(-1)),
                y_batch.contiguous().view(-1)
            )
            loss.backward()
            
            # Вычисляем accuracy
            with torch.no_grad():
                preds = torch.argmax(target_outputs, dim=-1)
                mask = (y_batch != 0)
                correct = (preds == y_batch) & mask
                
                correct_tokens += correct.sum().item()
                total_tokens += mask.sum().item()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            train_loss += loss.item() * mask.sum().item()
            
            if batch_idx % 50 == 0 and total_tokens > 0:
                batch_accuracy = correct.sum().item() / mask.sum().item()
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}')
        
        # Вычисляем средние метрики для эпохи
        avg_train_loss = train_loss / total_tokens if total_tokens > 0 else 0
        train_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Валидация
        model.eval()
        val_loss = 0
        val_tokens = 0
        val_correct = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Аналогично обучению
                combined_sequence = torch.cat([X_batch, y_batch], dim=1)
                attention_mask = (combined_sequence != 0).long()
                
                outputs, _ = model(combined_sequence, attention_mask)
                target_outputs = outputs[:, X_batch.size(1):X_batch.size(1) + y_batch.size(1), :]
                
                loss = criterion(
                    target_outputs.contiguous().view(-1, target_outputs.size(-1)),
                    y_batch.contiguous().view(-1)
                )
                
                # Вычисляем accuracy для валидации
                preds = torch.argmax(target_outputs, dim=-1)
                mask = (y_batch != 0)
                correct = (preds == y_batch) & mask
                
                val_correct += correct.sum().item()
                val_tokens += mask.sum().item()
                val_loss += loss.item() * mask.sum().item()
        
        avg_val_loss = val_loss / val_tokens if val_tokens > 0 else 0
        val_accuracy = val_correct / val_tokens if val_tokens > 0 else 0
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Сохраняем лучшую модель
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved - Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print(f'LR: {current_lr:.6f}')
        print('-' * 50)
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies, learning_rates