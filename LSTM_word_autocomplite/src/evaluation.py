import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

def evaluate_model_on_test(model, test_loader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
            if len(X_batch) == 0:
                continue
                
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            attention_mask = model.get_attention_mask(X_batch)
            
            outputs, _ = model(X_batch, attention_mask)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
    
    metrics = {
        'test_loss': avg_loss,
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'total_samples': total_samples
    }
    
    return metrics, all_predictions, all_targets

def test_model_predictions(model, test_dataset, tokenizer, num_examples=5, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    indices = np.random.choice(len(test_dataset), min(num_examples, len(test_dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        X_sample, y_true = test_dataset[idx]
        X_sample = X_sample.unsqueeze(0).to(device)
        attention_mask = model.get_attention_mask(X_sample)
        
        with torch.no_grad():
            output, _ = model(X_sample, attention_mask)
            y_pred = torch.argmax(output, dim=1).item()
        
        print(f"\nПример {i+1}:")
        print(f"Входная последовательность: {tokenizer.decode(X_sample.cpu().numpy().flatten().tolist())}")
        print(f"Истинный следующий токен: {tokenizer.decode(y_true)}")
        print(f"Предсказанный токен: {tokenizer.decode(y_pred)}")
        print(f"Правильно: {'✓' if y_true == y_pred else '✗'}")
        
        probabilities = torch.softmax(output, dim=1)
        top5_probs, top5_tokens = torch.topk(probabilities, 5)
        
        print("Топ-5 предсказаний:")
        for j in range(5):
            print(f"  Токен {tokenizer.decode(top5_tokens[0][j].item())}: {top5_probs[0][j].item():.4f}")

def save_test_results(metrics, predictions, targets, filename='./results/test_results.csv'):
    # Извлекаем имя файла без пути для создания имени файла метрик
    import os
    base_name = os.path.basename(filename)
    metrics_filename = f'./results/metrics_{base_name}'
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_filename, index=False)
    
    results_df = pd.DataFrame({
        'true_token': targets,
        'predicted_token': predictions,
        'correct': [t == p for t, p in zip(targets, predictions)]
    })
    results_df.to_csv(filename, index=False)
    
    print(f"Результаты сохранены в {filename}")
    print(f"Метрики сохранены в {metrics_filename}")