import torch
import torch.nn as nn
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def postprocess_generated_text(text):
    """Улучшенная пост-обработка сгенерированного текста"""
    import re
    
    if not text:
        return text
    
    # Удаляем повторяющиеся слова (2+ повторений)
    text = re.sub(r'\b(\w+)( \1\b){2,}', r'\1', text)
    
    # Удаляем одиночные повторения слов
    text = re.sub(r'\b(\w+) \1\b', r'\1', text)
    
    # Удаляем повторяющиеся фразы
    words = text.split()
    if len(words) > 8:
        # Ищем повторяющиеся n-граммы (2-4 слова)
        for n in range(4, 1, -1):
            for i in range(len(words) - n * 2):
                ngram = words[i:i+n]
                next_ngram = words[i+n:i+2*n]
                if ngram == next_ngram:
                    # Удаляем повторяющуюся часть
                    text = ' '.join(words[:i+n] + words[i+2*n:])
                    words = text.split()
                    break
    
    # Удаляем лишние пробелы и trim
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def evaluate_sequence_model(model, test_loader, tokenizer, rouge_scorer_obj, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    total_tokens = 0
    
    rouge1_scores = []
    rouge2_scores = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Генерируем последовательности
            predictions = []
            for i in range(X_batch.size(0)):
                input_seq = X_batch[i]
                input_seq = input_seq[input_seq != 0]  # Убираем pad токены
                
                if len(input_seq) > 0:
                    generated = model.generate_sequence(
                        input_seq, 
                        max_length=len(y_batch[i]),
                        temperature=0.7,
                        top_k=30,
                        top_p=0.9,
                        repetition_penalty=1.5
                    )
                    predictions.append(generated)
                else:
                    predictions.append([])
            
            # Сохраняем предсказания и цели
            for i in range(len(predictions)):
                target_seq = y_batch[i][y_batch[i] != 0].cpu().numpy()
                pred_seq = predictions[i]
                
                # Декодируем в текст
                target_text = tokenizer.decode(target_seq, skip_special_tokens=True)
                pred_text = tokenizer.decode(pred_seq, skip_special_tokens=True)
                
                # ПРИМЕНЯЕМ ПОСТ-ОБРАБОТКУ - ДОБАВЬТЕ ЭТУ СТРОКУ
                pred_text = postprocess_generated_text(pred_text)
                
                # Вычисляем ROUGE
                scores = rouge_scorer_obj.score(target_text, pred_text)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                
                all_predictions.append(pred_text)
                all_targets.append(target_text)
    
    # Вычисляем средние метрики
    avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0
    
    metrics = {
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'total_samples': len(all_predictions)
    }
    
    return metrics, all_predictions, all_targets

def test_sequence_predictions(model, test_data, tokenizer, num_examples=5, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    
    indices = np.random.choice(len(test_data), min(num_examples, len(test_data)), replace=False)
    
    for i, idx in enumerate(indices):
        X_seq, y_seq = test_data[idx]
        
        # Генерируем последовательность с улучшенными параметрами
        generated = model.generate_sequence(
            X_seq, 
            max_length=min(30, len(y_seq) + 5),  # Ограничиваем длину
            temperature=0.6,      # Более детерминировано
            top_k=15,             # Ограничиваем выбор
            top_p=0.8,            # Nucleus sampling
            repetition_penalty=3.0  # Штраф за повторения
        )
        
        # Извлекаем только сгенерированную часть (исключаем вход)
        generated_tokens = generated[len(X_seq):]
        
        # Декодируем тексты
        input_text = tokenizer.decode(X_seq, skip_special_tokens=True)
        target_text = tokenizer.decode(y_seq, skip_special_tokens=True)
        pred_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        pred_text = postprocess_generated_text(pred_text)

        # Вычисляем ROUGE
        scores = rouge_scorer_obj.score(target_text, pred_text)
        
        print(f"\nПример {i+1}:")
        print(f"Входной текст: '{input_text}'")
        print(f"Целевой текст: '{target_text}'")
        print(f"Предсказанный текст: '{pred_text}'")
        print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
        print(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
        print("-" * 80)