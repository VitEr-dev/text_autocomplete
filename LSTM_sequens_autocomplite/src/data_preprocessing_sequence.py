import pandas as pd
import requests
import zipfile
import io
import re
import html
import numpy as np
from transformers import AutoTokenizer

def create_text_sequences(input_ids_list, train_ratio=0.75, min_length=20, max_length=80):
    """Создание последовательностей с улучшенной фильтрацией"""
    X_sequences = []
    y_sequences = []
    
    for sequence in input_ids_list:
        if len(sequence) < min_length:
            continue
            
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        
        split_point = int(len(sequence) * train_ratio)
        
        # Убедимся, что целевая последовательность имеет минимальную длину
        if len(sequence) - split_point < 5:  # Увеличьте минимум до 5 токенов
            continue
            
        X_seq = sequence[:split_point]
        y_seq = sequence[split_point:]
        
        # Сильная фильтрация проблемных последовательностей
        if (len(X_seq) > 0 and len(y_seq) > 0 and 
            (X_seq[-1] == y_seq[0] or  # Последний токен входа = первый токен цели
             len(set(y_seq)) < 3)):    # Цель содержит мало уникальных токенов
            continue
        
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
    
    print(f"Отфильтровано {len(input_ids_list) - len(X_sequences)} проблемных последовательностей")
    return X_sequences, y_sequences

def download_and_load_data(url):
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    
    with zip_file.open('training.1600000.processed.noemoticon.csv') as f:
        tweets_df = pd.read_csv(f, encoding='latin-1', header=None, 
                              names=['target', 'id', 'date', 'flag', 'user', 'text'])
    return tweets_df

def remove_smiles(text):
    """Удаление смайликов"""
    emoticon_pattern = re.compile(
        r'[:;=8xB][\-]?[\)\]\(\[dDpP/\:\*\-\}\{@\|\\<>3]|'
        r'[\)\]\(\[dDpP/\:\*\-\}\{@\|\\<>3][\-]?[:;=8xB]|'
        r'<3|</3|:\'[\)\(]|x[\(\)]|X[\(\)]'
    )
    return emoticon_pattern.sub('', text)

def clean_text(text):
    """Очистка текста"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = text.encode('latin-1').decode('utf-8', errors='ignore')
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = html.unescape(text)
    text = remove_smiles(text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = text.lower()
    text = re.sub(r'\s*\.{2,}\s*', '', text)
    text = re.sub(r'(\.\s*){2,}', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[#$%&*+<=>@^~/_]', '', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'[!?]{2,}', '!', text)
    text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
    text = re.sub(r'-{2,}', '', text)
    text = re.sub(r'[?!:\"\[\];,\(\).`\'-]', '', text)
    text = re.sub(r'[0123456789]', '', text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    return text

def preprocess_data(tweets_df, save_path='./data/'):
    """Полный процесс предобработки данных"""
    # Сохраняем сырые данные
    tweets_df[['text']].to_csv(save_path + 'raw_dataset.csv', index=False, encoding='latin-1')
    
    # Очищаем текст
    raw_df = tweets_df[['text']].copy()
    raw_df['cleaned_text'] = raw_df['text'].apply(clean_text)
    clean_df = raw_df[['cleaned_text']].copy()
    clean_df.columns = ['text']
    
    # Токенизация
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    clean_df['input_ids'] = clean_df['text'].apply(
        lambda x: tokenizer.encode(str(x), add_special_tokens=True, max_length=100, truncation=True) 
        if not pd.isna(x) else []
    )
    
    # Сохраняем обработанные данные
    clean_df.to_csv(save_path + 'dataset_processed.csv', index=False, encoding='utf-8')
    
    return clean_df, tokenizer