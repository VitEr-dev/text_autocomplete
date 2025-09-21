import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_test_data(csv_path="./data/dataset_processed.csv", 
                     test_size=0.5, max_samples=1000, random_state=42):
    """Подготовка тестовых данных"""
    print("Загрузка очищенных тестовых данных...")
    
    df = pd.read_csv(csv_path)
    tweets = df['text'].dropna().tolist()
    texts = sorted(tweets, key=len, reverse=True)
    
    if len(texts) > max_samples:
        texts = texts[:max_samples]
    
    train_texts, test_texts = train_test_split(
        texts, test_size=test_size, random_state=random_state
    )
    
    print(f"Total texts: {len(texts)}")
    print(f"Test texts: {len(test_texts)}")
    
    return test_texts

def prepare_test_cases(texts, split_ratio=0.7, min_length=20):
    """
    Подготовка тестовых кейсов: разделяем тексты на input и expected output
    """
    test_cases = []
    skipped_too_short = 0
    
    for text in texts:
        if not text or not isinstance(text, str):
            continue
            
        words = text.split()
        
        if len(words) < min_length:
            skipped_too_short += 1
            continue
            
        split_index = int(len(words) * split_ratio)
        
        if split_index >= len(words) - 5:
            skipped_too_short += 1
            continue
        
        input_text = ' '.join(words[:split_index])
        expected_output = ' '.join(words[split_index:])
        
        test_cases.append({
            'input': input_text,
            'expected': expected_output,
            'full_text': text,
            'input_length': len(words[:split_index]),
            'output_length': len(words[split_index:])
        })
    
    print(f"Подготовлено {len(test_cases)} test cases")
    print(f"Пропущено {skipped_too_short} которких текстов")
    
    if test_cases:
        print(f"Средняя длинна исходных текстов: {np.mean([tc['input_length'] for tc in test_cases]):.1f} слов")
        print(f"Средняя длинна test cases: {np.mean([tc['output_length'] for tc in test_cases]):.1f} слов")
    
    return test_cases