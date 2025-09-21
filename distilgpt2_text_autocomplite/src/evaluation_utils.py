import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from tqdm import tqdm
import os
from .model_utils import autocomplete_text

def calculate_rouge_scores(references, predictions, metrics=['rouge1', 'rouge2'], use_stemmer=True):
    """
    Вычисление ROUGE метрик
    """
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=use_stemmer)
    
    scores_dict = {}
    for metric in metrics:
        scores_dict[metric] = {'precision': [], 'recall': [], 'f1': []}
    
    for ref, pred in zip(references, predictions):
        if not pred.strip():
            continue
            
        scores = scorer.score(ref, pred)
        
        for metric in metrics:
            scores_dict[metric]['precision'].append(scores[metric].precision)
            scores_dict[metric]['recall'].append(scores[metric].recall)
            scores_dict[metric]['f1'].append(scores[metric].fmeasure)
    
    # Усредняем метрики
    avg_scores = {}
    for metric in metrics:
        avg_scores[metric] = {k: np.mean(v) for k, v in scores_dict[metric].items()}
    
    return {
        'scores': avg_scores,
        'num_samples': len(scores_dict[metrics[0]]['f1'])
    }

def test_autocomplete_performance(model, tokenizer, device, test_cases, 
                                 num_samples=None, **generation_kwargs):
    """
    Тестирование производительности автодополнения
    """
    if num_samples and num_samples < len(test_cases):
        test_cases = test_cases[:num_samples]
    
    all_predictions = []
    all_references = []
    results = []
    
    print(f"Testing autocomplete on {len(test_cases)} samples...")
    
    for i, test_case in enumerate(tqdm(test_cases)):
        try:
            full_text, generated_text = autocomplete_text(
                model, tokenizer, device, test_case['input'], **generation_kwargs
            )
            
            result = {
                'input': test_case['input'],
                'expected': test_case['expected'],
                'generated': generated_text,
                'full_generated': full_text,
                'full_original': test_case['full_text'],
                'input_length': test_case['input_length'],
                'output_length': test_case['output_length']
            }
            
            results.append(result)
            all_predictions.append(generated_text)
            all_references.append(test_case['expected'])
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    return results, all_predictions, all_references

def print_results_analysis(results, rouge_scores):
    """Анализ и вывод результатов"""
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ АВТОДОПОЛНЕНИЯ")
    print("="*80)
    
    print(f"\nКоличество протестированных samples: {rouge_scores['num_samples']}")
    
    for metric, scores in rouge_scores['scores'].items():
        print(f"\n{metric.upper()} Metrics:")
        print(f"  Precision: {scores['precision']:.4f}")
        print(f"  Recall:    {scores['recall']:.4f}")
        print(f"  F1:        {scores['f1']:.4f}")
    
    # Статистика по длинам
    avg_input_len = np.mean([r['input_length'] for r in results])
    avg_output_len = np.mean([r['output_length'] for r in results])
    print(f"\nСредняя длина входного текста: {avg_input_len:.1f} слов")
    print(f"Средняя длина ожидаемого вывода: {avg_output_len:.1f} слов")
    
    print("\n" + "="*80)
    print("ПРИМЕРЫ РЕЗУЛЬТАТОВ:")
    print("="*80)
    
    for i, result in enumerate(results[:3]):
        print(f"\nПример {i+1}:")
        print(f"Вход: {result['input'][:100]}...")
        print(f"Ожидалось: {result['expected'][:100]}...")
        print(f"Сгенерировано: {result['generated'][:100]}...")
        print("-" * 50)

def plot_rouge_scores(rouge_scores, save_plot=True, results_dir="./results", plot_file="rouge_metrics_autocomplete.png"):
    """Визуализация ROUGE метрик"""

    metrics = list(rouge_scores['scores'].keys())
    
    # Подготовка данных для графика
    metric_names = ['Precision', 'Recall', 'F1']
    x = np.arange(len(metric_names))
    width = 0.8 / len(metrics)
    
    plt.figure(figsize=(12, 7))
    
    for i, metric in enumerate(metrics):
        offset = width * (i - (len(metrics) - 1) / 2)
        plt.bar(x + offset, [
            rouge_scores['scores'][metric]['precision'],
            rouge_scores['scores'][metric]['recall'],
            rouge_scores['scores'][metric]['f1']
        ], width, label=metric, alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('ROUGE Metrics for Text Autocompletion')
    plt.xticks(x, metric_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    if save_plot:
        # Создаем директорию если не существует
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, plot_file)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
    
    plt.show()

def save_results_to_csv(results, rouge_scores, save_results=True, 
                       results_dir="./results", 
                       results_file="autocomplete_results.csv", 
                       metrics_file="autocomplete_metrics.csv"):
    """Сохранение результатов в CSV файл"""
    if not save_results:
        print("Results saving is disabled")
        return
    
    # Создаем директорию если не существует
    os.makedirs(results_dir, exist_ok=True)
    
    # Сохраняем детальные результаты
    results_path = os.path.join(results_dir, results_file)
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False, encoding='utf-8')
    
    # Сохраняем метрики
    metrics_data = {'num_samples': rouge_scores['num_samples']}
    for metric, scores in rouge_scores['scores'].items():
        for score_type, value in scores.items():
            metrics_data[f'{metric}_{score_type}'] = value
    
    metrics_path = os.path.join(results_dir, metrics_file)
    metrics_df = pd.DataFrame([metrics_data])
    metrics_df.to_csv(metrics_path, index=False)
    
    print(f"Results saved to {results_path}")
    print(f"Metrics saved to {metrics_path}")