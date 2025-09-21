import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name="distilgpt2"):
    """Загрузка предобученной модели и токенизатора"""
    print("Загрузка модели и токенизатора ...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"Модель {model_name} загружена успешно на {device}!")
    return model, tokenizer, device

def autocomplete_text(model, tokenizer, device, input_text, 
                     max_length=1024, max_new_tokens=50, 
                     temperature=0.7, top_k=50, do_sample=True):
    """
    Генерация автодополнения для текста
    """
     # Токенизация с созданием attention mask
    inputs = tokenizer(
        input_text, 
        return_tensors='pt', 
        truncation=True, 
        max_length=max_length,
        padding=True
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Декодируем только сгенерированную часть (исключая исходный текст)
    generated_tokens = outputs[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Полный текст (исходный + сгенерированный)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return full_text, generated_text.strip()