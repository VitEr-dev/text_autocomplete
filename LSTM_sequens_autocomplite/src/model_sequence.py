import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class EnhancedLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 pad_token_id=0, dropout=0.5, use_layer_norm=True, use_mean_pooling=False):
        super(EnhancedLSTMModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_mean_pooling = use_mean_pooling
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.init_embedding()
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.init_lstm_weights()
        
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.init_fc_weights()
    
    def init_embedding(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.pad_token_id is not None:
            with torch.no_grad():
                self.embedding.weight[self.pad_token_id].zero_()
    
    def init_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def init_fc_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x, attention_mask=None, hidden=None):
        batch_size, seq_len = x.size()
    
        if attention_mask is None:
            attention_mask = (x != self.pad_token_id)
    
        embedded = self.embedding(x)
    
        lengths = attention_mask.sum(dim=1).cpu()
    
        if torch.any(lengths == 0):
            # Возвращаем нули правильной размерности
            return torch.zeros(batch_size, seq_len, self.fc.out_features, device=x.device), None
    
        # Убедимся, что длины не превышают seq_len
        lengths = torch.clamp(lengths, max=seq_len)
    
        packed_embedded = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
    
        packed_output, hidden = self.lstm(packed_embedded, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)
    
        if self.use_layer_norm:
            output = self.layer_norm(output)
    
        output = self.dropout(output)
        output = self.fc(output)
    
        return output, hidden

    def generate_sequence(self, input_sequence, max_length=50, temperature=0.7, top_k=20, top_p=0.85, repetition_penalty=2.0):
        """Усиленная генерация с улучшенной защитой от повторений"""
        self.eval()
        device = next(self.parameters()).device
    
        if isinstance(input_sequence, list):
            input_tensor = torch.tensor([input_sequence], dtype=torch.long, device=device)
        else:
            input_tensor = input_sequence.unsqueeze(0) if input_sequence.dim() == 1 else input_sequence
            input_tensor = input_tensor.to(device)
    
        generated = input_tensor.clone()
        hidden = None
        previous_tokens = []
        repetition_count = 0
    
        with torch.no_grad():
            for step in range(max_length):
                attention_mask = (generated != self.pad_token_id).long()
                output, hidden = self(generated, attention_mask, hidden)
            
                # Берем последний токен
                next_token_logits = output[:, -1, :] / temperature
            
                # Сильный penalty за повторения последних 5 токенов
                if repetition_penalty != 1.0 and len(previous_tokens) > 0:
                    recent_tokens = set(previous_tokens[-5:])  # Последние 5 токенов
                    for token_id in recent_tokens:
                        next_token_logits[0, token_id] /= repetition_penalty
            
                # Top-k sampling
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < values[:, -1].unsqueeze(1)] = -float('Inf')
            
                # Top-p (nucleus) sampling
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
            
                # Запрещаем некоторые токены
                forbidden_tokens = [self.pad_token_id, 100]  # [PAD], [UNK]
                for token_id in forbidden_tokens:
                    next_token_logits[0, token_id] = -float('Inf')
            
                probabilities = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
                current_token = next_token.item()
            
                # Сильная защита от повторений
                if len(previous_tokens) > 3 and current_token in previous_tokens[-3:]:
                    repetition_count += 1
                    # Если повторяется слишком много раз, сильно штрафуем
                    next_token_logits[0, current_token] = -float('Inf')
                    probabilities = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, 1)
                    current_token = next_token.item()
                    repetition_count = 0
            
                previous_tokens.append(current_token)
                if len(previous_tokens) > 10:
                    previous_tokens = previous_tokens[-10:]
            
                generated = torch.cat([generated, next_token], dim=1)
            
                # Останавливаемся на стоп-токенах или при слишком многих повторениях
                if (current_token in [102, 0] or  # [SEP] и [PAD]
                    repetition_count > 5 or
                    step >= max_length - 1):
                    break
                # Отладочная печать:
                #if step % 5 == 0:
                    #current_text = tokenizer.decode(generated[0].cpu().numpy(), skip_special_tokens=True)
                    #print(f"Step {step}: {current_text}")
    
        return generated[0].cpu().numpy()