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
        
        # Embedding с инициализацией Xavier
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.init_embedding()
        
        # LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.init_lstm_weights()
        
        # Layer Normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Дополнительные слои
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
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
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
                elif 'bias_hh' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
    
    def init_fc_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x, attention_mask=None, hidden=None):
        batch_size, seq_len = x.size()
        
        if attention_mask is None:
            attention_mask = (x != self.pad_token_id)
        
        # Эмбеддинги
        embedded = self.embedding(x)
        
        # Pack padded sequences
        lengths = attention_mask.sum(dim=1).cpu()
        
        if torch.any(lengths == 0):
            dummy_output = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            output = self.fc2(self.dropout(self.activation(self.fc1(self.dropout(dummy_output)))))
            return output, None
        
        packed_embedded = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        packed_output, hidden = self.lstm(packed_embedded, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Выбор стратегии агрегации
        if self.use_mean_pooling:
            output = self.mean_pooling(output, attention_mask)
        else:
            last_valid_indices = (lengths - 1).to(x.device)
            output = output[torch.arange(batch_size, device=x.device), last_valid_indices]
        
        # Layer Normalization
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        # Дополнительные слои
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output, hidden
    
    def mean_pooling(self, output, attention_mask):
        if torch.is_tensor(attention_mask):
            mask = attention_mask.detach().clone().unsqueeze(-1).float()
        else:
            mask = torch.tensor(attention_mask, device=output.device).unsqueeze(-1).float()
    
        sum_output = torch.sum(output * mask, dim=1)
    
        if torch.is_tensor(attention_mask):
            valid_counts = attention_mask.detach().clone().sum(dim=1).unsqueeze(-1).float()
        else:
            valid_counts = torch.tensor(attention_mask, device=output.device).sum(dim=1).unsqueeze(-1).float()
    
        mean_output = sum_output / valid_counts.clamp(min=1.0)
        return mean_output
    
    def get_attention_mask(self, x):
        return (x != self.pad_token_id).long()
    
    def generate_next_token(self, input_sequence, hidden=None, temperature=1.0, top_k=None):
        self.eval()
        with torch.no_grad():
            attention_mask = self.get_attention_mask(input_sequence)
            output, new_hidden = self(input_sequence, attention_mask, hidden)
            logits = output / temperature
            
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[:, -1].unsqueeze(1)] = -float('Inf')
            
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            
            return next_token.item(), new_hidden, probabilities.cpu().numpy()
    
    def generate_sequence(self, initial_sequence, num_tokens=4, temperature=1.0, 
                         top_k=None, max_length=50):
        if isinstance(initial_sequence, list):
            input_seq = torch.tensor([initial_sequence], dtype=torch.long, device=self.device)
        elif torch.is_tensor(initial_sequence):
            input_seq = initial_sequence.detach().clone()
            if input_seq.dim() == 1:
                input_seq = input_seq.unsqueeze(0)
            input_seq = input_seq.to(self.device)
        else:
            try:
                input_seq = torch.tensor(initial_sequence, device=self.device)
                if input_seq.dim() == 1:
                    input_seq = input_seq.unsqueeze(0)
            except:
                raise ValueError("Неподдерживаемый тип входных данных")
        
        hidden = None
        generated_tokens = []
        all_probabilities = []
        
        for i in range(num_tokens):
            if input_seq.size(1) > max_length:
                input_seq = input_seq[:, -max_length:]
            
            next_token, hidden, probs = self.generate_next_token(
                input_seq, hidden, temperature, top_k
            )
            
            generated_tokens.append(next_token)
            all_probabilities.append(probs)
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            input_seq = torch.cat([input_seq, next_token_tensor], dim=1)
           
        full_sequence = input_seq.cpu().numpy().flatten().tolist()
        
        return {
            'full_sequence': full_sequence,
            'generated_tokens': generated_tokens,
            'probabilities': all_probabilities,
            'initial_sequence': initial_sequence
        }