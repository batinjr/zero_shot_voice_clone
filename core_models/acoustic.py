import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create an empty matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply Sine to even indices, Cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (it is not a trainable parameter, it's a fixed math property)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional math wave to the input x
        x = x + self.pe[:, :x.size(1), :]
        return x

class AcousticModelTier1(nn.Module):
    def __init__(self, vocab_size, text_embed_dim=256, speaker_embed_dim=256, mel_channels=80):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim)
        
        # WE ADDED POSITIONAL ENCODING HERE
        self.pos_encoder = PositionalEncoding(d_model=text_embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=text_embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.mel_linear = nn.Linear(text_embed_dim, mel_channels)

    def forward(self, text_tokens, speaker_embedding):
        # 1. Text to Vector
        x = self.text_embedding(text_tokens) 
        
        # 2. Inject Positional Encoding (Teach the sequence order)
        x = self.pos_encoder(x)
        
        # 3. Inject Voice DNA
        speaker_embedding = speaker_embedding.unsqueeze(1) 
        x = x + speaker_embedding 
        
        # 4. Pass through Transformer
        x = self.transformer(x)
        
        # 5. Output Mel-spectrogram
        mel_output = self.mel_linear(x) 
        
        return mel_output