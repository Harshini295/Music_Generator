# =============================================================================
# Cross-Modal Transformer Model
# =============================================================================
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
from pathlib import Path
import pretty_midi
import librosa
from transformers import RobertaTokenizer, RobertaModel, CLIPProcessor, CLIPModel
import streamlit as st
import tempfile
import io
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # This will give you a better error trace
from typing import List, Dict, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings("ignore")
class CrossModalMusicTransformer(nn.Module):
    """Cross-modal Transformer for music generation with image and text conditioning"""
    
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, 
                num_layers: int = 2, max_seq_len: int = 512):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Music token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Cross-modal conditioning projections
        self.emotion_projection = nn.Linear(2, d_model)  # valence + arousal
        self.image_projection = nn.Linear(512, d_model)  # CLIP embedding
        self.text_projection = nn.Linear(768, d_model)   # RoBERTa embedding
        
        # Context tokens (learnable)
        self.emotion_context_tokens = nn.Parameter(torch.randn(2, d_model))
        self.image_context_tokens = nn.Parameter(torch.randn(4, d_model))
        self.text_context_tokens = nn.Parameter(torch.randn(4, d_model))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Emotion alignment head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # valence + arousal
        )
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, tokens, emotion_features=None, image_features=None, 
                text_features=None, return_loss=True):
        
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # Clip sequence length if > max_seq_len
        if seq_len > self.max_seq_len:
            tokens = tokens[:, :self.max_seq_len]
            seq_len = self.max_seq_len
        # Safety check: ensure tokens are within embedding range
        if (tokens < 0).any() or (tokens >= self.token_embedding.num_embeddings).any():
            tmin = int(tokens.min().item())
            tmax = int(tokens.max().item())
            raise ValueError(f"Invalid token IDs in forward(): min={tmin}, max={tmax}, vocab_size={self.token_embedding.num_embeddings}")
        # Music token embeddings
        token_embeds = self.token_embedding(tokens)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(pos_ids)
        music_embeds = token_embeds + pos_embeds
        
        # Prepare context tokens
        context_embeds = []
        
        # Emotion conditioning
        if emotion_features is not None:
            emotion_proj = self.emotion_projection(emotion_features)  # [batch, d_model]
            emotion_contexts = self.emotion_context_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            emotion_contexts = emotion_contexts + emotion_proj.unsqueeze(1)
            context_embeds.append(emotion_contexts)
            
        # Image conditioning
        if image_features is not None:
            image_proj = self.image_projection(image_features)
            image_contexts = self.image_context_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            image_contexts = image_contexts + image_proj.unsqueeze(1)
            context_embeds.append(image_contexts)
            
        # Text conditioning
        if text_features is not None:
            text_proj = self.text_projection(text_features)
            text_contexts = self.text_context_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            text_contexts = text_contexts + text_proj.unsqueeze(1)
            context_embeds.append(text_contexts)
        
        # Concatenate context and music embeddings
        if context_embeds:
            context_embeds = torch.cat(context_embeds, dim=1)
            full_embeds = torch.cat([context_embeds, music_embeds], dim=1)
            context_len = context_embeds.shape[1]
        else:
            full_embeds = music_embeds
            context_len = 0
            
        full_embeds = self.dropout(self.layer_norm(full_embeds))
        
        # Create causal mask for music tokens only
        total_len = full_embeds.shape[1]
        causal_mask = torch.triu(torch.ones(total_len, total_len, device=device), diagonal=1).bool()
        
        # Allow attention to context tokens
        if context_len > 0:
            causal_mask[:, :context_len] = False
            
        # Transformer forward pass
        # Using encoder layers as decoder for simplicity
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        ).to(device)
        
        hidden_states = full_embeds
        for _ in range(8):  # num_layers
            hidden_states = transformer_layer(hidden_states, src_mask=causal_mask)
            
        # Extract music token representations
        if context_len > 0:
            music_hidden = hidden_states[:, context_len:, :]
        else:
            music_hidden = hidden_states
            
        # Output projections
        logits = self.output_projection(music_hidden)
        
        # Emotion alignment prediction
        pooled_hidden = music_hidden.mean(dim=1)  # Global average pooling
        emotion_pred = self.emotion_classifier(pooled_hidden)
        
        if return_loss:
            return logits, emotion_pred
        else:
            return logits
    
    def generate(self, context_tokens=None, emotion_features=None, image_features=None,
                 text_features=None, max_length=512, temperature=1.0, top_k=50):
        """Generate music tokens with conditioning"""
        
        self.eval()
        device = next(self.parameters()).device
        
        # Start with BOS token
        generated = torch.tensor([[1]], device=device)  # Assuming BOS token ID = 1
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.forward(
                    generated, 
                    emotion_features=emotion_features,
                    image_features=image_features, 
                    text_features=text_features,
                    return_loss=False
                )
                
                # Apply temperature and top-k sampling
                next_logits = logits[0, -1, :] / temperature
                
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                    next_logits = torch.full_like(next_logits, -float('inf'))
                    next_logits.scatter_(0, top_k_indices, top_k_logits)
                
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Stop if EOS token
                if next_token.item() == 3:  # Assuming EOS token ID = 3
                    break
                    
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
        return generated[0].cpu().tolist()
