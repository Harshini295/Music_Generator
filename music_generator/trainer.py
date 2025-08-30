# =============================================================================
# Training Loop
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
class MusicGeneratorTrainer:
    """Training pipeline for cross-modal music generation"""
    
    def __init__(self, model, tokenizer, feature_extractor, device='cuda'):
        # Auto-select device (GPU if available and has >1.5GB, else CPU)
        if device is None:
            if torch.cuda.is_available():
                try:
                    # use available memory to make a conservative decision
                    props = torch.cuda.get_device_properties(0)
                    total_mem_mb = int(props.total_memory / 1024**2)
                except Exception:
                    total_mem_mb = 0
                # if VRAM too small, prefer CPU to avoid repeated CUDA crashes
                device = "cuda" if total_mem_mb >= 4096 else "cuda" if total_mem_mb >= 2000 else "cpu"
            else:
                device = "cpu"

        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        # Loss weights
        self.ce_weight = 1.0
        self.emotion_weight = 0.5
        
    # def train_step(self, batch):
    #     """Single training step"""
        
    #     self.model.train()
    #     self.optimizer.zero_grad()
        
    #     tokens = batch['tokens'].to(self.device)
    #     valence = batch['valence'].to(self.device)
    #     arousal = batch['arousal'].to(self.device)
        
    #     # Prepare emotion features
    #     emotion_features = torch.stack([valence, arousal], dim=1)
        
    #     # Shift tokens for next-token prediction
    #     input_tokens = tokens[:, :-1]
    #     target_tokens = tokens[:, 1:]
        
    #     # Forward pass
    #     logits, emotion_pred = self.model(
    #         input_tokens, 
    #         emotion_features=emotion_features,
    #         return_loss=True
    #     )
        
    #     # Compute losses
    #     ce_loss = F.cross_entropy(
    #         logits.reshape(-1, logits.size(-1)),
    #         target_tokens.reshape(-1),
    #         ignore_index=self.tokenizer.vocab['<PAD>']
    #     )
        
    #     # Emotion alignment loss
    #     emotion_target = torch.stack([valence, arousal], dim=1)
    #     emotion_loss = F.mse_loss(emotion_pred, emotion_target)
        
    #     # Total loss
    #     total_loss = self.ce_weight * ce_loss + self.emotion_weight * emotion_loss
        
    #     # Backward pass
    #     total_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #     self.optimizer.step()
    #     self.scheduler.step()
        
    #     return {
    #         'total_loss': total_loss.item(),
    #         'ce_loss': ce_loss.item(),
    #         'emotion_loss': emotion_loss.item()
    #     }

    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()

        tokens = batch['tokens'].to(self.device)
        valence = batch['valence'].to(self.device)
        arousal = batch['arousal'].to(self.device)

        # Prepare emotion features
        emotion_features = torch.stack([valence, arousal], dim=1)

        # Shift tokens for next-token prediction
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]

        # Forward pass
        logits, emotion_pred = self.model(
            input_tokens,
            emotion_features=emotion_features,
            return_loss=True
        )

        # Align logits with targets
        if logits.size(1) != target_tokens.size(1):
            min_len = min(logits.size(1), target_tokens.size(1))
            logits = logits[:, :min_len, :]
            target_tokens = target_tokens[:, :min_len]

        # ---- Fix for invalid token IDs ----
        vocab_size = logits.size(-1)
        pad_id = self.tokenizer.vocab.get("<PAD>", 0)
        invalid = (target_tokens < 0) | (target_tokens >= vocab_size)
        if invalid.any():
            print(f"⚠️ Found {invalid.sum().item()} invalid token IDs, replacing with <PAD>")
            target_tokens = torch.where(
                (target_tokens >= 0) & (target_tokens < vocab_size),
                target_tokens,
                torch.tensor(pad_id, device=target_tokens.device)
            )

        # Compute losses
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
            ignore_index=pad_id
        )

        # Emotion alignment loss
        emotion_target = torch.stack([valence, arousal], dim=1)
        emotion_loss = F.mse_loss(emotion_pred, emotion_target)

        # Total loss
        total_loss = self.ce_weight * ce_loss + self.emotion_weight * emotion_loss

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'emotion_loss': emotion_loss.item()
        }

    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        
        epoch_losses = []
        progress_bar = tqdm.tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            losses = self.train_step(batch)
            epoch_losses.append(losses)
            
            # Update progress bar
            avg_loss = np.mean([l['total_loss'] for l in epoch_losses[-10:]])
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
        return epoch_losses
    
    def save_model(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tokenizer_state': {
                'vocab': self.tokenizer.vocab,
                'vocab_size': self.tokenizer.vocab_size
            }
        }
        torch.save(checkpoint, path)
        
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # restore tokenizer if needed
        if 'tokenizer_state' in checkpoint:
            self.tokenizer.vocab = checkpoint['tokenizer_state']['vocab']
            self.tokenizer.vocab_size = checkpoint['tokenizer_state']['vocab_size']

