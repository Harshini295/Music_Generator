# =============================================================================
# Multi-Modal Feature Extractors
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
class MultiModalFeatureExtractor:
    """Extract features from text and images for cross-modal conditioning"""
    
    def __init__(self):
        # Load pre-trained models
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        
        # Set to eval mode
        self.roberta_model.eval()
        self.clip_model.eval()
        
    def extract_text_features(self, text: str) -> Tuple[np.ndarray, float, float]:
        """Extract RoBERTa embeddings and sentiment (valence/arousal)"""
        
        # Tokenize and encode
        inputs = self.roberta_tokenizer(text, return_tensors='pt', padding=True, 
                                       truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
            # Use [CLS] token embedding
            text_embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
            
        # Simple sentiment analysis (valence/arousal estimation)
        valence, arousal = self._estimate_emotion_from_text(text)
        
        return text_embedding, valence, arousal
    
    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        """Extract CLIP image embeddings"""
        
        inputs = self.clip_processor(images=image, return_tensors='pt')
        
        with torch.no_grad():
            image_embedding = self.clip_model.get_image_features(**inputs).numpy()[0]
            
        return image_embedding
    
    def _estimate_emotion_from_text(self, text: str) -> Tuple[float, float]:
        """Simple heuristic emotion estimation from text"""
        
        text = text.lower()
        
        # Positive/negative words for valence
        positive_words = ['happy', 'joy', 'love', 'beautiful', 'amazing', 'wonderful', 
                         'bright', 'peaceful', 'gentle', 'sweet', 'dream', 'hope']
        negative_words = ['sad', 'pain', 'dark', 'cold', 'lonely', 'cry', 'broken', 
                         'lost', 'fear', 'anger', 'hate', 'death']
        
        # Energy words for arousal
        high_arousal_words = ['fast', 'energy', 'power', 'loud', 'wild', 'dance', 
                             'exciting', 'intense', 'burst', 'rush']
        low_arousal_words = ['slow', 'calm', 'quiet', 'peaceful', 'gentle', 'soft', 
                            'rest', 'still', 'whisper']
        
        # Count word occurrences
        words = text.split()
        valence_score = 0
        arousal_score = 0.5  # Default neutral
        
        for word in words:
            if word in positive_words:
                valence_score += 0.1
            elif word in negative_words:
                valence_score -= 0.1
                
            if word in high_arousal_words:
                arousal_score += 0.1
            elif word in low_arousal_words:
                arousal_score -= 0.1
                
        # Normalize
        valence = np.clip(valence_score, -1, 1)
        arousal = np.clip(arousal_score, 0, 1)
        
        return valence, arousal
