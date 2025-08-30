# =============================================================================
# Dataset Implementation
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
class MIDIDataset(Dataset):
    """PyTorch Dataset for MIDI files with emotion annotations"""
    
    def __init__(self, midi_dir: str, tokenizer: REMITokenizer, max_seq_len: int = 2048,
                emotion_labels_path: str = None):
        self.midi_dir = Path(midi_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Find all MIDI files
        all_midi_files = []
        for ext in ['*.mid', '*.midi']:
            all_midi_files.extend(self.midi_dir.rglob(ext))
        
        # Load emotion labels if available
        self.emotion_labels = {}
        if emotion_labels_path and os.path.exists(emotion_labels_path):
            with open(emotion_labels_path, 'r') as f:
                self.emotion_labels = json.load(f)
        else:
            self._generate_pseudo_labels_placeholder(all_midi_files)
        
        # Pre-tokenize and validate tokens so training won't crash on GPU
        self.examples = []
        skipped = 0
        unk_id = self.tokenizer.vocab.get('<UNK>')
        pad_id = self.tokenizer.vocab.get('<PAD>')
        vocab_size = self.tokenizer.vocab_size
        
        for midi_file in all_midi_files:
            try:
                # tokens = self.tokenizer.midi_to_tokens(str(midi_file))
                

                tokens = safe_tokenize(midi_file, self.tokenizer, self.max_seq_len)
                if tokens is None:
                    skipped += 1
                    continue


                # Validate token IDs are ints
                tokens = [int(t) for t in tokens]
                
                # Replace out-of-range token IDs with <UNK>
                if any((t < 0 or t >= vocab_size) for t in tokens):
                    # map invalid ids to <UNK>
                    tokens = [t if (0 <= t < vocab_size) else unk_id for t in tokens]
                
                # Truncate or pad sequence
                if len(tokens) > self.max_seq_len:
                    tokens = tokens[:self.max_seq_len]
                else:
                    tokens.extend([pad_id] * (self.max_seq_len - len(tokens)))
                
                file_key = str(midi_file.relative_to(self.midi_dir))
                emotion = self.emotion_labels.get(file_key, {'valence': 0.0, 'arousal': 0.5})
                
                self.examples.append({
                    'tokens': torch.tensor(tokens, dtype=torch.long),
                    'valence': float(emotion['valence']),
                    'arousal': float(emotion['arousal']),
                    'file_path': str(midi_file)
                })
            except Exception as e:
                skipped += 1
                # print warning but continue
                print(f"Warning: Skipping {midi_file} — tokenization error: {e}")
        
        self.midi_files = [ex['file_path'] for ex in self.examples]
        print(f"Found {len(self.midi_files)} valid MIDI files (skipped {skipped} bad files)")
        
    def _generate_pseudo_labels(self):
        """Generate pseudo emotion labels for MIDI files"""
        np.random.seed(42)
        for midi_file in self.midi_files:
            # Simple heuristic-based pseudo-labeling
            file_key = str(midi_file.relative_to(self.midi_dir))
            
            # Random but consistent labels for demo
            valence = np.random.normal(0, 0.5)  # -1 to 1
            arousal = np.random.uniform(0, 1)   # 0 to 1
            
            self.emotion_labels[file_key] = {
                'valence': float(np.clip(valence, -1, 1)),
                'arousal': float(arousal)
            }
    def _generate_pseudo_labels_placeholder(self, file_list):
        """Generate pseudo emotion labels when none provided — deterministic by filename hash"""
        np.random.seed(42)
        for midi_file in file_list:
            file_key = str(midi_file.relative_to(self.midi_dir))
            valence = np.random.normal(0, 0.5)
            arousal = np.random.uniform(0, 1)
            self.emotion_labels[file_key] = {
                'valence': float(np.clip(valence, -1, 1)),
                'arousal': float(arousal)
            }

    
    # def __len__(self):
    #     return len(self.midi_files)
    
    # def __getitem__(self, idx):
    #     midi_file = self.midi_files[idx]
    #     file_key = str(midi_file.relative_to(self.midi_dir))
        
    #     # Tokenize MIDI
    #     tokens = self.tokenizer.midi_to_tokens(str(midi_file))
        
    #     # Truncate or pad sequence
    #     if len(tokens) > self.max_seq_len:
    #         tokens = tokens[:self.max_seq_len]
    #     else:
    #         tokens.extend([self.tokenizer.vocab['<PAD>']] * (self.max_seq_len - len(tokens)))
            
    #     # Get emotion labels
    #     emotion = self.emotion_labels.get(file_key, {'valence': 0.0, 'arousal': 0.5})
        
    #     return {
    #         'tokens': torch.tensor(tokens, dtype=torch.long),
    #         'valence': torch.tensor(emotion['valence'], dtype=torch.float),
    #         'arousal': torch.tensor(emotion['arousal'], dtype=torch.float),
    #         'file_path': str(midi_file)
    #     }
    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, idx):
    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex['tokens']
        
        # Validate tokens
        unk_id = self.tokenizer.vocab.get('<UNK>', 1)
        vocab_size = self.tokenizer.vocab_size
        
        # Replace out-of-range tokens
        if (tokens < 0).any() or (tokens >= vocab_size).any():
            tokens = tokens.clone()
            tokens = torch.where((tokens < 0) | (tokens >= vocab_size), 
                            torch.tensor(unk_id, dtype=torch.long), tokens)
        
        return {
            'tokens': tokens,
            'valence': torch.tensor(ex['valence'], dtype=torch.float),
            'arousal': torch.tensor(ex['arousal'], dtype=torch.float),
            'file_path': ex['file_path']
        }
