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

# =============================================================================
# REMI Tokenization Implementation
# =============================================================================

import pretty_midi
import mido

# --- Safe MIDI Loader Functions ---
def safe_load_midi(path):
    """Try to safely load a MIDI file, skipping bad files."""
    try:
        return pretty_midi.PrettyMIDI(path)
    except Exception as e1:
        print(f"[pretty_midi failed] {path}: {e1}")
        try:
            return mido.MidiFile(path)
        except Exception as e2:
            print(f"[mido failed] {path}: {e2}")
            return None

def safe_tokenize(file_path, tokenizer, max_seq_len=512):
    try:
        # Convert path to string for pretty_midi
        pm = pretty_midi.PrettyMIDI(str(file_path))
        tokens = tokenizer.midi_to_tokens(str(file_path))

        # Make sure tokens are integers
        tokens = [int(t) for t in tokens]

        # Clip / pad
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        else:
            pad_id = tokenizer.vocab.get("<PAD>", 0)
            tokens.extend([pad_id] * (max_seq_len - len(tokens)))

        return tokens
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return None


class REMITokenizer:
    """REMI (REvamped MIDI) tokenization for symbolic music representation"""
    
    def __init__(self, max_velocity=127, velocity_bins=32, time_shift_bins=100, 
                 tempo_bins=32, max_bar=64):
        self.max_velocity = max_velocity
        self.velocity_bins = velocity_bins
        self.time_shift_bins = time_shift_bins
        self.tempo_bins = tempo_bins
        self.max_bar = max_bar
        
        # Build vocabulary
        self.vocab = {}
        self.id2token = {}
        self._build_vocabulary()
        
    def _build_vocabulary(self):
        """Build REMI vocabulary"""
        vocab_idx = 0
        
        # Special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        for token in special_tokens:
            self.vocab[token] = vocab_idx
            self.id2token[vocab_idx] = token
            vocab_idx += 1
            
        # Bar tokens
        for bar in range(self.max_bar):
            token = f'Bar_{bar}'
            self.vocab[token] = vocab_idx
            self.id2token[vocab_idx] = token
            vocab_idx += 1
            
        # Position tokens (16th note resolution)
        for pos in range(16):
            token = f'Position_{pos}'
            self.vocab[token] = vocab_idx
            self.id2token[vocab_idx] = token
            vocab_idx += 1
            
        # Pitch tokens (MIDI notes 0-127)
        for pitch in range(128):
            token = f'Pitch_{pitch}'
            self.vocab[token] = vocab_idx
            self.id2token[vocab_idx] = token
            vocab_idx += 1
            
        # Velocity tokens
        for vel_bin in range(self.velocity_bins):
            token = f'Velocity_{vel_bin}'
            self.vocab[token] = vocab_idx
            self.id2token[vocab_idx] = token
            vocab_idx += 1
            
        # Duration tokens (in 16th notes)
        for dur in range(1, 17):  # 1/16 to 1 bar
            token = f'Duration_{dur}'
            self.vocab[token] = vocab_idx
            self.id2token[vocab_idx] = token
            vocab_idx += 1
            
        # Tempo tokens
        for tempo_bin in range(self.tempo_bins):
            token = f'Tempo_{tempo_bin}'
            self.vocab[token] = vocab_idx
            self.id2token[vocab_idx] = token
            vocab_idx += 1
            
        self.vocab_size = vocab_idx
        
    def midi_to_tokens(self, midi_path: str) -> List[int]:
        """Convert MIDI file to REMI tokens"""
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            tokens = [self.vocab['<BOS>']]
            
            # Extract tempo changes
            tempo_changes = midi.get_tempo_changes()
            current_tempo = 120.0  # Default tempo
            
            # Process each instrument (focus on piano/melody)
            for instrument in midi.instruments:
                if instrument.is_drum:
                    continue
                    
                # Sort notes by start time
                notes = sorted(instrument.notes, key=lambda x: x.start)
                
                current_bar = 0
                current_position = 0
                last_time = 0
                
                for note in notes:
                    # Calculate bar and position
                    note_bar = int(note.start // (4 * 60 / current_tempo))
                    note_position = int((note.start % (4 * 60 / current_tempo)) / (60 / current_tempo / 4))
                    
                    # Limit to max bars
                    if note_bar >= self.max_bar:
                        break
                        
                    # Add bar token if changed
                    if note_bar != current_bar:
                        tokens.append(self.vocab[f'Bar_{note_bar}'])
                        current_bar = note_bar
                        
                    # Add position token if changed
                    if note_position != current_position:
                        tokens.append(self.vocab[f'Position_{note_position}'])
                        current_position = note_position
                        
                    # Add note tokens
                    tokens.append(self.vocab[f'Pitch_{note.pitch}'])
                    
                    # Velocity bin
                    vel_bin = min(int(note.velocity / self.max_velocity * self.velocity_bins), 
                                 self.velocity_bins - 1)
                    tokens.append(self.vocab[f'Velocity_{vel_bin}'])
                    
                    # Duration bin (in 16th notes)
                    duration_16th = max(1, min(16, int((note.end - note.start) / (60 / current_tempo / 4))))
                    tokens.append(self.vocab[f'Duration_{duration_16th}'])
                    
                break  # Focus on first non-drum instrument
                
            tokens.append(self.vocab['<EOS>'])
            return tokens
            
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return [self.vocab['<BOS>'], self.vocab['<EOS>']]
    
    def tokens_to_midi(self, tokens: List[int], output_path: str = None) -> pretty_midi.PrettyMIDI:
        """Convert REMI tokens back to MIDI"""
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        current_time = 0
        current_velocity = 64
        tempo = 120
        bar_duration = 4 * 60 / tempo
        sixteenth_duration = 60 / tempo / 4
        
        i = 0
        while i < len(tokens):
            token_id = tokens[i]
            if token_id >= len(self.id2token):
                i += 1
                continue
                
            token = self.id2token[token_id]
            
            if token.startswith('Bar_'):
                bar_num = int(token.split('_')[1])
                current_time = bar_num * bar_duration
                
            elif token.startswith('Position_'):
                pos = int(token.split('_')[1])
                bar_start = int(current_time // bar_duration) * bar_duration
                current_time = bar_start + pos * sixteenth_duration
                
            elif token.startswith('Pitch_'):
                pitch = int(token.split('_')[1])
                
                # Look for velocity and duration
                velocity = current_velocity
                duration = sixteenth_duration
                
                if i + 1 < len(tokens) and self.id2token[tokens[i + 1]].startswith('Velocity_'):
                    vel_bin = int(self.id2token[tokens[i + 1]].split('_')[1])
                    velocity = int(vel_bin * self.max_velocity / self.velocity_bins)
                    i += 1
                    
                if i + 1 < len(tokens) and self.id2token[tokens[i + 1]].startswith('Duration_'):
                    dur_bin = int(self.id2token[tokens[i + 1]].split('_')[1])
                    duration = dur_bin * sixteenth_duration
                    i += 1
                    
                # Create note
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=current_time,
                    end=current_time + duration
                )
                instrument.notes.append(note)
                current_time += duration
                
            i += 1
            
        midi.instruments.append(instrument)
        
        if output_path:
            midi.write(output_path)
            
        return midi